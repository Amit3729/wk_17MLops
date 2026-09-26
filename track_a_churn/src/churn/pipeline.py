"""End-to-end workflow and the retraining step.

    data -> training -> tracking -> registry -> (serving) -> monitoring -> retraining (if needed)

Usage:  uv run churn-pipeline               # everything, in order
        uv run churn-pipeline --retrain-only

Retraining ("challenger vs champion"):
  * new training data = reference + 70% of the labelled current batch; the other 30% of the current
    batch is a hold-out that represents the *new* distribution;
  * the challenger reuses the current Production model's configuration (same candidate name), retrained;
  * both challenger and champion are scored on that same recent hold-out, and the challenger is promoted
    Staging -> Production only if its F1 is higher and its ROC-AUC is not worse; otherwise it stays in Staging.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from churn import compare, config, drift, register, train
from churn.tracking import setup_mlflow

EXPERIMENT_RETRAINING = "telco-churn-retraining"


def retrain(current_path=config.CURRENT_DRIFTED_CSV, reasons: list[str] | None = None) -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    champion_mv = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS)
    champion_run = client.get_run(champion_mv.run_id)
    champion = mlflow.sklearn.load_model(f"models:/{config.REGISTERED_MODEL_NAME}@{config.PRODUCTION_ALIAS}")

    reference = pd.read_csv(config.REFERENCE_CSV)
    current = pd.read_csv(current_path)
    cur_train, cur_hold = train_test_split(current, test_size=0.3, stratify=current[config.TARGET], random_state=config.SEED)
    combined = pd.concat([reference, cur_train], ignore_index=True)
    X_train, y_train = combined[config.FEATURES], combined[config.TARGET]
    X_hold, y_hold = cur_hold[config.FEATURES], cur_hold[config.TARGET]

    champion_scores = train.evaluate(y_hold, champion.predict_proba(X_hold)[:, 1])
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cand = next(c for c in train.candidates(pos_weight) if c.name == champion_run.data.tags["candidate"])

    setup_mlflow(EXPERIMENT_RETRAINING)
    data_info = {
        "data_rows_train": len(X_train),
        "data_rows_test": len(X_hold),
        "data_churn_rate_train": round(float(y_train.mean()), 4),
        "data_source": "reference + 70% current batch; test = 30% current batch",
        "data_split_seed": config.SEED,
    }
    run_id = train.train_one(cand, X_train, X_hold, y_train, y_hold, data_info)
    client.log_metric(run_id, "champion_test_f1", champion_scores["f1"])
    client.log_metric(run_id, "champion_test_roc_auc", champion_scores["roc_auc"])
    client.set_tag(run_id, "champion_version", champion_mv.version)
    client.set_tag(run_id, "retrain_reasons", " | ".join(reasons or []))

    challenger = client.get_run(run_id).data.metrics
    wins = challenger["test_f1"] > champion_scores["f1"] and challenger["test_roc_auc"] >= champion_scores["roc_auc"]

    mv = mlflow.register_model(client.get_run(run_id).data.tags["model_uri"], config.REGISTERED_MODEL_NAME, tags={"source_run_name": f"retrain-{cand.name}"})
    client.update_model_version(
        config.REGISTERED_MODEL_NAME,
        mv.version,
        description=f"Retrained challenger ({cand.name}) on reference + recent batch; recent hold-out F1 {challenger['test_f1']:.4f}, ROC-AUC {challenger['test_roc_auc']:.4f}",
    )
    comparison = (
        f"recent hold-out F1 {challenger['test_f1']:.4f} vs champion v{champion_mv.version} {champion_scores['f1']:.4f}, "
        f"ROC-AUC {challenger['test_roc_auc']:.4f} vs {champion_scores['roc_auc']:.4f}"
    )
    register.promote(client, mv.version, "Staging", config.STAGING_ALIAS, f"retrained after drift: {comparison}")
    if wins:
        register.promote(client, mv.version, "Production", config.PRODUCTION_ALIAS, f"beat champion on recent hold-out: {comparison}")
    else:
        print(f"  challenger v{mv.version} stays in Staging: {comparison}")
    register.export_registry(client)

    result = {
        "challenger_version": mv.version,
        "champion_version": champion_mv.version,
        "promoted": wins,
        "challenger": {k: challenger[f"test_{k}"] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "champion": {k: champion_scores[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "reasons": reasons or [],
    }
    (config.RESULTS_DIR / "retraining_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def run_all() -> None:
    print("== 1-3. data -> train -> track (MLflow)")
    train.train_all([])
    print("== 4. registry: best run -> Staging -> Production")
    register.register_best()
    compare.export()
    print("== 5. monitoring (Evidently): control, then drifted batch")
    drift._print(drift.run_drift_check(config.CURRENT_CLEAN_CSV, "control"))
    summary = drift.run_drift_check(config.CURRENT_DRIFTED_CSV, "drifted")
    drift._print(summary)
    print("== 6. retraining decision")
    if summary["retrain_recommended"]:
        retrain(config.CURRENT_DRIFTED_CSV, summary["reasons"])
    else:
        print("No retraining needed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain-only", action="store_true")
    args = parser.parse_args(argv)
    if args.retrain_only:
        summary = json.loads((config.RESULTS_DIR / "drift_summary.json").read_text())
        retrain(config.CURRENT_DRIFTED_CSV, summary["reasons"])
    else:
        run_all()


if __name__ == "__main__":
    main()
