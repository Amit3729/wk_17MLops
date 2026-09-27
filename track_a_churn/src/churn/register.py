"""Pick the best training run, register it, and promote it None -> Staging -> Production.

Selection rule (config.SELECTION_METRIC): highest hold-out F1 on the churn class, tie-broken by ROC-AUC.
Promotion to Staging and then Production is gated: the model must load back from the registry and
reproduce its logged hold-out F1, and it must beat the current Production version (if any).

Both the classic stages (deprecated in MLflow 3 but still what the brief asks for) and the modern
aliases (@staging, @production) are set, so either URI style works:
    models:/telco-churn-classifier/Production
    models:/telco-churn-classifier@production
"""

from __future__ import annotations

import argparse
import json
import warnings

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score

from churn import config
from churn.data import prepare_splits, train_test
from churn.tracking import setup_mlflow

warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


def best_run(client: MlflowClient, experiment_id: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{config.SELECTION_METRIC} DESC", f"metrics.{config.TIEBREAK_METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No finished training runs: run `uv run churn-train` first.")
    return runs[0]


def _production_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None


def validate(model_uri: str, expected_f1: float) -> float:
    """Reload from the registry and re-score the hold-out: guards against a broken or mis-logged artifact."""
    _, X_test, _, y_test = train_test(prepare_splits()["reference"])
    model = mlflow.sklearn.load_model(model_uri)
    f1 = f1_score(y_test, model.predict(X_test))
    if abs(f1 - expected_f1) > 1e-6:
        raise SystemExit(f"Validation failed: registry model F1 {f1:.4f} != logged {expected_f1:.4f}")
    return f1


def promote(client: MlflowClient, version: str, stage: str, alias: str, reason: str) -> None:
    client.transition_model_version_stage(
        config.REGISTERED_MODEL_NAME, version, stage, archive_existing_versions=(stage == "Production")
    )
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, alias, version)
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, version, f"promoted_to_{stage.lower()}_because", reason)
    print(f"  version {version} -> {stage} (@{alias}): {reason}")


def export_registry(client: MlflowClient) -> dict:
    versions = []
    for found in client.search_model_versions(f"name = '{config.REGISTERED_MODEL_NAME}'"):
        mv = client.get_model_version(config.REGISTERED_MODEL_NAME, found.version)  # search omits aliases
        run = client.get_run(mv.run_id)
        versions.append(
            {
                "version": mv.version,
                "current_stage": mv.current_stage,
                "aliases": list(mv.aliases),
                "run_id": mv.run_id,
                "run_name": run.info.run_name,
                "source": mv.source,
                "tags": mv.tags,
                "test_f1": run.data.metrics.get("test_f1"),
                "test_roc_auc": run.data.metrics.get("test_roc_auc"),
            }
        )
    info = {"registered_model": config.REGISTERED_MODEL_NAME, "versions": sorted(versions, key=lambda v: int(v["version"]))}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "model_registry.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def register_best(run_id: str | None = None, force: bool = False) -> str:
    exp_id = setup_mlflow(config.EXPERIMENT_TRAINING)
    client = MlflowClient()
    run = client.get_run(run_id) if run_id else best_run(client, exp_id)
    f1 = run.data.metrics[config.SELECTION_METRIC]
    auc = run.data.metrics[config.TIEBREAK_METRIC]
    print(f"Best run: {run.info.run_name} ({run.info.run_id}) test_f1={f1:.4f} test_roc_auc={auc:.4f}")

    current = _production_version(client)
    if current is not None and not force:
        prod_f1 = client.get_run(current.run_id).data.metrics[config.SELECTION_METRIC]
        if current.run_id == run.info.run_id:
            print(f"Run already in Production as version {current.version}; nothing to do.")
            export_registry(client)
            return current.version
        if f1 <= prod_f1:
            print(f"Candidate F1 {f1:.4f} does not beat Production v{current.version} ({prod_f1:.4f}); not promoting.")
            export_registry(client)
            return current.version

    model_uri = run.data.tags.get("model_uri", f"runs:/{run.info.run_id}/model")
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME, tags={"source_run_name": run.info.run_name})
    client.update_registered_model(
        config.REGISTERED_MODEL_NAME,
        description=(
            "Telco customer churn classifier (sklearn Pipeline: scaler + one-hot + estimator). "
            f"Selected by highest hold-out {config.SELECTION_METRIC}, tie-break {config.TIEBREAK_METRIC}."
        ),
    )
    client.update_model_version(
        config.REGISTERED_MODEL_NAME,
        mv.version,
        description=f"{run.info.run_name}: test_f1={f1:.4f}, test_roc_auc={auc:.4f}, recall={run.data.metrics['test_recall']:.4f}",
    )
    print(f"Registered {config.REGISTERED_MODEL_NAME} version {mv.version}")

    promote(client, mv.version, "Staging", config.STAGING_ALIAS, f"highest {config.SELECTION_METRIC} among finished runs ({f1:.4f})")
    checked = validate(f"models:/{config.REGISTERED_MODEL_NAME}@{config.STAGING_ALIAS}", f1)
    promote(client, mv.version, "Production", config.PRODUCTION_ALIAS, f"staging validation reproduced hold-out F1 {checked:.4f}")
    export_registry(client)
    return mv.version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="register this run instead of the best one")
    parser.add_argument("--force", action="store_true", help="promote even if it does not beat Production")
    args = parser.parse_args(argv)
    register_best(args.run_id, args.force)


if __name__ == "__main__":
    main()
