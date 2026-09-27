"""Drift monitoring with Evidently, logged to MLflow.

reference        = data/reference.csv       (random 70%, the "training-time" data the model was built on)
current (drift)  = data/current_drifted.csv (remaining 30%, with synthetic drift injected)
current (clean)  = data/current_clean.csv   (the same 30% untouched: a control that should NOT alert)

Produces, under reports/:
  data_drift_report.html          DataDriftPreset, reference vs drifted current
  data_drift_control_report.html  DataDriftPreset, reference vs clean current (false-positive check)
  target_drift_report.html        target / prediction drift + the two custom metrics, with tests
  model_performance_report.html   ClassificationPreset on the reference hold-out vs drifted current
and results/drift_summary.json, then logs everything to the MLflow "telco-churn-monitoring" experiment.

Usage:  uv run churn-drift                       # drifted current (default)
        uv run churn-drift --current data/current_clean.csv --tag control
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently.tests import lte
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from churn import config
from churn.data import prepare_splits, train_test
from churn.monitoring_metrics import MeanDifference, SegmentRateShift
from churn.tracking import setup_mlflow

warnings.filterwarnings("ignore")

PRODUCTION_URI = f"models:/{config.REGISTERED_MODEL_NAME}@{config.PRODUCTION_ALIAS}"


def _definition(with_classification: bool = False) -> DataDefinition:
    kwargs = {}
    if with_classification:
        kwargs["classification"] = [
            BinaryClassification(target=config.TARGET, prediction_labels="prediction", prediction_probas="churn_proba")
        ]
    return DataDefinition(
        id_column=config.ID_COLUMN,
        numerical_columns=config.NUMERIC_FEATURES + ["churn_proba"],
        categorical_columns=config.CATEGORICAL_FEATURES + [config.TARGET, "prediction"],
        **kwargs,
    )


def score(model, df: pd.DataFrame) -> pd.DataFrame:
    """Attach the Production model's outputs, so prediction drift can be monitored even before labels arrive."""
    df = df.copy()
    df["churn_proba"] = model.predict_proba(df[config.FEATURES])[:, 1]
    df["prediction"] = (df["churn_proba"] >= 0.5).astype(int)
    return df


def _ds(df: pd.DataFrame, with_classification: bool = False) -> Dataset:
    return Dataset.from_pandas(df, data_definition=_definition(with_classification))


def data_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report([DataDriftPreset(columns=config.FEATURES, drift_share=config.DRIFT_SHARE_THRESHOLD)], include_tests=True)
    return report.run(_ds(current), _ds(reference))


def target_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report(
        [
            ValueDrift(column=config.TARGET),
            ValueDrift(column="prediction"),
            ValueDrift(column="churn_proba"),
            MeanDifference(column="MonthlyCharges", tests=[lte(config.MEAN_MONTHLY_CHARGES_DELTA_LIMIT)]),
            SegmentRateShift(target=config.TARGET, tests=[lte(config.SEGMENT_CHURN_SHIFT_LIMIT)]),
            SegmentRateShift(
                target=config.TARGET,
                segment_column="Contract",
                segment_value="Month-to-month",
                tests=[lte(config.SEGMENT_CHURN_SHIFT_LIMIT)],
            ),
        ],
        include_tests=True,
    )
    return report.run(_ds(current), _ds(reference))


def performance_report(reference_holdout: pd.DataFrame, current: pd.DataFrame):
    report = Report([ClassificationPreset()], include_tests=False)
    return report.run(_ds(current, True), _ds(reference_holdout, True))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def _status(status) -> str:
    return getattr(status, "value", str(status))


def summarise_data_drift(snapshot) -> dict:
    d = snapshot.dict()
    per_column = {}
    for m in d["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("ValueDrift"):
            per_column[cfg["column"]] = {"score": float(m["value"]), "method": cfg["method"], "threshold": cfg["threshold"]}
    for t in d["tests"]:
        match = re.match(r"Value Drift for column (.+)", t["name"])
        if match and match.group(1) in per_column:
            per_column[match.group(1)]["drifted"] = _status(t["status"]) == "FAIL"
    drifted = sorted(c for c, v in per_column.items() if v.get("drifted"))
    return {
        "n_features": len(per_column),
        "n_drifted": len(drifted),
        "share_drifted": len(drifted) / max(len(per_column), 1),
        "drifted_columns": drifted,
        "per_column": per_column,
    }


def summarise_tests(snapshot) -> dict:
    d = snapshot.dict()
    values = {m["metric_name"]: float(m["value"]) if isinstance(m["value"], (int, float)) else m["value"] for m in d["metrics"]}
    tests = [{"name": t["name"], "status": _status(t["status"]), "description": t["description"]} for t in d["tests"]]
    return {"metric_values": values, "tests": tests, "n_tests_failed": sum(t["status"] == "FAIL" for t in tests)}


def model_quality(df: pd.DataFrame) -> dict:
    return {
        "f1": f1_score(df[config.TARGET], df["prediction"]),
        "recall": recall_score(df[config.TARGET], df["prediction"]),
        "roc_auc": roc_auc_score(df[config.TARGET], df["churn_proba"]),
        "churn_rate": float(df[config.TARGET].mean()),
        "predicted_churn_rate": float(df["prediction"].mean()),
    }


def retrain_decision(data_drift: dict, target: dict, quality_ref: dict, quality_cur: dict) -> tuple[bool, list[str]]:
    """The retraining policy shared by the CLI and the Airflow DAG. Any one trigger is enough."""
    reasons = []
    if data_drift["share_drifted"] >= config.DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"{data_drift['n_drifted']}/{data_drift['n_features']} features drifted "
            f"(share {data_drift['share_drifted']:.2f} >= {config.DRIFT_SHARE_THRESHOLD})"
        )
    target_test = next((t for t in target["tests"] if t["name"] == f"Value Drift for column {config.TARGET}"), None)
    if target_test and target_test["status"] == "FAIL":
        reasons.append("target (Churn) distribution drifted")
    for t in target["tests"]:
        if t["status"] == "FAIL" and not t["name"].startswith("Value Drift"):
            reasons.append(f"custom test failed: {t['description']}")
    drop = quality_ref["f1"] - quality_cur["f1"]
    if drop > config.F1_DEGRADATION_LIMIT:
        reasons.append(f"F1 dropped {drop:.3f} (> {config.F1_DEGRADATION_LIMIT}) on labelled current data")
    # F1 depends on prevalence (it can rise just because churn rose); ROC-AUC does not, so check it separately.
    auc_drop = quality_ref["roc_auc"] - quality_cur["roc_auc"]
    if auc_drop > config.AUC_DEGRADATION_LIMIT:
        reasons.append(f"ROC-AUC dropped {auc_drop:.3f} (> {config.AUC_DEGRADATION_LIMIT}) on labelled current data")
    return bool(reasons), reasons


def run_drift_check(current_path: Path = config.CURRENT_DRIFTED_CSV, tag: str = "drifted", log_to_mlflow: bool = True) -> dict:
    splits = prepare_splits()
    reference = splits["reference"]
    current = pd.read_csv(current_path)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(PRODUCTION_URI)
    prod = mlflow.MlflowClient().get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS)

    ref_scored = score(model, reference)
    cur_scored = score(model, current)
    _, X_hold, _, y_hold = train_test(reference)
    holdout = score(model, reference.loc[X_hold.index])

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if tag == "drifted" else f"_{tag}"
    paths = {
        "data_drift": config.REPORTS_DIR / f"data_drift{suffix}_report.html",
        "target_drift": config.REPORTS_DIR / f"target_drift{suffix}_report.html",
        "performance": config.REPORTS_DIR / f"model_performance{suffix}_report.html",
    }
    dd_snap = data_drift_report(ref_scored, cur_scored)
    td_snap = target_drift_report(ref_scored, cur_scored)
    perf_snap = performance_report(holdout, cur_scored)
    dd_snap.save_html(str(paths["data_drift"]))
    td_snap.save_html(str(paths["target_drift"]))
    perf_snap.save_html(str(paths["performance"]))

    data_drift = summarise_data_drift(dd_snap)
    target = summarise_tests(td_snap)
    q_ref, q_cur = model_quality(holdout), model_quality(cur_scored)
    retrain, reasons = retrain_decision(data_drift, target, q_ref, q_cur)

    summary = {
        "scenario": tag,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "current_path": str(Path(current_path).relative_to(config.PROJECT_ROOT)) if Path(current_path).is_absolute() else str(current_path),
        "production_model": {"version": prod.version, "run_id": prod.run_id},
        "data_drift": data_drift,
        "target_and_custom": target,
        "model_quality": {"reference_holdout": q_ref, "current": q_cur},
        "retrain_recommended": retrain,
        "reasons": reasons,
        "reports": {k: str(p.relative_to(config.PROJECT_ROOT)) for k, p in paths.items()},
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"drift_summary{suffix}.json").write_text(json.dumps(summary, indent=2, default=str))

    if log_to_mlflow:
        _log(summary, paths)
    return summary


def _log(summary: dict, paths: dict[str, Path]) -> None:
    setup_mlflow(config.EXPERIMENT_MONITORING)
    dd, tgt, q = summary["data_drift"], summary["target_and_custom"], summary["model_quality"]
    with mlflow.start_run(run_name=f"drift-check-{summary['scenario']}"):
        mlflow.set_tags(
            {
                "scenario": summary["scenario"],
                "production_model_version": summary["production_model"]["version"],
                "retrain_recommended": str(summary["retrain_recommended"]),
                "drifted_columns": ",".join(dd["drifted_columns"]) or "none",
            }
        )
        mlflow.log_params(
            {
                "current_path": summary["current_path"],
                "reference_rows": summary["reference_rows"],
                "current_rows": summary["current_rows"],
                "drift_share_threshold": config.DRIFT_SHARE_THRESHOLD,
                "injected_monthly_charges_offset": str(config.DRIFT_MONTHLY_CHARGES_OFFSET),
                "injected_month_to_month_share": config.DRIFT_MONTH_TO_MONTH_SHARE,
                "injected_label_flip_rate": config.DRIFT_LABEL_FLIP_RATE,
            }
        )
        metrics = {
            "n_drifted_features": dd["n_drifted"],
            "share_drifted_features": dd["share_drifted"],
            "n_custom_tests_failed": tgt["n_tests_failed"],
            "retrain_recommended": int(summary["retrain_recommended"]),
            **{f"drift_score_{_slug(c)}": v["score"] for c, v in dd["per_column"].items()},
            **{f"ref_holdout_{k}": v for k, v in q["reference_holdout"].items()},
            **{f"current_{k}": v for k, v in q["current"].items()},
        }
        for name, value in tgt["metric_values"].items():
            if isinstance(value, (int, float)):
                metrics[_slug(name)[:200]] = float(value)
        mlflow.log_metrics(metrics)
        for p in paths.values():
            mlflow.log_artifact(str(p), artifact_path="evidently_reports")
        mlflow.log_dict(summary, "drift_summary.json")


def _print(summary: dict) -> None:
    dd = summary["data_drift"]
    print(f"[{summary['scenario']}] drifted features {dd['n_drifted']}/{dd['n_features']}: {dd['drifted_columns']}")
    for c, v in sorted(dd["per_column"].items(), key=lambda kv: -kv[1]["score"])[:8]:
        print(f"    {c:<18} {v['method']:<32} score={v['score']:.3f} threshold={v['threshold']} drifted={v.get('drifted')}")
    for t in summary["target_and_custom"]["tests"]:
        print(f"  {t['status']:<7} {t['description']}")
    q = summary["model_quality"]
    print("  quality ref hold-out:", {k: round(v, 3) for k, v in q["reference_holdout"].items()})
    print("  quality current     :", {k: round(v, 3) for k, v in q["current"].items()})
    print(f"  retrain_recommended={summary['retrain_recommended']}")
    for r in summary["reasons"]:
        print("   -", r)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", type=Path, help="current-data CSV (default: injected-drift set)")
    parser.add_argument("--tag", default=None, help="scenario name used in file names and the MLflow run")
    parser.add_argument("--both", action="store_true", help="run the clean control and the drifted set")
    args = parser.parse_args(argv)

    if args.both or (args.current is None and args.tag is None):
        control = run_drift_check(config.CURRENT_CLEAN_CSV, "control")
        _print(control)
        _print(run_drift_check(config.CURRENT_DRIFTED_CSV, "drifted"))
        return
    _print(run_drift_check(args.current or config.CURRENT_DRIFTED_CSV, args.tag or "drifted"))


if __name__ == "__main__":
    main()
