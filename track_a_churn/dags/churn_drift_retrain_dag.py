"""Airflow DAG: weekly drift check on the latest batch; retrain only when the policy says so.

    check_drift ──► decide ──┬──► retrain_and_evaluate ──► notify
                             └──► no_drift_detected  ──► notify

* Schedule: weekly (Mondays 06:00 UTC), no catch-up.
* check_drift runs churn.drift.run_drift_check (Evidently data + target drift + custom metrics),
  logs the HTML reports to MLflow and pushes the summary to XCom.
* decide branches on summary["retrain_recommended"], computed by churn.drift.retrain_decision:
  >= 30% of features drifted, target drift, a failed custom test, or F1/ROC-AUC dropping by > 0.05.
* retrain_and_evaluate runs churn.pipeline.retrain (challenger vs champion); the challenger reaches
  Production only if it beats the champion on the recent hold-out, otherwise it stays in Staging.

The batch path defaults to the injected-drift set and can be overridden per run:
    airflow dags trigger telco_churn_drift_monitoring --conf '{"current_path": "data/current_clean.csv"}'
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airflow.sdk import DAG, get_current_context  # noqa: E402
from airflow.providers.standard.operators.empty import EmptyOperator  # noqa: E402
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator  # noqa: E402


def _current_path() -> Path:
    dag_run = get_current_context().get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}
    return PROJECT_ROOT / conf.get("current_path", "data/current_drifted.csv")


def check_drift(**_) -> dict:
    from churn.drift import run_drift_check

    summary = run_drift_check(_current_path(), tag="airflow")
    return {k: summary[k] for k in ("retrain_recommended", "reasons", "data_drift", "model_quality")}


def decide(ti, **_) -> str:
    summary = ti.xcom_pull(task_ids="check_drift")
    return "retrain_and_evaluate" if summary["retrain_recommended"] else "no_drift_detected"


def retrain_and_evaluate(ti, **_) -> dict:
    from churn.pipeline import retrain

    return retrain(_current_path(), ti.xcom_pull(task_ids="check_drift")["reasons"])


def notify(ti, **_) -> None:
    summary = ti.xcom_pull(task_ids="check_drift")
    result = ti.xcom_pull(task_ids="retrain_and_evaluate")
    print(f"drifted columns: {summary['data_drift']['drifted_columns']}")
    if result is None:
        print("No significant drift: Production model unchanged.")
    elif result["promoted"]:
        print(f"ALERT: drift -> retrained; v{result['challenger_version']} promoted to Production. Reasons: {summary['reasons']}")
    else:
        print(f"ALERT: drift -> retrained; v{result['challenger_version']} left in Staging for review. Reasons: {summary['reasons']}")


with DAG(
    dag_id="telco_churn_drift_monitoring",
    description="Weekly Evidently drift check with conditional retraining of the churn model",
    schedule="0 6 * * 1",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "track-a", "evidently", "mlflow"],
) as dag:
    t_check = PythonOperator(task_id="check_drift", python_callable=check_drift)
    t_decide = BranchPythonOperator(task_id="decide", python_callable=decide)
    t_retrain = PythonOperator(task_id="retrain_and_evaluate", python_callable=retrain_and_evaluate)
    t_skip = EmptyOperator(task_id="no_drift_detected")
    t_notify = PythonOperator(task_id="notify", python_callable=notify, trigger_rule="none_failed_min_one_success")

    t_check >> t_decide >> [t_retrain, t_skip] >> t_notify
