"""Nightly regression evaluation for the agentic assistant.

    run_eval  ->  branch_on_degradation  ->  alert_degradation | report_ok

``run_eval`` re-runs the production prompt/config (results/production.json)
through the harness and the Evidently regression suite and logs a new MLflow run.
``branch_on_degradation`` is a BranchPythonOperator: if any watched metric fell
more than the threshold below the production baseline - or pct_tests_passed is
under the 80% gate - it routes to ``alert_degradation``, otherwise ``report_ok``.

All the logic lives in ``mlops.nightly`` and is unit-tested without Airflow.
This file only imports Airflow when it is installed, so it stays importable
(and the tests stay green) in the main environment. Airflow is an optional
dependency group in the same uv.lock, not installed by a plain `uv sync`:
`uv sync --group airflow` - see README "Orchestration".
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mlops import nightly  # noqa: E402

THRESHOLD = 0.10  # absolute drop in a rate (ten points) that counts as degradation


def _run_eval(**_):
    # NIGHTLY_DRY_RUN=1 compares the baseline with itself (no API calls) - used to
    # exercise the DAG wiring with `airflow dags test`. Add NIGHTLY_SIMULATE_DEGRADATION=1
    # to compare against a degraded copy instead, which takes the alert branch.
    dry_run = os.getenv("NIGHTLY_DRY_RUN") == "1"
    simulate = os.getenv("NIGHTLY_SIMULATE_DEGRADATION") == "1"
    return nightly.run_nightly(rate_threshold=THRESHOLD, dry_run=dry_run, simulate=simulate)  # goes to XCom


def _branch(ti=None, **_):
    return nightly.choose_branch(ti.xcom_pull(task_ids="run_eval"))


def _alert(ti=None, **_):
    return nightly.alert(ti.xcom_pull(task_ids="run_eval"))


try:  # Airflow 3 first, then Airflow 2 import paths
    try:
        from airflow.providers.standard.operators.empty import EmptyOperator
        from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
        from airflow.sdk import DAG
    except ImportError:
        from airflow import DAG
        from airflow.operators.empty import EmptyOperator
        from airflow.operators.python import BranchPythonOperator, PythonOperator
    AIRFLOW = True
except ImportError:
    AIRFLOW = False

if AIRFLOW:
    with DAG(
        dag_id="nightly_regression_eval",
        description="Harness + Evidently regression vs the production baseline, nightly",
        schedule="0 2 * * *",
        start_date=datetime(2026, 9, 1),
        catchup=False,
        default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
        tags=["mlops", "regression", "track-b"],
    ) as dag:
        run_eval = PythonOperator(task_id="run_eval", python_callable=_run_eval,
                                  execution_timeout=timedelta(hours=2))
        branch = BranchPythonOperator(task_id="branch_on_degradation", python_callable=_branch)
        alert = PythonOperator(task_id=nightly.ALERT_TASK, python_callable=_alert)
        ok = EmptyOperator(task_id=nightly.OK_TASK)
        run_eval >> branch >> [alert, ok]
