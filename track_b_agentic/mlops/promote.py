"""Promote a run that passed the regression gate to the production baseline.

    uv run python -m mlops.promote --run-name <run_name>

Refuses a run whose Evidently ``pct_tests_passed`` is below the gate. Writes
``results/production.json`` (what the nightly DAG compares against) and tags the
MLflow run ``stage=production``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .common import PRODUCTION_PATH, RUNS_DIR
from .regression import GATE


def gate_allows(summary: dict[str, Any], gate: float = GATE) -> tuple[bool, str]:
    regression = summary.get("regression")
    if not regression:
        return False, "no regression result recorded for this run"
    pct = float(regression["pct_tests_passed"])
    if pct < gate:
        return False, f"pct_tests_passed {pct:.0%} is below the {gate:.0%} gate"
    return True, f"pct_tests_passed {pct:.0%} meets the {gate:.0%} gate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    summary = json.loads((RUNS_DIR / args.run_name / "summary.json").read_text(encoding="utf-8"))
    allowed, why = gate_allows(summary)
    if not allowed:
        raise SystemExit(f"not promoted: {why}")
    record = {
        "run_name": args.run_name,
        "mlflow_run_id": summary["mlflow_run_id"],
        "prompt_version": summary["config"]["prompt_version"],
        "config": summary["config"],
        "metrics": summary["harness"] | summary["regression"],
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": why,
    }
    PRODUCTION_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    from .run_experiment import setup_mlflow

    mlflow = setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    for run in mlflow.search_runs(filter_string="tags.stage = 'production'", output_format="list"):
        client.set_tag(run.info.run_id, "stage", "archived")
    client.set_tag(summary["mlflow_run_id"], "stage", "production")
    print(f"promoted {args.run_name}: {why}")


if __name__ == "__main__":
    main()
