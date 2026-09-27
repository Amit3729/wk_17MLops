"""Nightly regression evaluation - the logic behind dags/nightly_regression_eval.py.

Plain Python on purpose: the DAG only wires these functions together, so the
decision it makes can be unit-tested without Airflow installed.

    uv run python -m mlops.nightly            # one nightly cycle, from a shell or cron
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .common import PRODUCTION_PATH, RESULTS_DIR, load_config

# Metrics that are compared with the production baseline, and the direction in
# which each one gets worse. Rates degrade downward, cost and latency upward.
WATCHED = {
    "pct_tests_passed": "higher_is_better",
    "task_completion_rate": "higher_is_better",
    "citation_hit_rate": "higher_is_better",
    "mean_tokens": "lower_is_better",
}
# Absolute drop allowed on a rate (0.10 = ten points), relative rise allowed on a cost.
DEFAULT_RATE_THRESHOLD = 0.10
DEFAULT_COST_THRESHOLD = 0.50
ALERT_TASK = "alert_degradation"
OK_TASK = "report_ok"


def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    rate_threshold: float = DEFAULT_RATE_THRESHOLD,
    cost_threshold: float = DEFAULT_COST_THRESHOLD,
    gate: float = 0.80,
) -> dict[str, Any]:
    """Which watched metrics degraded beyond threshold versus the production baseline."""
    reasons: list[str] = []
    deltas: dict[str, float] = {}
    for metric, direction in WATCHED.items():
        if metric not in current or metric not in baseline:
            continue
        now, before = float(current[metric]), float(baseline[metric])
        deltas[metric] = now - before
        if direction == "higher_is_better" and before - now > rate_threshold:
            reasons.append(f"{metric} fell {before:.2f} -> {now:.2f} (> {rate_threshold:.2f} allowed)")
        if direction == "lower_is_better" and before > 0 and (now - before) / before > cost_threshold:
            reasons.append(f"{metric} rose {before:,.0f} -> {now:,.0f} (> {cost_threshold:.0%} allowed)")
    if "pct_tests_passed" in current and float(current["pct_tests_passed"]) < gate:
        reasons.append(f"pct_tests_passed {float(current['pct_tests_passed']):.0%} is below the {gate:.0%} gate")
    return {"degraded": bool(reasons), "reasons": reasons, "deltas": deltas}


def choose_branch(comparison: dict[str, Any]) -> str:
    """The BranchPythonOperator callable's decision: which downstream task runs."""
    return ALERT_TASK if comparison.get("degraded") else OK_TASK


def load_production() -> dict[str, Any]:
    if not PRODUCTION_PATH.exists():
        raise SystemExit("no production baseline yet - promote a run first (python -m mlops.promote)")
    return json.loads(PRODUCTION_PATH.read_text(encoding="utf-8"))


def simulate_degradation(metrics: dict[str, float]) -> dict[str, float]:
    """A copy of the baseline with the drops a bad prompt change would cause (for testing the alert path)."""
    degraded = dict(metrics)
    for metric in ("pct_tests_passed", "task_completion_rate"):
        if metric in degraded:
            degraded[metric] = max(0.0, float(degraded[metric]) - 0.25)
    return degraded


def run_nightly(rate_threshold: float = DEFAULT_RATE_THRESHOLD, dry_run: bool = False,
                simulate: bool = False) -> dict[str, Any]:
    """Re-run the production prompt/config through the harness + regression suite and compare.

    ``dry_run`` makes no API calls: the baseline is compared with itself, or, with ``simulate``,
    with a copy degraded by 25 points, so both DAG branches can be exercised for free.
    """
    production = load_production()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    run_name = f"nightly-{production['prompt_version']}-{stamp}"
    if dry_run:
        current = dict(production["metrics"])
        if simulate:
            current = simulate_degradation(current)
            run_name += "-simulated"
    else:
        from .run_experiment import run

        cfg = production["config"]
        # The exact promoted config, every field, so the nightly run is the baseline's twin.
        config, prompt_text = load_config(
            cfg["prompt_version"], **{key: value for key, value in cfg.items() if key not in {"prompt_version"}}
        )
        meta = run(config, prompt_text, run_name)
        current = meta["harness"] | meta.get("regression", {})
    comparison = compare_to_baseline(current, production["metrics"], rate_threshold)
    outcome = {"run_name": run_name, "baseline": production["run_name"], **comparison,
               "branch": choose_branch(comparison)}
    out = RESULTS_DIR / "nightly"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_name}.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    return outcome


def alert(outcome: dict[str, Any]) -> str:
    """Stand-in for a pager/Slack hook: log loudly and leave a file an operator will see."""
    message = f"[REGRESSION] nightly {outcome['run_name']} degraded vs {outcome['baseline']}: " + "; ".join(
        outcome["reasons"]
    )
    print(message)
    (RESULTS_DIR / "nightly" / "ALERT.txt").write_text(message + "\n", encoding="utf-8")
    return message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=DEFAULT_RATE_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true", help="compare the baseline with itself (no API calls)")
    parser.add_argument("--simulate-degradation", action="store_true",
                        help="with --dry-run: compare against a copy degraded by 25 points (tests the alert path)")
    args = parser.parse_args()
    outcome = run_nightly(args.threshold, dry_run=args.dry_run, simulate=args.simulate_degradation)
    print(json.dumps(outcome, indent=2))
    if outcome["branch"] == ALERT_TASK:
        alert(outcome)


if __name__ == "__main__":
    main()
