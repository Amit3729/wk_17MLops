"""Export the MLflow run comparison into the repo.

    uv run python -m mlops.export_runs

Writes ``results/mlflow_runs_comparison.csv`` (every param and metric) and
``results/mlflow_runs_comparison.md`` (the columns that matter, one row per run).
"""

from __future__ import annotations

import argparse

import pandas as pd

from .common import EXPERIMENT_NAME, RESULTS_DIR
from .run_experiment import setup_mlflow

TABLE = [
    ("run", "tags.mlflow.runName"),
    ("prompt", "params.prompt_version"),
    ("mode", "params.agent_mode"),
    ("temp", "params.temperature"),
    ("top_k", "params.top_k"),
    ("max_iter", "params.max_iterations"),
    ("recover", "params.recover_tool_answers"),
    ("completion", "metrics.task_completion_rate"),
    ("tool_ok", "metrics.tool_correctness_rate"),
    ("citation_hit", "metrics.citation_hit_rate"),
    ("decline_acc", "metrics.decline_accuracy"),
    ("pct_tests_passed", "metrics.pct_tests_passed"),
    ("no_contradict", "metrics.judge_no_contradiction_rate"),
    ("complete", "metrics.judge_completeness_rate"),
    ("judge_both", "metrics.judge_correctness_rate"),
    ("cites_check", "metrics.regression_citation_rate"),
    ("gate", "tags.gate"),
    ("mean_iter", "metrics.mean_iterations"),
    ("pct_max_iter", "metrics.pct_terminated_max_iterations"),
    ("pct_error", "metrics.pct_terminated_error"),
    ("tok_in/q", "metrics.tokens_in_per_query"),
    ("tok_out/q", "metrics.tokens_out_per_query"),
    ("p50_net_s", "metrics.p50_latency_net_ms"),
    ("p95_net_s", "metrics.p95_latency_net_ms"),
    ("p50_wall_s", "metrics.p50_latency_ms"),
    ("cost_usd", "metrics.cost_usd_total"),
    ("golden", "tags.regression_note"),
]

PERCENT = {"completion", "tool_ok", "citation_hit", "decline_acc", "pct_tests_passed", "judge_correct", "cites_check",
           "pct_max_iter", "pct_error"}


def fetch() -> pd.DataFrame:
    mlflow = setup_mlflow()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["attributes.start_time ASC"])
    return runs[runs["status"] == "FINISHED"] if "status" in runs else runs


def to_markdown(runs: pd.DataFrame) -> str:
    headers = [label for label, _ in TABLE]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in runs.iterrows():
        cells = []
        for label, column in TABLE:
            value = row.get(column)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("-")
            elif label in PERCENT:
                cells.append(f"{float(value):.0%}")
            elif label.endswith("_s"):
                cells.append(f"{float(value) / 1000:.1f}")
            elif label == "cost_usd":
                cells.append(f"{float(value):.4f}")
            elif label in {"tok_in/q", "tok_out/q"}:
                cells.append(f"{float(value):,.0f}")
            elif label in {"mean_iter"}:
                cells.append(f"{float(value):.1f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(RESULTS_DIR / "mlflow_runs_comparison"))
    args = parser.parse_args()
    runs = fetch()
    keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "tags.mlflow.runName", "tags.gate",
                                                     "tags.stage", "tags.regression_note")) or c in {"run_id", "start_time"}]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs[keep].to_csv(f"{args.out}.csv", index=False)
    table = to_markdown(runs)
    header = (
        "# MLflow run comparison\n\n"
        f"Exported from the `{EXPERIMENT_NAME}` experiment (sqlite:///mlflow.db) by `mlops.export_runs`. "
        "One row per finished run, in start order. Percentages are over the fixed 8-query set; "
        "`pct_tests_passed` is the Evidently regression suite (gate >= 80%). Latencies `*_net_s` exclude time "
        "slept waiting on Groq's rate limits; `p50_wall_s` includes it. Cost uses Groq's published "
        "gpt-oss-120b price ($0.15 / $0.60 per million input / output tokens) and covers the assistant's "
        "calls only, not the judge. `golden` says how the regression references relate to the run: all were "
        "reviewed from v3's answers, so v3's own score is circular.\n\n"
    )
    with open(f"{args.out}.md", "w", encoding="utf-8") as handle:
        handle.write(header + table + "\n")
    print(table)
    print(f"\nwrote {args.out}.csv and {args.out}.md ({len(runs)} runs)")


if __name__ == "__main__":
    main()
