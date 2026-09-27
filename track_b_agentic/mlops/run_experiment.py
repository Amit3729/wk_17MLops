"""Run one prompt version end to end and track it in MLflow.

    uv run python -m mlops.run_experiment --prompt v1
    uv run python -m mlops.run_experiment --prompt v3 --temperature 0.8 --run-name v3-temp0.8

Per run this logs:

* **params**  - prompt_version, prompt file, provider, model, temperature, top_p,
  top_k, max_iterations, reasoning_effort, agent_mode, query-set size;
* **metrics** - harness metrics (task completion, tool-call correctness,
  citation/retrieval hit, iterations, tokens, latency, cost, termination
  counts) and the Evidently regression metrics (pct_tests_passed, ...);
* **artifacts** - the prompt text, every per-query trace, 2-3 representative
  traces (a success and a failure where both exist), the harness report and the
  Evidently HTML report;
* **MLflow traces** - one AGENT span per query with the OpenAI-client spans
  (via ``mlflow.openai.autolog``) nested under it.

Everything is also written to ``results/runs/<run_name>/`` so the numbers can
be read without MLflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from eval.dataset import select
from eval.harness import Result, render, summarise

from .common import (
    ARTIFACT_ROOT,
    CASE_IDS,
    EXPERIMENT_NAME,
    GOLDEN_PATH,
    ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRACES_DIR,
    TRACKING_URI,
    RunConfig,
    load_config,
    write_jsonl,
)
from .runner import run_cases


def setup_mlflow():
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def harness_metrics(traces: list[dict[str, Any]], config: RunConfig) -> dict[str, float]:
    results = [Result(**trace["harness"]) for trace in traces]
    summary = summarise(results)
    n = len(traces)
    latencies = sorted(trace["latency_ms"] for trace in traces)
    net = sorted(trace.get("latency_net_ms", trace["latency_ms"]) for trace in traces)
    terminations = Counter(trace["termination"] for trace in traces)
    prompt_tokens = sum(trace["usage"].get("prompt_tokens", 0) for trace in traces)
    completion_tokens = sum(trace["usage"].get("completion_tokens", 0) for trace in traces)
    answerable = [r for r, t in zip(results, traces) if not t["case"]["expect_decline"] and t["case"]["expect_documents"]]
    unanswerable = [r for r, t in zip(results, traces) if t["case"]["expect_decline"]]
    cost = sum(trace["cost_usd"] for trace in traces)
    return {
        "n_queries": n,
        "task_completion_rate": summary["completion_rate"],
        "tool_correctness_rate": summary["tool_correctness_rate"],
        # Citation/retrieval hit: the agent cited or itself retrieved the expected
        # documents (pre-retrieval alone does not count - see eval/harness.py).
        "citation_hit_rate": sum(r.grounded for r in answerable) / len(answerable) if answerable else 0.0,
        "decline_accuracy": sum(r.declined for r in unanswerable) / len(unanswerable) if unanswerable else 0.0,
        "mean_iterations": summary["mean_iterations"],
        "max_iterations_used": summary["max_iterations"],
        "mean_tokens": summary["mean_tokens"],
        "total_tokens": summary["total_tokens"],
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "tokens_in_per_query": prompt_tokens / n if n else 0.0,
        "tokens_out_per_query": completion_tokens / n if n else 0.0,
        # Per model turn: separates "more turns" from "bigger turns" when tokens move.
        "tokens_per_step": (prompt_tokens + completion_tokens) / max(1, sum(len(t["steps"]) for t in traces)),
        "mean_model_calls": statistics.mean(r.model_calls for r in results) if n else 0.0,
        "mean_latency_ms": summary["mean_latency_ms"],
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        # Net of rate-limit backoff (Groq free tier: 8K tokens/minute), which
        # otherwise dominates wall clock and says nothing about the prompt.
        "p50_latency_net_ms": _percentile(net, 0.50),
        "p95_latency_net_ms": _percentile(net, 0.95),
        "pct_terminated_max_iterations": terminations.get("max_iterations", 0) / n if n else 0.0,
        "pct_terminated_error": terminations.get("error", 0) / n if n else 0.0,
        "cost_usd_total": cost,
        "cost_usd_per_query": cost / n if n else 0.0,
        "termination_success": terminations.get("success", 0),
        "termination_max_iterations": terminations.get("max_iterations", 0),
        "termination_error": terminations.get("error", 0),
        "failures_hard": summary["hard"],
        "failures_soft": summary["soft"],
        "failures_cascading_soft": summary["cascading_soft"],
        "mean_confidence": statistics.mean(r.confidence for r in results) if n else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    if not values:
        return 0.0
    return float(values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))])


def representative(traces: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    """One success, one failure, then the most informative remaining trace."""
    ok = [t for t in traces if t["harness"]["completed"] and t["harness"]["tools_correct"]]
    bad = [t for t in traces if not (t["harness"]["completed"] and t["harness"]["tools_correct"])]
    picked: list[dict[str, Any]] = []
    if ok:
        picked.append(ok[0])
    if bad:
        picked.append(bad[0])
    unusual = [t for t in traces if t["termination"] != "success" and t not in picked]
    rest = unusual + [t for t in bad + ok if t not in picked and t not in unusual]
    picked.extend(rest[: max(0, k - len(picked))])
    return picked[:k]


def write_run_dir(run_dir: Path, run_name: str, traces: list[dict[str, Any]], settings) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "traces").mkdir(parents=True)
    for trace in traces:
        (run_dir / "traces" / f"{trace['case']['id']}.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    write_jsonl(run_dir / "responses.jsonl", [
        {
            "case_id": t["case"]["id"],
            "question": t["case"]["question"],
            "answer": t["final"]["answer"],
            "citations": t["final"]["citations"],
            "confidence": t["final"]["confidence"],
            "termination": t["termination"],
            "iterations": t["iterations"],
            "completed": t["harness"]["completed"],
            "failure": t["harness"]["failure"],
            "reason": t["harness"]["reason"],
        }
        for t in traces
    ])
    render({run_name: [Result(**t["harness"]) for t in traces]}, run_dir, settings)
    (run_dir / "report.md").rename(run_dir / "harness_report.md")
    (run_dir / "results.json").rename(run_dir / "harness_results.json")


def regression_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "pct_tests_passed": summary["pct_tests_passed"],
        "pct_cases_passed": summary["pct_cases_passed"],
        "judge_correctness_rate": summary["correctness_rate"],
        "judge_no_contradiction_rate": summary["no_contradiction_rate"],
        "judge_completeness_rate": summary["completeness_rate"],
        "regression_citation_rate": summary["citation_rate"],
        "evidently_tests_failed": summary["evidently_tests_failed"],
        "gate_passed": float(summary["gate_passed"]),
    }


def log_regression(mlflow, run_dir: Path, summary: dict[str, Any]) -> None:
    mlflow.log_metrics(regression_metrics(summary))
    mlflow.set_tag("gate", "passed" if summary["gate_passed"] else "failed")
    mlflow.set_tag("stage", "candidate" if summary["gate_passed"] else "rejected")
    mlflow.log_artifact(str(run_dir / "regression.json"), artifact_path="regression")
    mlflow.log_artifact(str(ROOT / summary["html"]), artifact_path="regression")
    # Provenance: a version graded against references reviewed from its own
    # answers scores itself (circular); later versions are graded honestly.
    sources = summary.get("golden_source_runs", [])
    mlflow.set_tag("golden_source_runs", ",".join(sources))
    run_name = run_dir.name
    if run_name in sources:
        mlflow.set_tag("regression_note", "circular: golden references were reviewed from this run's own answers")
    elif sources:
        mlflow.set_tag("regression_note", f"graded against references reviewed from {','.join(sources)}")


def log_regression_to_existing_run(run_dir: Path, summary: dict[str, Any]) -> None:
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    mlflow = setup_mlflow()
    with mlflow.start_run(run_id=meta["mlflow_run_id"]):
        log_regression(mlflow, run_dir, summary)
    meta["regression"] = regression_metrics(summary)
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(config: RunConfig, prompt_text: str, run_name: str, *, case_ids=CASE_IDS, regression: bool = True,
        tracing: bool = True) -> dict[str, Any]:
    cases = select(ids=tuple(case_ids))
    run_dir = RUNS_DIR / run_name
    mlflow = setup_mlflow()
    if tracing:
        mlflow.openai.autolog()

    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params(config.as_params() | {"n_queries": len(cases), "query_set": ",".join(c.id for c in cases)})
        mlflow.set_tags({"prompt_version": config.prompt_version, "agent_mode": config.agent_mode,
                         "model": config.model, "provider": config.provider})
        mlflow.log_text(prompt_text, f"prompt/{config.prompt_file}")

        print(f"\n== {run_name}: {len(cases)} queries, {config.provider}/{config.model} ==", flush=True)
        traces = asyncio.run(run_cases(cases, config, prompt_text, run_name, mlflow_spans=tracing,
                                       partial_dir=RESULTS_DIR / "partial" / run_name))

        from eval.harness import Stack

        write_run_dir(run_dir, run_name, traces, Stack(**config.settings_overrides(prompt_text)).settings)
        shutil.rmtree(RESULTS_DIR / "partial" / run_name, ignore_errors=True)
        metrics = harness_metrics(traces, config)
        from .runner import LAST_RUN_STATS

        if LAST_RUN_STATS.get("http_requests") is not None:
            metrics["http_requests"] = LAST_RUN_STATS["http_requests"]
        if LAST_RUN_STATS.get("remaining_requests_per_day") is not None:
            metrics["remaining_requests_per_day"] = LAST_RUN_STATS["remaining_requests_per_day"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_dir / "traces"), artifact_path="traces/all")
        mlflow.log_artifact(str(run_dir / "harness_report.md"), artifact_path="harness")
        mlflow.log_artifact(str(run_dir / "harness_results.json"), artifact_path="harness")
        mlflow.log_artifact(str(run_dir / "responses.jsonl"), artifact_path="harness")

        rep_dir = TRACES_DIR / run_name
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        rep_dir.mkdir(parents=True)
        for trace in representative(traces):
            verdict = "success" if trace["harness"]["completed"] and trace["harness"]["tools_correct"] else "failure"
            name = f"{verdict}__{trace['case']['id']}__{trace['termination']}.json"
            (rep_dir / name).write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        mlflow.log_artifacts(str(rep_dir), artifact_path="traces/representative")

        meta: dict[str, Any] = {
            "run_name": run_name,
            "mlflow_run_id": active.info.run_id,
            "config": config.as_params(),
            "harness": metrics,
        }
        (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if regression and GOLDEN_PATH.exists():
            from .regression import evaluate

            print("  running Evidently regression suite...", flush=True)
            summary = evaluate(traces, REPORTS_DIR / f"evidently_{run_name}.html")
            (run_dir / "regression.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            log_regression(mlflow, run_dir, summary)
            meta["regression"] = regression_metrics(summary)
            (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        elif regression:
            print(f"  no {GOLDEN_PATH.name} yet - skipping the regression suite", flush=True)
        mlflow.log_artifact(str(run_dir / "summary.json"))

    try:
        mlflow.flush_trace_async_logging()
    except Exception:
        pass
    printable = {k: round(v, 4) if isinstance(v, float) else v for k, v in (metrics | meta.get("regression", {})).items()}
    print(json.dumps(printable, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="registered prompt version, e.g. v1 (prompts/versions.json)")
    parser.add_argument("--mode", dest="agent_mode", choices=["single", "multi"], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ids", nargs="*", default=None, help="subset of the query set (debugging)")
    parser.add_argument("--no-regression", action="store_true")
    parser.add_argument("--no-tracing", action="store_true", help="skip MLflow span tracing")
    args = parser.parse_args()

    config, prompt_text = load_config(
        args.prompt,
        agent_mode=args.agent_mode,
        temperature=args.temperature,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
        reasoning_effort=args.reasoning_effort,
    )
    run(config, prompt_text, args.run_name or config.default_run_name(), case_ids=args.ids or CASE_IDS,
        regression=not args.no_regression, tracing=not args.no_tracing)


if __name__ == "__main__":
    main()
