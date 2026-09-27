"""Run the query set and keep a full, structured trace of every query.

The W16 harness scores a run but throws its trajectory away. Here every query
produces one JSON trace: the pre-retrieved context, each model turn (the
decision it made, the reasoning the provider exposed, the text it wrote), each
tool call with its arguments and raw result, token and latency deltas per step,
total iterations and the termination reason - ``success``, ``max_iterations``
or ``error``. Those traces are what each new prompt version is written against.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from typing import Any

from app.llm import AgentRunError, Completion
from eval.dataset import Case
from eval.harness import Result, Stack, score

from .common import RunConfig, cost_usd


# Stop a run while this many requests are still left in the daily quota (Groq's
# free tier allows 1,000 per day for gpt-oss-120b), so the judge and a rerun fit.
MIN_DAILY_REQUESTS = 60
# Request counters for the most recent run_cases call, logged as MLflow metrics.
LAST_RUN_STATS: dict[str, Any] = {}


class QuotaExhausted(RuntimeError):
    """The provider's per-day quota ran out mid-run; the run is not representative."""


def _daily_quota_text(error: str) -> bool:
    return any(marker in error for marker in ("tokens per day", "requests per day", "(TPD)", "(RPD)"))


def _prefetch_summary(prefetched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document": match.get("document"),
            "anchor": match.get("anchor"),
            "section": match.get("section"),
            "score": match.get("score"),
            "text": match.get("text"),
        }
        for match in prefetched
    ]


def build_trace(
    case: Case,
    config: RunConfig,
    run_name: str,
    completion: Completion | None,
    steps: list[dict[str, Any]],
    termination: str,
    error: str,
    prefetched: list[dict[str, Any]],
    result: Result,
    failed_usage: Any = None,
) -> dict[str, Any]:
    if completion is not None:
        usage = completion.usage.as_dict()
    else:
        usage = failed_usage.as_dict() if failed_usage is not None else {}
    return {
        "run_name": run_name,
        "prompt_version": config.prompt_version,
        "config": config.as_params(),
        "case": {
            "id": case.id,
            "kind": case.kind,
            "question": case.question,
            "expect_documents": list(case.expect_documents),
            "expect_tools": list(case.expect_tools),
            "forbid_tools": list(case.forbid_tools),
            "must_include": list(case.must_include),
            "expect_decline": case.expect_decline,
        },
        "prefetched_context": _prefetch_summary(prefetched),
        "steps": steps,
        "iterations": result.iterations,
        "termination": termination,
        "error": error,
        "final": {
            "answer": completion.answer if completion else "",
            "confidence": completion.confidence if completion else 0.0,
            "citations": completion.citations if completion else [],
            "agent_mode": completion.agent_mode if completion else config.agent_mode,
        },
        "usage": usage,
        "latency_ms": result.latency_ms,
        # Wall clock minus time slept in rate-limit backoff: the assistant's own latency.
        "latency_net_ms": max(0, result.latency_ms - usage.get("wait_ms", 0)),
        "cost_usd": cost_usd(config.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        "harness": asdict(result),
    }


def trace_tools(registry) -> None:
    """Wrap ``registry.execute`` so every tool call is a TOOL span in the MLflow trace.

    The span nests under the query's AGENT span (contextvars carry it through
    asyncio), next to the CHAT_MODEL spans ``mlflow.openai.autolog`` records, so
    the MLflow UI shows model turn -> tool call -> raw result in order.
    """
    import mlflow
    from mlflow.entities import SpanType

    inner = registry.execute

    async def execute(name: str, raw_arguments: str):
        with mlflow.start_span(name=f"tool:{name}", span_type=SpanType.TOOL) as span:
            span.set_inputs({"name": name, "arguments": raw_arguments})
            payload, matches = await inner(name, raw_arguments)
            span.set_outputs({"result": payload})
            return payload, matches

    registry.execute = execute


async def run_case(stack: Stack, case: Case, config: RunConfig) -> tuple[Completion | None, list, str, str, list, int, Any]:
    """Mirror ``/chat``: pre-retrieve, then hand off to the configured pattern."""
    started = time.perf_counter()
    prefetched: list[dict[str, Any]] = []
    try:
        try:
            matches = await asyncio.to_thread(stack.retriever.search, case.question)
        except Exception:
            matches = []
        prefetched = [m for m in matches if m["score"] >= stack.settings.min_score]
        context = "\n\n".join(f"[{m['document']}] {m['text']}" for m in prefetched)
        if config.agent_mode == "multi":
            completion = await stack.team.answer(case.question, context, config.temperature, config.top_p)
        else:
            completion = await stack.llm.complete(case.question, context, config.temperature, config.top_p)
            completion.agent_mode = "single"
        latency = int((time.perf_counter() - started) * 1000)
        return completion, completion.steps, completion.termination, "", prefetched, latency, None
    except Exception as error:  # the trace of a failed run is the one most worth keeping
        latency = int((time.perf_counter() - started) * 1000)
        steps = error.steps if isinstance(error, AgentRunError) else []
        usage = error.usage if isinstance(error, AgentRunError) else None
        return None, steps, "error", f"{type(error).__name__}: {error}", prefetched, latency, usage


async def run_cases(
    cases: list[Case],
    config: RunConfig,
    system_prompt: str,
    run_name: str,
    *,
    mlflow_spans: bool = False,
    pause_seconds: float = 0.0,
    partial_dir: Any = None,
) -> list[dict[str, Any]]:
    """Run every case sequentially and return one trace dict per case.

    Sequential on purpose: latency is then the query's own, not queueing behind
    its neighbours, and a tokens-per-minute rate limit is not tripped by a burst.
    """
    stack = Stack(agent_mode=config.agent_mode, **config.settings_overrides(system_prompt))
    if mlflow_spans:
        trace_tools(stack.registry)
    traces: list[dict[str, Any]] = []
    LAST_RUN_STATS.clear()
    for case in cases:
        span_cm = contextlib.nullcontext(None)
        if mlflow_spans:
            import mlflow
            from mlflow.entities import SpanType

            span_cm = mlflow.start_span(name=f"query:{case.id}", span_type=SpanType.AGENT)
        with span_cm as span:
            if span is not None:
                span.set_inputs({"question": case.question, "case_id": case.id, "prompt_version": config.prompt_version})
            completion, steps, termination, error, prefetched, latency, failed_usage = await run_case(stack, case, config)
            if error and _daily_quota_text(error):
                # A spent daily quota would score every remaining query as a hard
                # failure and log a meaningless run. Stop instead.
                raise QuotaExhausted(f"daily provider quota exhausted at {case.id}: {error[:300]}")
            result = score(case, completion, error, latency, config.agent_mode, "none", prefetched)
            if completion is None and failed_usage is not None:
                # A run that raised still spent tokens and turns; count them.
                result.prompt_tokens = failed_usage.prompt_tokens
                result.completion_tokens = failed_usage.completion_tokens
                result.total_tokens = failed_usage.total_tokens
                result.model_calls = failed_usage.model_calls
                result.iterations = sum(1 for step in steps if step.get("decision") != "error")
            trace = build_trace(case, config, run_name, completion, steps, termination, error, prefetched, result,
                                failed_usage)
            if span is not None:
                span.set_outputs({
                    "answer": trace["final"]["answer"],
                    "citations": trace["final"]["citations"],
                    "termination": termination,
                    "iterations": result.iterations,
                    "completed": result.completed,
                })
                span.set_attributes({
                    "termination": termination,
                    "iterations": result.iterations,
                    "harness.failure": result.failure or "none",
                    "prompt_tokens": trace["usage"].get("prompt_tokens", 0),
                    "completion_tokens": trace["usage"].get("completion_tokens", 0),
                    # The structured step records, so the UI shows the decision path
                    # (decision, reasoning, tool calls) without opening an artifact.
                    "steps": [
                        {key: step.get(key) for key in ("step", "agent", "decision", "reasoning", "error")}
                        | {"tools": [call["name"] for call in step.get("tool_calls", [])]}
                        for step in steps
                    ],
                })
        flag = "ok" if result.completed and result.tools_correct else (result.failure or "fail")
        print(
            f"  [{flag:>14}] {case.id:<24} {termination:<15} {result.iterations} turns "
            f"{result.total_tokens:>6} tok {latency / 1000:6.1f}s",
            flush=True,
        )
        traces.append(trace)
        if partial_dir is not None:  # survive an interrupted run: every finished query is on disk
            partial_dir.mkdir(parents=True, exist_ok=True)
            (partial_dir / f"{case.id}.json").write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
        pacer = stack.llm.pacer
        LAST_RUN_STATS.update(http_requests=pacer.requests, remaining_requests_per_day=pacer.remaining_requests)
        if pacer.remaining_requests is not None and pacer.remaining_requests < MIN_DAILY_REQUESTS:
            raise QuotaExhausted(
                f"only {pacer.remaining_requests} requests left in the provider's daily quota after {case.id}; "
                f"stopping before it runs out (floor {MIN_DAILY_REQUESTS})"
            )
        if pause_seconds:
            await asyncio.sleep(pause_seconds)
    return traces
