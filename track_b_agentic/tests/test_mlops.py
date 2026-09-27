"""W17 MLOps layer: step tracing, prompt registry, regression plumbing, nightly logic.

No network: the model client is stubbed exactly as in test_llm.py, and nothing
here calls an LLM judge.
"""

import importlib
import json

import pytest

import httpx
import openai

from app.llm import SYSTEM_PROMPT, AgentRunError, _retry_after, assistant_turn
from mlops import nightly
from mlops.common import CASE_IDS, RunConfig, cost_usd, load_config, load_versions
from mlops.promote import gate_allows
from mlops.regression import build_frame, citation_ok
from mlops.run_experiment import representative

from test_llm import assistant, build, status_error, wire


# ------------------------------------------------------------ step tracing


async def test_trace_records_each_step_tool_call_and_success(settings):
    _, service = build(settings)
    final = json.dumps({"answer": "vLLM uses paged attention.", "confidence": 0.9, "citations": ["vllm.md"]})
    wire(service, [
        assistant(tool_calls=[("search_knowledge_base", json.dumps({"query": "vLLM"}))]),
        assistant(content=final),
    ])
    completion = await service.complete("What is vLLM?", "", 0.2, 0.9)

    assert completion.termination == "success"
    assert [step["decision"] for step in completion.steps] == ["call_tools", "final_answer"]
    call = completion.steps[0]["tool_calls"][0]
    assert call["name"] == "search_knowledge_base"
    assert call["arguments"] == {"query": "vLLM"}
    assert call["result"]["matches"], "the raw tool result is kept in the trace"
    assert completion.steps[1]["content"] == final


async def test_trace_marks_max_iterations_when_budget_is_spent(settings):
    _, service = build(settings, max_tool_iterations=2)
    search = ("search_knowledge_base", json.dumps({"query": "vLLM"}))
    final = json.dumps({"answer": "forced", "confidence": 0.4, "citations": []})
    wire(service, [assistant(tool_calls=[search]), assistant(tool_calls=[search]), assistant(content=final)])
    completion = await service.complete("q", "", 0.2, 0.9)

    assert completion.termination == "max_iterations"
    assert completion.steps[-1]["decision"] == "forced_final_answer"


async def test_error_keeps_the_partial_trajectory(settings):
    _, service = build(settings, max_retries=1, fallback_provider="none")
    wire(service, [assistant(tool_calls=[("calculator", json.dumps({"expression": "2+2"}))]), status_error(400)])
    try:
        await service.complete("q", "", 0.2, 0.9)
    except AgentRunError as error:
        assert error.steps and error.steps[0]["tool_calls"][0]["result"]["result"] == 4
    else:
        raise AssertionError("expected AgentRunError")


async def test_versioned_system_prompt_is_what_gets_sent(settings):
    _, service = build(settings, system_prompt="PROMPT V9")
    stub = wire(service, [assistant(content=json.dumps({"answer": "a", "confidence": 1, "citations": []}))])
    await service.complete("q", "", 0.2, 0.9)
    assert stub.requests[0]["messages"][0]["content"] == "PROMPT V9"


def test_assistant_turn_drops_provider_extras():
    message = assistant(content="x")
    message.reasoning = "private chain of thought"  # Groq's gpt-oss extra field
    assert set(assistant_turn(message)) <= {"role", "content", "tool_calls", "name"}


def test_retry_after_parses_groq_reset_formats():
    def error(headers):
        request = httpx.Request("POST", "http://stub")
        return openai.RateLimitError("429", response=httpx.Response(429, request=request, headers=headers), body=None)

    assert _retry_after(error({"retry-after": "7"})) == 7.0
    assert abs(_retry_after(error({"x-ratelimit-reset-tokens": "1m2.5s"})) - 62.5) < 1e-9
    assert _retry_after(error({})) == 0.0


# ---------------------------------------------------------- prompt registry


def test_prompt_v1_is_the_w15_prompt_verbatim():
    config, text = load_config("v1")
    assert text == SYSTEM_PROMPT.strip()
    assert config.model == "openai/gpt-oss-120b" and config.provider == "groq"


def test_every_registered_version_has_a_prompt_file():
    for version in load_versions():
        config, text = load_config(version)
        assert text, version


def test_overrides_pin_one_model_and_absolute_paths():
    overrides = RunConfig("v1", "prompt_v1.txt", top_k=7).settings_overrides("p")
    assert overrides["fallback_provider"] == "none"
    assert overrides["groq_model"] == "openai/gpt-oss-120b"
    assert overrides["top_k"] == 7 and overrides["chroma_dir"].startswith("/")


def test_cost_uses_published_price():
    assert abs(cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000) - 0.75) < 1e-9
    assert cost_usd("unknown-model", 10, 10) == 0.0


def test_query_set_is_within_brief():
    assert 8 <= len(CASE_IDS) <= 12


# --------------------------------------------------------------- regression


def _trace(case_id, *, citations=(), expect_documents=(), expect_decline=False, declined=False, completed=True):
    return {
        "run_name": "r",
        "case": {"id": case_id, "question": f"q-{case_id}", "expect_documents": list(expect_documents),
                 "expect_decline": expect_decline},
        "final": {"answer": "a", "citations": list(citations)},
        "harness": {"declined": declined, "completed": completed, "tools_correct": completed},
        "termination": "success",
    }


def test_citation_check():
    assert citation_ok(_trace("a", citations=["library/json.html#json.dumps"], expect_documents=["library/json.html"]))
    assert not citation_ok(_trace("b", citations=["serving.md"], expect_documents=["library/json.html"]))
    assert citation_ok(_trace("c", expect_decline=True, declined=True))
    assert not citation_ok(_trace("d", expect_decline=True, declined=False))


def test_build_frame_joins_on_case_id():
    golden = [{"case_id": "a", "reference": "ref a"}]
    frame = build_frame([_trace("a"), _trace("z")], golden)
    assert list(frame["case_id"]) == ["a"] and frame.loc[0, "reference"] == "ref a"


def test_representative_traces_include_success_and_failure():
    traces = [_trace("ok1"), _trace("bad1", completed=False), _trace("ok2")]
    picked = [t["case"]["id"] for t in representative(traces)]
    assert picked[:2] == ["ok1", "bad1"] and len(picked) == 3


def test_promotion_gate():
    assert gate_allows({"regression": {"pct_tests_passed": 0.8}})[0]
    assert not gate_allows({"regression": {"pct_tests_passed": 0.79}})[0]
    assert not gate_allows({})[0]


# ------------------------------------------------------------------ nightly


BASE = {"pct_tests_passed": 0.9, "task_completion_rate": 0.8, "citation_hit_rate": 0.8, "mean_tokens": 4000}


def test_nightly_ok_within_threshold():
    current = BASE | {"task_completion_rate": 0.75}
    comparison = nightly.compare_to_baseline(current, BASE, rate_threshold=0.10)
    assert not comparison["degraded"]
    assert nightly.choose_branch(comparison) == nightly.OK_TASK


def test_nightly_alerts_on_degradation():
    current = BASE | {"task_completion_rate": 0.6}
    comparison = nightly.compare_to_baseline(current, BASE, rate_threshold=0.10)
    assert comparison["degraded"] and nightly.choose_branch(comparison) == nightly.ALERT_TASK


def test_nightly_alerts_when_gate_fails_or_cost_explodes():
    assert nightly.compare_to_baseline(BASE | {"pct_tests_passed": 0.7}, BASE | {"pct_tests_passed": 0.75})["degraded"]
    assert nightly.compare_to_baseline(BASE | {"mean_tokens": 9000}, BASE)["degraded"]


def test_simulated_degradation_takes_the_alert_branch():
    degraded = nightly.simulate_degradation(BASE)
    assert degraded["pct_tests_passed"] == pytest.approx(0.65)
    comparison = nightly.compare_to_baseline(degraded, BASE)
    assert nightly.choose_branch(comparison) == nightly.ALERT_TASK
    assert nightly.choose_branch(nightly.compare_to_baseline(BASE, BASE)) == nightly.OK_TASK


def test_dag_module_imports_without_airflow():
    module = importlib.import_module("dags.nightly_regression_eval")
    assert module.THRESHOLD == 0.10


# ------------------------------------------- tracing: errors and multi mode


async def test_error_step_is_the_last_record(settings):
    _, service = build(settings, max_retries=1, fallback_provider="none")
    wire(service, [status_error(400)])
    try:
        await service.complete("q", "", 0.2, 0.9)
    except AgentRunError as error:
        assert error.steps[-1]["decision"] == "error"
        assert "400" in error.steps[-1]["error"] or "boom" in error.steps[-1]["error"]
    else:
        raise AssertionError("expected AgentRunError")


async def test_multi_mode_trace_covers_planner_researchers_and_synthesiser(settings):
    from test_agents import ANSWER, FINDINGS, PLAN_TWO, RoutingStub
    from test_agents import build as build_team
    from test_agents import wire as wire_team

    _, team, llm = build_team(settings)
    wire_team(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))
    completion = await team.answer("two things", "", 0.2, 0.9)

    agents = [step["agent"] for step in completion.steps]
    assert agents[0] == "planner" and agents[-1] == "synthesiser"
    assert sum(agent.startswith("researcher:") for agent in agents) >= 4  # 2 researchers x (search + report)
    searches = [call for step in completion.steps for call in step["tool_calls"]]
    assert searches and all("result" in call and "arguments" in call for call in searches)
    assert completion.termination == "success"


async def test_multi_mode_error_keeps_planner_and_researcher_steps(settings):
    from test_agents import ANSWER, FINDINGS, PLAN_TWO, RoutingStub
    from test_agents import build as build_team
    from test_agents import wire as wire_team

    _, team, llm = build_team(settings, fallback_provider="none", max_retries=1)
    stub = wire_team(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))
    inner = stub.chat.completions.create

    async def synthesiser_down(**kwargs):
        if kwargs["messages"][0]["content"].startswith("You write the final answer"):
            raise status_error(400)
        return await inner(**kwargs)

    stub.chat.completions.create = synthesiser_down
    try:
        await team.answer("two things", "", 0.2, 0.9)
    except AgentRunError as error:
        agents = [step["agent"] for step in error.steps]
        assert agents[0] == "planner" and any(a.startswith("researcher:") for a in agents)
        assert error.steps[-1]["decision"] == "error"
    else:
        raise AssertionError("expected AgentRunError")


# ------------------------------------------------ Groq wiring and pacing


def test_groq_target_never_combines_json_mode_with_tools(settings):
    from app.llm import LLMService
    from app.retrieval import Retriever
    from app.tools import ToolRegistry

    configured = settings.model_copy(update={"provider": "groq", "groq_api_key": "k"})
    target = LLMService(configured, ToolRegistry(Retriever(configured)))._primary()
    assert target.label == "groq" and target.model == "openai/gpt-oss-120b"
    assert target.json_mode_with_tools is False


async def test_pacer_waits_only_when_the_window_is_spent(monkeypatch):
    from app import llm as llm_module

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)
    pacer = llm_module.RatePacer()
    assert await pacer.wait_for(5000) == 0.0  # nothing observed yet
    pacer.observe({"x-ratelimit-remaining-tokens": "9000", "x-ratelimit-reset-tokens": "3s",
                   "x-ratelimit-remaining-requests": "950"})
    assert await pacer.wait_for(5000) == 0.0 and pacer.remaining_requests == 950
    pacer.observe({"x-ratelimit-remaining-tokens": "1200", "x-ratelimit-reset-tokens": "7.5s"})
    waited = await pacer.wait_for(5000)
    assert 7.0 < waited <= 8.5 and slept and pacer.requests == 2


def test_daily_quota_errors_are_not_retried():
    from app.llm import _daily_quota, _duration

    assert _daily_quota(RuntimeError("Rate limit reached ... on tokens per day (TPD): Limit 200000"))
    assert not _daily_quota(RuntimeError("Rate limit reached ... on tokens per minute (TPM)"))
    assert _duration("2m59.56s") == 2 * 60 + 59.56 and _duration("120ms") == 0.12 and _duration("1h2m") == 3720


def test_harness_metrics_report_percentiles_and_termination_shares():
    from mlops.run_experiment import harness_metrics

    def trace(latency, termination, completed):
        return {
            "harness": {"case_id": "c", "kind": "single_hop", "mode": "single", "fault": "none", "completed": completed,
                        "tools_correct": completed, "iterations": 2, "total_tokens": 100, "confidence": 0.5},
            "latency_ms": latency, "latency_net_ms": latency // 2, "termination": termination,
            "usage": {"prompt_tokens": 80, "completion_tokens": 20}, "cost_usd": 0.0,
            "case": {"expect_decline": False, "expect_documents": ["x"]}, "steps": [{}, {}],
        }

    traces = [trace(1000, "success", True), trace(2000, "max_iterations", False), trace(3000, "success", True),
              trace(4000, "error", False)]
    metrics = harness_metrics(traces, RunConfig(prompt_version="v1", prompt_file="prompt_v1.txt"))
    assert metrics["p50_latency_ms"] == 2000 and metrics["p95_latency_ms"] == 4000
    assert metrics["p50_latency_net_ms"] == 1000
    assert metrics["pct_terminated_max_iterations"] == 0.25 and metrics["pct_terminated_error"] == 0.25
    assert metrics["tokens_in_per_query"] == 80 and metrics["tokens_per_step"] == 50


def test_error_step_keeps_the_rejected_generation():
    from app.llm import error_step

    request = httpx.Request("POST", "http://stub/v1/chat/completions")
    body = {"error": {"code": "tool_use_failed",
                      "failed_generation": '{"name": "json", "arguments": {"answer": "x"}}'}}
    rejected = openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=body)
    try:
        try:
            raise rejected
        except openai.BadRequestError as inner:
            raise RuntimeError("groq request failed") from inner
    except RuntimeError as wrapped:
        record = error_step(2, "agent", wrapped)
    assert record["error_code"] == "tool_use_failed"
    assert record["failed_generation"]["name"] == "json"


def _tool_use_failed(generation: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "http://stub/v1/chat/completions")
    return openai.BadRequestError("tool call validation failed", response=httpx.Response(400, request=request),
                                  body={"code": "tool_use_failed", "failed_generation": generation})


async def test_pseudo_tool_answer_is_recovered_when_enabled(settings):
    _, service = build(settings, recover_tool_answers=True, fallback_provider="none", max_retries=1)
    envelope = {"answer": "64.58984375", "confidence": 1.0, "citations": []}
    wire(service, [
        assistant(tool_calls=[("calculator", json.dumps({"expression": "16535 * 4096 / 1048576"}))]),
        _tool_use_failed(json.dumps({"name": "json", "arguments": envelope})),
    ])
    completion = await service.complete("q", "", 0.2, 0.9)
    assert completion.answer == "64.58984375" and completion.termination == "success"
    assert completion.steps[-1]["recovered_from"].startswith("tool_use_failed")


async def test_pseudo_tool_answer_fails_when_disabled_or_a_real_tool(settings):
    _, service = build(settings, fallback_provider="none", max_retries=1)
    wire(service, [_tool_use_failed(json.dumps({"name": "json", "arguments": {"answer": "x"}}))])
    try:
        await service.complete("q", "", 0.2, 0.9)
    except AgentRunError as error:
        assert error.steps[-1]["error_code"] == "tool_use_failed"
    else:
        raise AssertionError("v1/v2 behaviour: the rejection must surface as an error")

    _, service = build(settings, recover_tool_answers=True, fallback_provider="none", max_retries=1)
    bad_args = json.dumps({"name": "calculator", "arguments": {"answer": "x"}})  # a real tool, malformed call
    wire(service, [_tool_use_failed(bad_args)])
    try:
        await service.complete("q", "", 0.2, 0.9)
    except AgentRunError:
        pass
    else:
        raise AssertionError("only non-existent tools carrying an answer are recovered")
