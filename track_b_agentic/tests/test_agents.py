"""The multi-agent research mode: compaction, isolation, and degradation.

The upstream client is a router rather than a script: researchers run
concurrently, so a fixed replay order would be flaky.
"""

import json

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from app.agents import ResearchTeam, compact_history
from app.config import Settings
from app.llm import LLMService
from app.retrieval import Retriever
from app.tools import ToolRegistry

MARKER = "paged attention"  # a phrase that only exists in the indexed chunk text


def assistant(content=None, tool_calls=None):
    calls = [
        ChatCompletionMessageFunctionToolCall(id=f"c{index}", type="function", function=Function(name=name, arguments=args))
        for index, (name, args) in enumerate(tool_calls or [])
    ]
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=calls or None)


class RoutingStub:
    """Answers by role, inferred from the system prompt, and records requests."""

    def __init__(self, plan, findings, answer, researcher_searches=1):
        self.plan, self.findings, self.answer = plan, findings, answer
        self.searches = researcher_searches
        self.requests = []
        outer = self

        class Completions:
            async def create(self, **kwargs):
                outer.requests.append(kwargs)
                system = kwargs["messages"][0]["content"]
                if system.startswith("You plan documentation research"):
                    message = assistant(content=json.dumps(outer.plan))
                elif system.startswith("You are a documentation researcher"):
                    done = sum(1 for m in kwargs["messages"] if m.get("role") == "tool")
                    if done < outer.searches and "tools" in kwargs:
                        message = assistant(tool_calls=[("search_knowledge_base", json.dumps({"query": "vLLM"}))])
                    else:
                        message = assistant(content=json.dumps(outer.findings))
                else:
                    message = assistant(content=json.dumps(outer.answer))
                return type("R", (), {"choices": [type("C", (), {"message": message})()], "usage": None})()

        self.chat = type("Chat", (), {"completions": Completions()})()

    def prompts_for(self, prefix):
        return [r for r in self.requests if r["messages"][0]["content"].startswith(prefix)]


def build(settings: Settings, **overrides):
    settings = settings.model_copy(update={"provider": "openai", "openai_api_key": "k", **overrides})
    retriever = Retriever(settings)
    retriever.ingest()
    registry = ToolRegistry(retriever, settings)
    llm = LLMService(settings, registry)
    return settings, ResearchTeam(settings, registry, llm), llm


def wire(llm, stub):
    llm._client = lambda target: stub
    return stub


PLAN_TWO = {"tasks": [
    {"question": "What is vLLM?", "source": ""},
    {"question": "What is RAG?", "source": ""},
]}
FINDINGS = {"findings": [{"claim": "vLLM raises throughput.", "citation": "vllm.md"}], "gaps": ""}
ANSWER = {"answer": "vLLM serves LLMs at high throughput.", "confidence": 0.8, "citations": ["vllm.md"]}


# ----------------------------------------------------------------- compaction


def test_compaction_keeps_the_newest_payload_and_cites_the_rest():
    payload = json.dumps({"query": "vLLM", "matches": [
        {"document": "vllm.md", "chunk_id": 0, "anchor": "serving", "text": f"It uses {MARKER} to batch."},
    ]})
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "a", "content": payload},
        {"role": "tool", "tool_call_id": "b", "content": payload},
    ]
    compact_history(messages, keep_raw=1)

    assert MARKER not in messages[1]["content"], "the stale payload keeps only its citations"
    assert "vllm.md#serving" in messages[1]["content"]
    assert MARKER in messages[2]["content"], "the newest payload is left intact"


def test_compaction_is_idempotent():
    payload = json.dumps({"query": "q", "matches": [{"document": "a.md", "chunk_id": 0, "text": MARKER}]})
    messages = [{"role": "tool", "tool_call_id": "a", "content": payload}, {"role": "tool", "content": payload}]
    compact_history(messages, keep_raw=1)
    once = messages[0]["content"]
    compact_history(messages, keep_raw=1)
    assert messages[0]["content"] == once


def test_compaction_leaves_non_retrieval_payloads_alone():
    messages = [{"role": "tool", "tool_call_id": "a", "content": json.dumps({"expression": "1+1", "result": 2})},
                {"role": "tool", "tool_call_id": "b", "content": "{}"}]
    compact_history(messages, keep_raw=1)
    assert json.loads(messages[0]["content"])["result"] == 2


# ---------------------------------------------------------------- the team


async def test_findings_cross_the_boundary_but_chunk_text_does_not(settings):
    _, team, llm = build(settings)
    stub = wire(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))

    completion = await team.answer("What is vLLM and what is RAG?", "", 0.2, 0.9)

    assert completion.agent_mode == "multi"
    assert completion.answer == ANSWER["answer"]

    synthesis = stub.prompts_for("You write the final answer")
    assert synthesis, "the synthesiser ran"
    prompt = json.dumps(synthesis[0]["messages"])
    assert "vLLM raises throughput." in prompt, "compacted findings are what the synthesiser sees"
    assert MARKER not in prompt, "raw retrieved chunks must never enter the synthesiser's context"

    # The evidence is still returned to the caller, just out-of-band.
    assert any(MARKER in source["text"] for source in completion.retrieved)


async def test_the_planner_fans_out_one_researcher_per_task(settings):
    _, team, llm = build(settings)
    stub = wire(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))
    completion = await team.answer("two things", "", 0.2, 0.9)

    questions = {r["messages"][1]["content"] for r in stub.prompts_for("You are a documentation researcher")}
    assert len(completion.plan) == 2
    assert questions == {"Question: What is vLLM?", "Question: What is RAG?"}


async def test_researchers_are_capped_by_max_researchers(settings):
    _, team, llm = build(settings, max_researchers=1)
    plan = {"tasks": [{"question": f"q{i}"} for i in range(5)]}
    wire(llm, RoutingStub(plan, FINDINGS, ANSWER))
    completion = await team.answer("many things", "", 0.2, 0.9)
    assert len(completion.plan) == 1


async def test_a_question_needing_no_research_uses_the_single_agent_loop(settings):
    _, team, llm = build(settings)
    stub = wire(llm, RoutingStub({"tasks": []}, FINDINGS, ANSWER))

    completion = await team.answer("what is 12 * 4?", "", 0.2, 0.9)

    assert completion.agent_mode == "multi->single"
    assert not stub.prompts_for("You are a documentation researcher"), "no researcher was paid for"


async def test_tokens_and_trajectory_are_reported_for_the_whole_team(settings):
    _, team, llm = build(settings)
    wire(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))
    completion = await team.answer("two things", "", 0.2, 0.9)

    # planner + researcher turns + synthesiser, counted as one trajectory.
    assert completion.iterations >= 3
    assert completion.usage.model_calls == sum(task["iterations"] for task in completion.plan) + 2


# --------------------------------------------------------------- degradation


async def test_withdrawing_the_search_tool_is_reported_not_papered_over(settings):
    _, team, llm = build(settings, fault_injection="tool_unavailable")
    stub = wire(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))

    completion = await team.answer("What is vLLM?", "", 0.2, 0.9)

    assert not stub.prompts_for("You are a documentation researcher"), "no researcher ran without its tool"
    assert all(task["error"] for task in completion.plan)
    brief = json.dumps(stub.prompts_for("You write the final answer")[0]["messages"])
    assert "unavailable" in brief, "the synthesiser is told the evidence is missing"


async def test_one_failing_researcher_does_not_sink_the_answer(settings):
    _, team, llm = build(settings, fallback_provider="none")
    stub = RoutingStub(PLAN_TWO, FINDINGS, ANSWER)
    calls = {"n": 0}
    inner = stub.chat.completions.create

    async def flaky(**kwargs):
        if kwargs["messages"][0]["content"].startswith("You are a documentation researcher"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("researcher upstream exploded")
        return await inner(**kwargs)

    stub.chat.completions.create = flaky
    wire(llm, stub)

    completion = await team.answer("two things", "", 0.2, 0.9)

    assert completion.answer == ANSWER["answer"]
    assert sum(1 for task in completion.plan if task["error"]) == 1


# ------------------------------------------------------- retrieval breadth


def test_the_researcher_budget_is_applied_when_the_model_omits_top_k(settings):
    from app.agents import with_default_top_k

    assert json.loads(with_default_top_k('{"query": "json"}', 5)) == {"query": "json", "top_k": 5}
    # An explicit choice by the model is respected.
    assert json.loads(with_default_top_k('{"query": "json", "top_k": 2}', 5))["top_k"] == 2
    # A malformed call is passed through so the registry can report it.
    assert with_default_top_k("{oops", 5) == "{oops"


async def test_researcher_searches_use_the_configured_breadth(settings):
    _, team, llm = build(settings, researcher_top_k=7)
    wire(llm, RoutingStub(PLAN_TWO, FINDINGS, ANSWER))
    completion = await team.answer("two things", "", 0.2, 0.9)

    searches = [c for c in completion.tool_calls if c["name"] == "search_knowledge_base"]
    assert searches and all(c["arguments"]["top_k"] == 7 for c in searches)
