"""Covers the tool-calling loop, retry, fallback, and structured-output parsing.

The upstream client is stubbed so these run without a network or an API key.
"""

import json

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from app.config import Settings
from app.llm import LLMService
from app.retrieval import Retriever
from app.tools import ToolRegistry


def assistant(content=None, tool_calls=None):
    calls = [
        ChatCompletionMessageFunctionToolCall(id=f"c{index}", type="function", function=Function(name=name, arguments=arguments))
        for index, (name, arguments) in enumerate(tool_calls or [])
    ]
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=calls or None)


def status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://stub/v1/chat/completions")
    return openai.APIStatusError("boom", response=httpx.Response(code, request=request), body=None)


class StubClient:
    """Replays a scripted list of messages/exceptions and records each request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        outer = self

        class Completions:
            async def create(self, **kwargs):
                outer.requests.append(kwargs)
                step = outer.script.pop(0)
                if isinstance(step, Exception):
                    raise step
                return type("Response", (), {"choices": [type("Choice", (), {"message": step})()]})()

        self.chat = type("Chat", (), {"completions": Completions()})()


def build(settings: Settings, **overrides):
    settings = settings.model_copy(update={"provider": "openai", "openai_api_key": "test-key", **overrides})
    service = LLMService(settings, ToolRegistry(Retriever(settings)))
    return settings, service


def wire(service: LLMService, script) -> StubClient:
    stub = StubClient(script)
    service._client = lambda target: stub
    return stub


async def test_model_driven_tool_calls_are_executed_and_fed_back(settings):
    _, service = build(settings)
    final = json.dumps({"answer": "vLLM batches requests; 12*4=48.", "confidence": 0.9, "citations": ["vllm.md"]})
    stub = wire(
        service,
        [
            assistant(tool_calls=[
                ("search_knowledge_base", json.dumps({"query": "vLLM throughput", "top_k": 2})),
                ("calculator", json.dumps({"expression": "12 * 4"})),
            ]),
            assistant(content=final),
        ],
    )

    completion = await service.complete("What does vLLM do, and what is 12*4?", "", 0.2, 0.9)

    assert [call["name"] for call in completion.tool_calls] == ["search_knowledge_base", "calculator"]
    assert completion.tool_calls[1]["result"]["result"] == 48
    assert completion.retrieved, "search results are captured so the API can cite them"
    assert completion.answer.endswith("12*4=48.")
    assert completion.confidence == 0.9 and completion.citations == ["vllm.md"]

    # The first request advertises the tools; the second carries the tool results.
    assert "tools" in stub.requests[0]
    roles = [message["role"] for message in stub.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]


async def test_tool_errors_are_returned_to_the_model_rather_than_raised(settings):
    _, service = build(settings)
    stub = wire(
        service,
        [
            assistant(tool_calls=[("calculator", json.dumps({"expression": "__import__('os')"}))]),
            assistant(content=json.dumps({"answer": "I could not compute that.", "confidence": 0.2})),
        ],
    )
    completion = await service.complete("compute something odd", "", 0.2, 0.9)
    assert "error" in completion.tool_calls[0]["result"]
    assert completion.answer == "I could not compute that."


async def test_tool_budget_is_bounded_and_still_answers(settings):
    _, service = build(settings, max_tool_iterations=2)
    loop = assistant(tool_calls=[("calculator", json.dumps({"expression": "1+1"}))])
    stub = wire(service, [loop, loop, assistant(content=json.dumps({"answer": "done", "confidence": 0.5}))])

    completion = await service.complete("loop forever", "", 0.2, 0.9)

    assert completion.answer == "done"
    # Last scripted turn must have tools disabled so the model has to answer.
    assert "tools" not in stub.requests[-1]


async def test_retry_on_transient_status_then_success(settings):
    _, service = build(settings, max_retries=3)
    stub = wire(service, [status_error(503), status_error(429), assistant(content=json.dumps({"answer": "ok"}))])
    completion = await service.complete("q", "ctx", 0.2, 0.9)
    assert completion.answer == "ok"
    assert len(stub.requests) == 3


async def test_non_retryable_status_is_not_retried(settings):
    _, service = build(settings, max_retries=3, fallback_provider="none")
    stub = wire(service, [status_error(401)])
    with pytest.raises(RuntimeError):
        await service.complete("q", "ctx", 0.2, 0.9)
    assert len(stub.requests) == 1


async def test_fallback_provider_takes_over_when_primary_fails(settings):
    _, service = build(
        settings,
        provider="vllm",
        fallback_provider="openai",
        fallback_model="gpt-4o-mini",
        max_retries=1,
    )

    primary = StubClient([status_error(500)])
    fallback = StubClient([assistant(content=json.dumps({"answer": "served by fallback", "confidence": 0.7}))])
    service._client = lambda target: fallback if target.label.startswith("fallback") else primary

    completion = await service.complete("q", "ctx", 0.2, 0.9)

    assert completion.provider == "fallback-openai"
    assert completion.model == "gpt-4o-mini"
    assert completion.answer == "served by fallback"


async def test_both_providers_failing_raises_with_both_causes(settings):
    _, service = build(settings, provider="vllm", fallback_provider="openai", max_retries=1)
    service._client = lambda target: StubClient([status_error(500)])
    with pytest.raises(RuntimeError, match="fallback"):
        await service.complete("q", "ctx", 0.2, 0.9)


async def test_non_json_reply_degrades_instead_of_failing(settings):
    _, service = build(settings)
    wire(service, [assistant(content="plain prose, not JSON")])
    completion = await service.complete("q", "ctx", 0.2, 0.9)
    assert completion.answer == "plain prose, not JSON"
    assert completion.confidence == 0.3


async def test_confidence_is_clamped_and_bad_citations_ignored(settings):
    _, service = build(settings)
    wire(service, [assistant(content=json.dumps({"answer": "a", "confidence": 7, "citations": "vllm.md"}))])
    completion = await service.complete("q", "ctx", 0.2, 0.9)
    assert completion.confidence == 1.0
    assert completion.citations == []


async def test_local_provider_needs_no_client(settings):
    service = LLMService(settings, ToolRegistry(Retriever(settings)))
    completion = await service.complete("calculate 12 * 4", "", 0.2, 0.9)
    assert completion.provider == "local"
    assert completion.tool_calls[0]["result"] == 48


async def test_ollama_is_reachable_as_a_keyless_openai_compatible_provider(settings):
    """A local model must need no key, and must be usable as a fallback too."""
    _, service = build(settings, provider="ollama", ollama_model="qwen2.5:7b", fallback_provider="none")
    wire(service, [assistant(content=json.dumps({"answer": "local", "confidence": 0.6}))])

    completion = await service.complete("q", "ctx", 0.2, 0.9)

    assert completion.provider == "ollama" and completion.model == "qwen2.5:7b"
    assert completion.answer == "local"


async def test_ollama_can_serve_as_the_fallback_for_a_hosted_primary(settings):
    _, service = build(settings, provider="openai", fallback_provider="ollama", max_retries=1)
    primary = StubClient([status_error(500)])
    fallback = StubClient([assistant(content=json.dumps({"answer": "served locally"}))])
    service._client = lambda target: fallback if target.label.startswith("fallback") else primary

    completion = await service.complete("q", "ctx", 0.2, 0.9)

    assert completion.provider == "fallback-ollama"
    assert completion.answer == "served locally"


async def test_json_mode_is_dropped_while_tools_are_offered_on_ollama(settings):
    """Ollama cannot do both at once: forcing JSON silently disables tool calls."""
    _, service = build(settings, provider="ollama", fallback_provider="none", max_tool_iterations=2)
    stub = wire(service, [
        assistant(tool_calls=[("calculator", json.dumps({"expression": "1+1"}))]),
        assistant(content=json.dumps({"answer": "2", "confidence": 0.9})),
    ])

    await service.complete("what is 1+1?", "", 0.2, 0.9)

    tool_turn, final_turn = stub.requests
    assert "tools" in tool_turn and "response_format" not in tool_turn
    # The turn that actually produces the answer still has JSON enforced.
    assert "tools" not in final_turn and final_turn["response_format"] == {"type": "json_object"}


async def test_a_hosted_provider_keeps_json_mode_alongside_tools(settings):
    _, service = build(settings, provider="openai", max_tool_iterations=2)
    stub = wire(service, [assistant(content=json.dumps({"answer": "ok"}))])
    await service.complete("q", "ctx", 0.2, 0.9)
    assert stub.requests[0]["response_format"] == {"type": "json_object"}
    assert "tools" in stub.requests[0]
