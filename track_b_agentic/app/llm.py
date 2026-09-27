"""Provider adapters plus the model-driven tool-calling loop."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from .config import Settings
from .tools import ToolRegistry, heuristic_tool_calls

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise engineering assistant for a retrieval-augmented system.

Rules:
- Prefer the supplied context and tool results over prior knowledge.
- Call `search_knowledge_base` when the answer may live in the project documents, and search again with different wording if the first results are thin.
- Call `calculator` for any arithmetic. Never compute it yourself.
- If the retrieved material does not answer the question, say so plainly instead of guessing.

Reply with a single JSON object and nothing else, using exactly these keys:
  "answer": string, the response for the user.
  "confidence": number between 0 and 1, how well the evidence supports the answer.
  "citations": array of strings, the document names you actually relied on."""

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class Usage:
    """Token spend for one user question, summed over every upstream call.

    The evaluation harness needs this to make the coordination cost of the
    multi-agent mode visible next to the single-agent baseline, so it is
    accumulated here rather than reconstructed from logs.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    # Time spent sleeping in retry backoff (mostly rate-limit waits on a hosted
    # free tier), so wall-clock latency can be reported net of throttling.
    wait_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, response: Any) -> None:
        self.model_calls += 1
        raw = getattr(response, "usage", None)
        if raw is None:  # stubs in tests, and providers that omit the block
            return
        self.prompt_tokens += int(getattr(raw, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(raw, "completion_tokens", 0) or 0)

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.model_calls += other.model_calls
        self.wait_ms += other.wait_ms

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
            "wait_ms": self.wait_ms,
        }


@dataclass
class Completion:
    answer: str
    confidence: float = 0.5
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "local"
    model: str = "offline"
    # Trajectory length: how many loop turns the agent actually consumed.
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    agent_mode: str = "single"
    plan: list[dict[str, Any]] = field(default_factory=list)
    # W17 tracing: one record per model turn (decision, reasoning, tool calls with
    # arguments and raw results, tokens, latency) and why the loop stopped:
    # "success" (the model chose to answer), "max_iterations" (the tool budget ran
    # out and a final answer was forced) or "error" (see AgentRunError).
    steps: list[dict[str, Any]] = field(default_factory=list)
    termination: str = "success"


class AgentRunError(RuntimeError):
    """A run that raised part-way, carrying the steps recorded before it failed.

    Without this an exception throws the partial trajectory away, and an error
    trace is exactly the one you most need to read.
    """

    def __init__(self, message: str, steps: list[dict[str, Any]], usage: "Usage | None" = None):
        super().__init__(message)
        self.steps = steps
        # Tokens spent before the failure: an error still costs money.
        self.usage = usage or Usage()


def message_reasoning(message: Any) -> str:
    """The model's own intermediate reasoning, when the provider exposes it.

    Reasoning models on Groq (gpt-oss) return it as a ``reasoning`` field beside
    ``content``; other providers use ``reasoning_content``. Neither is part of the
    typed OpenAI message, so it is read from the extras.
    """
    for key in ("reasoning", "reasoning_content"):
        value = getattr(message, key, None)
        if value is None:
            extra = getattr(message, "model_extra", None) or {}
            value = extra.get(key)
        if value:
            return str(value)
    return ""


def assistant_turn(message: Any) -> dict[str, Any]:
    """An assistant message fit to send back upstream.

    Provider-specific extras such as ``reasoning`` are dropped: they are recorded
    in the trace, but some OpenAI-compatible endpoints reject them as input.
    """
    dumped = message.model_dump(exclude_none=True)
    return {key: value for key, value in dumped.items() if key in {"role", "content", "tool_calls", "name"}}


def step_record(
    step: int,
    agent: str,
    message: Any,
    *,
    decision: str,
    tools_offered: bool,
    before: dict[str, int],
    usage: "Usage",
    latency_ms: int,
) -> dict[str, Any]:
    after = usage.as_dict()
    return {
        "step": step,
        "agent": agent,
        "decision": decision,
        "tools_offered": tools_offered,
        "reasoning": message_reasoning(message),
        "recovered_from": (getattr(message, "model_extra", None) or {}).get("recovered_from", ""),
        "content": getattr(message, "content", None) or "",
        "tool_calls": [],
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "model_calls": after["model_calls"] - before["model_calls"],
        "latency_ms": latency_ms,
    }


def error_step(step: int, agent: str, error: Exception) -> dict[str, Any]:
    """The last record of a trajectory that raised: which turn failed, and why.

    When the provider rejected a generation it had already produced (Groq's
    ``tool_use_failed`` carries it as ``failed_generation``), that text is the
    model's actual decision at this step, so it is kept as a field of its own.
    """
    record = {
        "step": step, "agent": agent, "decision": "error", "error": f"{type(error).__name__}: {error}",
        "tool_calls": [], "reasoning": "", "content": "",
    }
    cause: BaseException | None = error
    while cause is not None:
        body = getattr(cause, "body", None)
        if isinstance(body, dict):
            detail = body.get("error", body) if isinstance(body.get("error", body), dict) else body
            if detail.get("code"):
                record["error_code"] = detail["code"]
            if detail.get("failed_generation"):
                record["failed_generation"] = _safe_json(detail["failed_generation"])
            break
        cause = cause.__cause__
    return record


class RatePacer:
    """Wait *before* a call when the provider says the per-minute token budget is spent.

    Groq returns ``x-ratelimit-remaining-tokens`` / ``x-ratelimit-reset-tokens``
    (tokens per minute) and ``x-ratelimit-remaining-requests`` (requests per day)
    on every response. Sleeping until the window resets when the next request
    will not fit is cheaper than sending it, taking a 429 and backing off, and it
    keeps the daily request count from being spent on rejected calls.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.remaining_tokens: int | None = None
        self.reset_at: float = 0.0
        self.remaining_requests: int | None = None
        self.requests = 0  # responses observed, including 4xx: they count against the daily quota too

    def observe(self, headers: Any) -> None:
        self.requests += 1
        try:
            if headers.get("x-ratelimit-remaining-tokens") is not None:
                self.remaining_tokens = int(float(headers["x-ratelimit-remaining-tokens"]))
                self.reset_at = time.monotonic() + _duration(str(headers.get("x-ratelimit-reset-tokens", "0")))
            if headers.get("x-ratelimit-remaining-requests") is not None:
                self.remaining_requests = int(float(headers["x-ratelimit-remaining-requests"]))
        except (TypeError, ValueError):
            pass

    async def wait_for(self, needed: int) -> float:
        if not self.enabled or self.remaining_tokens is None:
            return 0.0
        pause = self.reset_at - time.monotonic()
        if self.remaining_tokens >= needed or pause <= 0:
            return 0.0
        pause = min(pause + 0.5, 60.0)
        logger.info("pacing: %d tokens left in the window, need ~%d; waiting %.1fs", self.remaining_tokens, needed, pause)
        await asyncio.sleep(pause)
        self.remaining_tokens = None  # unknown until the next response reports it
        return pause


def _estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """Rough request size for pacing: ~4 characters per token plus room for the reply."""
    chars = sum(len(str(message.get("content") or "")) + len(str(message.get("tool_calls") or "")) for message in messages)
    chars += len(json.dumps(tools)) if tools else 0
    return chars // 4 + 800


@dataclass(frozen=True)
class Target:
    label: str
    base_url: str
    api_key: str
    model: str
    # Can this endpoint enforce JSON output *and* offer tools in the same call?
    # Ollama cannot: with response_format set, qwen2.5 writes the tool call out
    # as JSON prose and returns tool_calls=null, so the loop never fires a tool.
    # Hosted OpenAI handles both together, and keeps the stricter behaviour.
    json_mode_with_tools: bool = True


def _is_json_object(content: str | None) -> bool:
    try:
        return isinstance(json.loads(content or ""), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code in RETRYABLE_STATUS
    return False


class LLMService:
    def __init__(self, settings: Settings, registry: ToolRegistry):
        self.settings = settings
        self.registry = registry
        self._clients: dict[tuple[str, str], AsyncOpenAI] = {}
        # Bounds in-flight upstream calls so a burst queues instead of stampeding.
        self._gate = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        # Proactive pacing from the provider's own rate-limit headers.
        self.pacer = RatePacer(enabled=settings.pace_from_headers)

    def _client(self, target: Target) -> AsyncOpenAI:
        key = (target.base_url, target.api_key)
        if key not in self._clients:
            pacer = self.pacer

            async def observe(response) -> None:  # httpx response hook: read the rate-limit headers
                pacer.observe(response.headers)

            self._clients[key] = AsyncOpenAI(
                api_key=target.api_key or "EMPTY",
                base_url=target.base_url,
                timeout=self.settings.request_timeout_seconds,
                max_retries=0,  # retries are handled here so backoff is observable
                http_client=openai.DefaultAsyncHttpxClient(event_hooks={"response": [observe]}),
            )
        return self._clients[key]

    def _primary(self) -> Target | None:
        if self.settings.provider == "vllm":
            return Target("vllm", self.settings.vllm_base_url, "EMPTY", self.settings.vllm_model)
        if self.settings.provider == "ollama":
            return Target(
                "ollama", self.settings.ollama_base_url, "ollama", self.settings.ollama_model,
                json_mode_with_tools=False,
            )
        if self.settings.provider == "openai":
            return Target("openai", self.settings.openai_base_url, self.settings.openai_api_key, self.settings.openai_model)
        if self.settings.provider == "groq":
            # Groq rejects response_format=json_object alongside tools (HTTP 400
            # "json mode cannot be combined with tool/function calling"), so tool
            # turns run unconstrained and `structured()` restores the envelope.
            return Target(
                "groq", self.settings.groq_base_url, self.settings.groq_api_key, self.settings.groq_model,
                json_mode_with_tools=False,
            )
        return None

    def _fallback(self, primary: Target) -> Target | None:
        if self.settings.fallback_provider == "groq" and self.settings.groq_api_key:
            candidate = Target(
                "fallback-groq", self.settings.groq_base_url, self.settings.groq_api_key, self.settings.groq_model,
                json_mode_with_tools=False,
            )
        elif self.settings.fallback_provider == "openai" and self.settings.openai_api_key:
            candidate = Target(
                "fallback-openai", self.settings.openai_base_url, self.settings.openai_api_key, self.settings.fallback_model
            )
        elif self.settings.fallback_provider == "vllm":
            candidate = Target("fallback-vllm", self.settings.vllm_base_url, "EMPTY", self.settings.vllm_model)
        elif self.settings.fallback_provider == "ollama":
            candidate = Target(
                "fallback-ollama", self.settings.ollama_base_url, "ollama", self.settings.ollama_model,
                json_mode_with_tools=False,
            )
        else:
            return None
        if (candidate.base_url, candidate.model) == (primary.base_url, primary.model):
            return None
        return candidate

    async def complete(self, question: str, context: str, temperature: float, top_p: float) -> Completion:
        if self._primary() is None:
            return self._offline(question, context)
        return await self._run(question, context, temperature, top_p)

    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        usage: Usage | None = None,
    ) -> tuple[Any, Target]:
        """One upstream turn, with retries on the primary and a provider fail-over.

        Exposed so the multi-agent mode can drive its own loops (planner,
        researcher, synthesiser) without re-implementing retry and fail-over.
        """
        primary = self._primary()
        if primary is None:
            raise RuntimeError("no model provider is configured")

        try:
            return await self._chat(primary, messages, temperature, top_p, tools, usage), primary
        except Exception as primary_error:
            fallback = self._fallback(primary)
            if fallback is None:
                raise
            logger.warning("primary provider %s failed (%s); trying %s", primary.label, primary_error, fallback.label)
            try:
                return await self._chat(fallback, messages, temperature, top_p, tools, usage), fallback
            except Exception as fallback_error:
                raise RuntimeError(
                    f"primary {primary.label} failed ({primary_error}); fallback {fallback.label} failed ({fallback_error})"
                ) from fallback_error

    async def structured(
        self,
        messages: list[dict[str, Any]],
        message: Any,
        target: Target,
        *,
        temperature: float,
        top_p: float,
        usage: Usage | None = None,
    ):
        """Guarantee a JSON envelope from a turn that ended without a tool call.

        A provider that cannot combine tools with JSON mode runs its tool turns
        unconstrained, so a model that decides to answer *mid-loop* answers in
        prose - and prose carries no confidence and no citations, which silently
        empties two fields the caller depends on. Re-issuing the same turn with
        tools off puts JSON mode back on. It costs an extra call only when it is
        actually needed, and nothing at all on a provider that never had the
        problem.
        """
        if target.json_mode_with_tools or _is_json_object(message.content):
            return message
        logger.info("%s answered mid-loop without JSON mode; re-issuing for the envelope", target.label)
        retry, _ = await self.call(messages, tools=None, temperature=temperature, top_p=top_p, usage=usage)
        return retry

    async def _run(self, question: str, context: str, temperature: float, top_p: float) -> Completion:
        steps: list[dict[str, Any]] = []
        usage = Usage()
        try:
            return await self._loop(question, context, temperature, top_p, steps, usage)
        except Exception as error:
            steps.append(error_step(len(steps) + 1, "agent", error))
            raise AgentRunError(f"{type(error).__name__}: {error}", steps, usage) from error

    async def _loop(
        self, question: str, context: str, temperature: float, top_p: float, steps: list[dict[str, Any]],
        usage: Usage | None = None,
    ) -> Completion:
        user_content = f"Retrieved context:\n{context or '(nothing retrieved yet - use search_knowledge_base)'}\n\nQuestion: {question}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.settings.system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        executed: list[dict[str, Any]] = []
        retrieved: list[dict[str, Any]] = []
        usage = usage if usage is not None else Usage()
        target = self._primary()
        iterations = 0

        for iteration in range(self.settings.max_tool_iterations):
            iterations = iteration + 1
            last_round = iteration == self.settings.max_tool_iterations - 1
            tools = None if last_round else self.registry.schemas
            before, started = usage.as_dict(), time.perf_counter()
            message, target = await self.call(messages, tools=tools, temperature=temperature, top_p=top_p, usage=usage)

            if not getattr(message, "tool_calls", None):
                first = message
                message = await self.structured(
                    messages, message, target, temperature=temperature, top_p=top_p, usage=usage
                )
                step = step_record(
                    iterations, "agent", message, decision="final_answer", tools_offered=tools is not None,
                    before=before, usage=usage, latency_ms=int((time.perf_counter() - started) * 1000),
                )
                step["reasoning"] = step["reasoning"] or message_reasoning(first)
                steps.append(step)
                completion = self._parse(message.content, executed, retrieved, target, iterations, usage)
                completion.steps, completion.termination = steps, "success"
                return completion

            messages.append(assistant_turn(message))
            step = step_record(
                iterations, "agent", message, decision="call_tools", tools_offered=True,
                before=before, usage=usage, latency_ms=0,
            )
            for call in message.tool_calls:
                payload, matches = await self.registry.execute(call.function.name, call.function.arguments)
                record = {
                    "name": call.function.name,
                    "arguments": _safe_json(call.function.arguments),
                    "result": _safe_json(payload),
                }
                executed.append(record)
                step["tool_calls"].append(record)
                if matches:
                    retrieved.extend(matches)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})
            step["latency_ms"] = int((time.perf_counter() - started) * 1000)
            steps.append(step)

        # Tool budget spent: ask once more with tools off so a final answer is produced.
        iterations += 1
        before, started = usage.as_dict(), time.perf_counter()
        message, target = await self.call(messages, tools=None, temperature=temperature, top_p=top_p, usage=usage)
        steps.append(
            step_record(
                iterations, "agent", message, decision="forced_final_answer", tools_offered=False,
                before=before, usage=usage, latency_ms=int((time.perf_counter() - started) * 1000),
            )
        )
        completion = self._parse(message.content, executed, retrieved, target, iterations, usage)
        completion.steps, completion.termination = steps, "max_iterations"
        return completion

    async def _chat(
        self,
        target: Target,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        tools: list[dict[str, Any]] | None,
        usage: Usage | None = None,
    ):
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = "auto"
        if self.settings.reasoning_effort and target.label.endswith("groq"):
            extra["reasoning_effort"] = self.settings.reasoning_effort
        if not tools or target.json_mode_with_tools:
            # The final turn always runs with tools off, so structured output is
            # still enforced where the answer is actually produced.
            extra["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        attempts = 0

        for attempt in range(self.settings.max_retries):
            attempts = attempt + 1
            try:
                async with self._gate:
                    waited = await self.pacer.wait_for(_estimate_tokens(messages, tools))
                    if usage is not None:
                        usage.wait_ms += int(waited * 1000)
                    response = await self._client(target).chat.completions.create(
                        model=target.model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        **extra,
                    )
                if usage is not None:
                    usage.record(response)
                return response.choices[0].message
            except Exception as error:
                last_error = error
                recovered = self._recover_answer_tool_call(error, tools) if self.settings.recover_tool_answers else None
                if recovered is not None:
                    if usage is not None:
                        usage.model_calls += 1  # the rejected turn reports no token usage
                    return recovered
                if attempt == self.settings.max_retries - 1 or not _is_retryable(error) or _daily_quota(error):
                    break
                # Exponential backoff with jitter so retries do not synchronise,
                # stretched to the server's own Retry-After when it sends one:
                # a hosted free tier (Groq) rate-limits on tokens per minute, and
                # retrying sooner than it asks only burns an attempt.
                delay = 0.4 * (2**attempt) * (1 + random.random() * 0.25)
                delay = min(max(delay, _retry_after(error)), 90.0)
                logger.warning("%s attempt %d failed (%s); retrying in %.2fs", target.label, attempt + 1, error, delay)
                await asyncio.sleep(delay)
                if usage is not None:
                    usage.wait_ms += int(delay * 1000)

        raise RuntimeError(f"{target.label} request failed after {attempts} attempt(s): {last_error}") from last_error

    @staticmethod
    def _recover_answer_tool_call(error: Exception, tools: list[dict[str, Any]] | None) -> Any | None:
        """Keep a final answer the provider rejected only because it came as a pseudo-tool call.

        gpt-oss on Groq sometimes "calls" a tool named ``json`` whose arguments
        are the finished answer envelope. Groq rejects it with HTTP 400
        ``tool_use_failed`` and returns the text as ``failed_generation``. When
        that name is not an offered tool and the arguments hold an ``answer``,
        the generation *is* the model's final answer: return it as ordinary
        content, flagged ``recovered_from`` so the trace shows what happened. Any
        other rejection is left to fail.
        """
        body = getattr(error, "body", None)
        if not isinstance(body, dict):
            return None
        detail = body.get("error", body) if isinstance(body.get("error", body), dict) else body
        if detail.get("code") != "tool_use_failed" or not detail.get("failed_generation"):
            return None
        generation = _safe_json(detail["failed_generation"])
        if not isinstance(generation, dict):
            return None
        offered = {tool["function"]["name"] for tool in tools or []}
        name, arguments = str(generation.get("name", "")), generation.get("arguments")
        if name in offered or not isinstance(arguments, dict) or "answer" not in arguments:
            return None
        from openai.types.chat import ChatCompletionMessage

        return ChatCompletionMessage.model_validate({
            "role": "assistant",
            "content": json.dumps(arguments),
            "recovered_from": f"tool_use_failed: pseudo-tool '{name}'",
        })

    def _parse(
        self,
        content: str | None,
        executed: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
        target: Target,
        iterations: int = 0,
        usage: Usage | None = None,
    ) -> Completion:
        payload: dict[str, Any]
        try:
            parsed = json.loads(content or "{}")
            payload = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # Graceful degradation: a non-JSON reply is still worth returning.
            logger.warning("%s returned non-JSON content; passing it through verbatim", target.label)
            payload = {"answer": (content or "").strip(), "confidence": 0.3}

        confidence = payload.get("confidence", 0.5)
        citations = payload.get("citations") or []
        return Completion(
            answer=str(payload.get("answer") or "The model returned an empty answer."),
            confidence=min(max(float(confidence), 0.0), 1.0) if isinstance(confidence, (int, float)) else 0.5,
            citations=[str(item) for item in citations] if isinstance(citations, list) else [],
            tool_calls=executed,
            retrieved=retrieved,
            provider=target.label,
            model=target.model,
            iterations=iterations,
            usage=usage or Usage(),
        )

    def _offline(self, question: str, context: str) -> Completion:
        """Keyless demo path. No model, so tool selection is heuristic."""
        tool_calls, tool_context = heuristic_tool_calls(question, self.registry)
        if tool_context:
            return Completion(answer=tool_context, confidence=1.0, tool_calls=tool_calls)
        if not context:
            return Completion(answer="I do not have enough indexed context to answer that question.", confidence=0.0)
        first = context.splitlines()[0]
        return Completion(answer=f"Based on the indexed documents: {first}", confidence=0.4)


def _daily_quota(error: Exception) -> bool:
    """A per-day quota (Groq's TPD/RPD) is exhausted: retrying within minutes cannot help."""
    text = str(error)
    return "tokens per day" in text or "requests per day" in text or "(TPD)" in text or "(RPD)" in text


def _retry_after(error: Exception) -> float:
    """Seconds the server asked us to wait, from a 429's headers; 0 if none."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key) if hasattr(headers, "get") else None
        if not raw:
            continue
        text = str(raw).strip()
        try:
            return float(text)
        except ValueError:
            pass
        seconds = _duration(text)
        if seconds:
            return seconds
    return 0.0


def _duration(text: str) -> float:
    """Seconds in a Groq reset header: "7.66s", "2m59.56s", "120ms" or a bare number."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?", text)
    if match and any(match.groups()):
        hours, minutes, seconds, millis = (float(group) if group else 0.0 for group in match.groups())
        return hours * 3600 + minutes * 60 + seconds + millis / 1000
    return 0.0


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
