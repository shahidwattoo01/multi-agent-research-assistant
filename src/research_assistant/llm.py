"""LLM layer: a small provider-agnostic interface plus the Claude (Anthropic API) implementation.

Agents depend only on :class:`LLMClient`, which offers three primitives:

* ``complete``   -- plain text generation
* ``structured`` -- schema-constrained JSON, validated into a Pydantic model
* ``tool_loop``  -- an agentic tool-calling loop that ends when the model calls a *finish tool*

Keeping the interface narrow makes agents trivially testable with a scripted fake client.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from research_assistant.models import UsageStats
from research_assistant.schema import strict_json_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLMRefusalError(LLMError):
    def __init__(self, category: str | None, explanation: str | None = None):
        super().__init__(f"Model declined the request (category={category}): {explanation or ''}")
        self.category = category


# ---------------------------------------------------------------------------
# Tool-loop contracts
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }

    @classmethod
    def from_model(cls, name: str, description: str, model: type[BaseModel]) -> ToolSpec:
        return cls(name=name, description=description, input_schema=strict_json_schema(model))


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False


ToolExecutor = Callable[[str, dict[str, Any]], ToolOutcome]


@dataclass
class ToolLoopResult:
    final_input: dict[str, Any] | None  # arguments passed to the finish tool
    tool_calls: int
    transcript_text: str  # assistant text produced along the way (for fallback extraction)
    stopped_reason: str  # "finished" | "budget" | "no_finish"


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


@dataclass
class UsageTracker:
    stats: UsageStats = field(default_factory=UsageStats)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, agent: str, usage: Any) -> None:
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        with self._lock:
            s = self.stats
            s.llm_calls += 1
            s.input_tokens += inp
            s.output_tokens += out
            s.cache_read_tokens += cached
            bucket = s.by_agent.setdefault(
                agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += inp
            bucket["output_tokens"] += out

    def record_tool_call(self, n: int = 1) -> None:
        with self._lock:
            self.stats.tool_calls += n

    def snapshot(self) -> UsageStats:
        with self._lock:
            return self.stats.model_copy(deep=True)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    usage: UsageTracker

    def complete(self, *, agent: str, system: str, prompt: str, model: str, effort: str) -> str: ...

    def structured(
        self, *, agent: str, system: str, prompt: str, schema: type[T], model: str, effort: str
    ) -> T: ...

    def tool_loop(
        self,
        *,
        agent: str,
        system: str,
        prompt: str,
        tools: list[ToolSpec],
        finish_tool: ToolSpec,
        executor: ToolExecutor,
        model: str,
        effort: str,
        max_tool_calls: int,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ToolLoopResult: ...


# ---------------------------------------------------------------------------
# Claude implementation
# ---------------------------------------------------------------------------


class ClaudeLLM:
    """:class:`LLMClient` backed by the Anthropic Messages API."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        max_tokens: int = 16000,
        enable_fallbacks: bool = True,
        usage: UsageTracker | None = None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.enable_fallbacks = enable_fallbacks
        self.usage = usage or UsageTracker()

    # -- transport -----------------------------------------------------------

    def _create(self, agent: str, **params: Any) -> Any:
        params.setdefault("max_tokens", self.max_tokens)
        # Keep the (static) system prompt cacheable across calls from the same agent.
        if isinstance(params.get("system"), str):
            params["system"] = [
                {"type": "text", "text": params["system"], "cache_control": {"type": "ephemeral"}}
            ]
        try:
            if self.enable_fallbacks:
                response = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self.client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Rate limited by the Anthropic API: {exc.message}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Bad request: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Authentication failed. Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

        self.usage.record(agent, response.usage)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )
        return response

    @staticmethod
    def _text(response: Any) -> str:
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")

    # -- primitives ----------------------------------------------------------

    def complete(self, *, agent: str, system: str, prompt: str, model: str, effort: str) -> str:
        response = self._create(
            agent,
            model=model,
            system=system,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
        return self._text(response).strip()

    def structured(
        self, *, agent: str, system: str, prompt: str, schema: type[T], model: str, effort: str
    ) -> T:
        response = self._create(
            agent,
            model=model,
            system=system,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{agent}: structured output truncated at max_tokens")
        raw = self._text(response)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(f"{agent}: model output failed schema validation: {exc}") from exc

    def tool_loop(
        self,
        *,
        agent: str,
        system: str,
        prompt: str,
        tools: list[ToolSpec],
        finish_tool: ToolSpec,
        executor: ToolExecutor,
        model: str,
        effort: str,
        max_tool_calls: int,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ToolLoopResult:
        """Run a manual agentic loop.

        A manual loop (rather than the SDK tool runner) is used because the loop needs
        custom control: a per-task tool budget, a finish tool that ends the loop with
        structured output, and a nudge when the model stops without submitting.
        """
        api_tools = [t.to_api() for t in [*tools, finish_tool]]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        calls = 0
        nudged = False
        text_parts: list[str] = []

        for _ in range(max_tool_calls + 4):  # hard ceiling on model turns
            response = self._create(
                agent,
                model=model,
                system=system,
                tools=api_tools,
                output_config={"effort": effort},
                messages=messages,
            )
            text_parts.append(self._text(response))
            # Append the full content (including thinking blocks) so the history stays append-only.
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            finish = next((b for b in tool_uses if b.name == finish_tool.name), None)
            if finish is not None:
                return ToolLoopResult(dict(finish.input), calls, "\n".join(text_parts), "finished")

            if not tool_uses:
                if nudged:
                    return ToolLoopResult(None, calls, "\n".join(text_parts), "no_finish")
                nudged = True
                messages.append(
                    {
                        "role": "user",
                        "content": f"Please call `{finish_tool.name}` now with your findings.",
                    }
                )
                continue

            results = self._run_tools(tool_uses, executor, calls, max_tool_calls, on_tool_call)
            calls += sum(1 for r in results if not r.get("_budget"))
            self.usage.record_tool_call(sum(1 for r in results if not r.get("_budget")))
            for r in results:
                r.pop("_budget", None)
            messages.append({"role": "user", "content": results})

        return ToolLoopResult(None, calls, "\n".join(text_parts), "budget")

    def _run_tools(
        self,
        tool_uses: list[Any],
        executor: ToolExecutor,
        calls_so_far: int,
        budget: int,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        """Execute parallel tool calls concurrently; return all results in one message."""

        def run(index_block: tuple[int, Any]) -> dict[str, Any]:
            index, block = index_block
            if calls_so_far + index >= budget:
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Tool budget exhausted. Call submit_findings now with what you have.",
                    "is_error": True,
                    "_budget": True,
                }
            args = dict(block.input) if isinstance(block.input, dict) else {}
            if on_tool_call:
                on_tool_call(block.name, args)
            try:
                outcome = executor(block.name, args)
            except Exception as exc:  # tool failures are reported to the model, not raised
                logger.warning("tool %s failed: %s", block.name, exc)
                outcome = ToolOutcome(f"Error running {block.name}: {exc}", is_error=True)
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": outcome.content,
            }
            if outcome.is_error:
                result["is_error"] = True
            return result

        with ThreadPoolExecutor(max_workers=min(4, len(tool_uses))) as pool:
            return list(pool.map(run, enumerate(tool_uses)))
