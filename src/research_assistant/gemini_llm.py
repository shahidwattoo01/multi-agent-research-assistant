"""GeminiLLM: LLMClient implementation backed by Google Gemini (free tier)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from research_assistant.llm import (
    LLMError,
    ToolExecutor,
    ToolLoopResult,
    ToolOutcome,
    ToolSpec,
    UsageTracker,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_JSON_TO_GEMINI = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _schema_to_dict(js: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    t = js.get("type", "string")
    result["type"] = _JSON_TO_GEMINI.get(t, "STRING")
    if "description" in js:
        result["description"] = js["description"]
    if "enum" in js:
        result["enum"] = js["enum"]
    if t == "object" and "properties" in js:
        result["properties"] = {k: _schema_to_dict(v) for k, v in js["properties"].items()}
        if "required" in js:
            result["required"] = js["required"]
    if t == "array" and "items" in js:
        result["items"] = _schema_to_dict(js["items"])
    return result


class _FakeUsage:
    def __init__(self, meta: Any) -> None:
        self.input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        self.output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        self.cache_read_input_tokens = 0


class GeminiLLM:
    def __init__(self, model: str = "gemini-1.5-flash", usage: UsageTracker | None = None) -> None:
        import google.generativeai as genai
        self._genai = genai
        self.default_model = model
        self.usage = usage or UsageTracker()

    def _record(self, agent: str, meta: Any) -> None:
        if meta is not None:
            self.usage.record(agent, _FakeUsage(meta))

    def complete(self, *, agent: str, system: str, prompt: str, model: str, effort: str) -> str:
        m = self._genai.GenerativeModel(model_name=model or self.default_model, system_instruction=system)
        try:
            response = m.generate_content(prompt)
        except Exception as exc:
            raise LLMError(f"Gemini API error in complete: {exc}") from exc
        self._record(agent, response.usage_metadata)
        return response.text.strip()

    def structured(self, *, agent: str, system: str, prompt: str, schema: type[T], model: str, effort: str) -> T:
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        m = self._genai.GenerativeModel(
            model_name=model or self.default_model,
            system_instruction=system,
            generation_config={"response_mime_type": "application/json"},
        )
        try:
            response = m.generate_content(full_prompt)
        except Exception as exc:
            raise LLMError(f"Gemini API error in structured: {exc}") from exc
        self._record(agent, response.usage_metadata)
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMError(f"{agent}: Gemini output failed schema validation: {exc}") from exc

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
        genai = self._genai
        all_specs = [*tools, finish_tool]
        tools_dict = [
            {
                "function_declarations": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "parameters": _schema_to_dict(s.input_schema),
                    }
                    for s in all_specs
                ]
            }
        ]
        m = self._genai.GenerativeModel(
            model_name=model or self.default_model,
            system_instruction=system,
            tools=tools_dict,
        )
        chat = m.start_chat(history=[])
        calls = 0
        text_parts: list[str] = []
        nudged = False
        try:
            response = chat.send_message(prompt)
        except Exception as exc:
            raise LLMError(f"Gemini tool_loop initial call failed: {exc}") from exc

        for _ in range(max_tool_calls + 4):
            self._record(agent, response.usage_metadata)
            for part in response.parts:
                txt = getattr(part, "text", None)
                if txt:
                    text_parts.append(txt)
            fc_parts = [p for p in response.parts if getattr(p, "function_call", None) and p.function_call.name]
            if not fc_parts:
                if nudged:
                    return ToolLoopResult(None, calls, "\n".join(text_parts), "no_finish")
                nudged = True
                try:
                    response = chat.send_message(f"Please call `{finish_tool.name}` now with your findings.")
                except Exception as exc:
                    raise LLMError(f"Gemini nudge failed: {exc}") from exc
                continue
            finish_fc = next(
                (p.function_call for p in fc_parts if p.function_call.name == finish_tool.name),
                None,
            )
            if finish_fc:
                return ToolLoopResult(dict(finish_fc.args), calls, "\n".join(text_parts), "finished")
            tool_responses: list[Any] = []
            for part in fc_parts:
                fc = part.function_call
                if calls >= max_tool_calls:
                    tool_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fc.name,
                                response={"result": "Tool budget exhausted. Call the finish tool now."},
                            )
                        )
                    )
                    continue
                args = dict(fc.args)
                if on_tool_call:
                    on_tool_call(fc.name, args)
                try:
                    outcome = executor(fc.name, args)
                except Exception as exc:
                    logger.warning("tool %s failed: %s", fc.name, exc)
                    outcome = ToolOutcome(f"Error running {fc.name}: {exc}", is_error=True)
                calls += 1
                self.usage.record_tool_call(1)
                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response={"result": outcome.content},
                        )
                    )
                )
            try:
                response = chat.send_message(tool_responses)
            except Exception as exc:
                raise LLMError(f"Gemini tool response failed: {exc}") from exc

        return ToolLoopResult(None, calls, "\n".join(text_parts), "budget")
