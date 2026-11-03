"""Fake Anthropic SDK clients that return real ``anthropic.types.Message`` objects."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from typing import Any

from anthropic.types import Message

_ids = itertools.count(1)


def make_message(*blocks: dict[str, Any], stop_reason: str | None = None) -> Message:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return Message.model_validate(
        {
            "id": f"msg_{next(_ids)}",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 40, "cache_read_input_tokens": 10},
        }
    )


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def thinking() -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": f"sig_{next(_ids)}"}


def tool_use(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"toolu_{next(_ids):04d}", "name": name, "input": args}


class _Messages:
    def __init__(self, handler: Callable[[dict[str, Any]], Message], log: list[dict[str, Any]]):
        self._handler = handler
        self._log = log

    def create(self, **params: Any) -> Message:
        # snapshot the history: callers keep appending to the same list
        self._log.append({**params, "messages": list(params.get("messages", []))})
        return self._handler(params)


class _Beta:
    def __init__(self, messages: _Messages) -> None:
        self.messages = messages


class FakeAnthropic:
    """Sync stand-in for ``anthropic.Anthropic``; ``handler`` maps request params to a Message."""

    def __init__(self, handler: Callable[[dict[str, Any]], Message]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(handler, self.requests)
        self.beta = _Beta(self.messages)


# ---------------------------------------------------------------------------
# Async fake that scripts a full AutoGen planner -> researcher -> critic run
# ---------------------------------------------------------------------------


def _blocks(message: dict[str, Any]) -> list[Any]:
    content = message.get("content")
    return content if isinstance(content, list) else []


def _get(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def _tool_results(params: dict[str, Any]) -> list[str]:
    last = params["messages"][-1]
    if last.get("role") != "user":
        return []
    return [str(_get(b, "content")) for b in _blocks(last) if _get(b, "type") == "tool_result"]


def _system_text(params: dict[str, Any]) -> str:
    system = params.get("system", "")
    if isinstance(system, list):
        return " ".join(_get(b, "text") or "" for b in system)
    return str(system)


class ScriptedAsyncAnthropic:
    """Plays planner, researcher and critic, and checks request shape on every call."""

    def __init__(self, critic_scores: list[float]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.critic_scores = critic_scores
        self.critic_round = 0
        self.messages = self

    async def create(self, **params: Any) -> Message:
        self.requests.append(params)
        self._check_request(params)
        system = _system_text(params)
        if "Planner" in system:
            return self._planner(params)
        if "Researcher" in system:
            return self._researcher(params)
        if "Critic" in system:
            return self._critic(params)
        return make_message(text("ok"))

    @staticmethod
    def _check_request(params: dict[str, Any]) -> None:
        assert "temperature" not in params, "sampling params must be stripped"
        assert params.get("output_config", {}).get("effort")
        for m in params["messages"]:
            if m.get("role") != "assistant":
                continue
            blocks = _blocks(m)
            for b in blocks:
                if _get(b, "type") == "text":
                    assert _get(b, "text"), "empty text block replayed"
            if any(_get(b, "type") == "tool_use" for b in blocks):
                assert _get(blocks[0], "type") == "thinking", "thinking block not restored"

    def _planner(self, params: dict[str, Any]) -> Message:
        if _tool_results(params):
            return make_message(thinking(), text("Plan registered: solar and wind."))
        return make_message(
            thinking(),
            tool_use(
                "register_plan",
                {
                    "objective": "Explain renewable cost trends",
                    "sub_questions": [
                        "How have solar costs changed since 2010?",
                        "How have onshore wind costs changed since 2010?",
                    ],
                    "success_criteria": ["Quantify declines"],
                },
            ),
        )

    def _researcher(self, params: dict[str, Any]) -> Message:
        results = _tool_results(params)
        joined = "\n".join(results)
        ids = re.findall(r"(task_[0-9a-f]{8}):", joined)
        if ids:  # just got pending tasks -> search and read, remembering ids in the input
            return make_message(
                thinking(),
                tool_use("web_search", {"query": "renewable cost " + ",".join(ids)}),
                tool_use("fetch_page", {"url": "https://example.org/solar", "focus": "costs"}),
            )
        if "Results for" in joined:  # got search + page -> submit one finding per task id
            last_assistant = next(
                m for m in reversed(params["messages"]) if m["role"] == "assistant"
            )
            queries = json.dumps([_get(b, "input") for b in _blocks(last_assistant)])
            ids = list(dict.fromkeys(re.findall(r"task_[0-9a-f]{8}", queries)))
            return make_message(
                thinking(),
                *[
                    tool_use(
                        "submit_finding",
                        {
                            "task_id": tid,
                            "summary": "Costs fell sharply.",
                            "key_points": ["Solar LCOE fell 90%"],
                            "source_urls": ["https://example.org/solar", "https://made.up/x"],
                            "confidence": 0.8,
                            "gaps": [],
                        },
                    )
                    for tid in ids
                ],
            )
        if "Finding recorded" in joined or "No pending tasks" in joined:
            return make_message(thinking(), text("All assigned tasks researched."))
        return make_message(thinking(), tool_use("get_pending_tasks", {}))

    def _critic(self, params: dict[str, Any]) -> Message:
        results = _tool_results(params)
        if not results:
            return make_message(thinking(), tool_use("get_findings", {}))
        if any("Q:" in r for r in results):
            score = self.critic_scores[min(self.critic_round, len(self.critic_scores) - 1)]
            self.critic_round += 1
            follow = [] if score >= 0.75 else ["How have battery storage costs changed since 2015?"]
            return make_message(
                thinking(),
                tool_use(
                    "submit_review",
                    {
                        "score": score,
                        "approved": score >= 0.75,
                        "strengths": ["quantified"],
                        "issues": ["storage missing"] if follow else [],
                        "follow_up_questions": follow,
                    },
                ),
            )
        return make_message(thinking(), text("Verdict delivered."))
