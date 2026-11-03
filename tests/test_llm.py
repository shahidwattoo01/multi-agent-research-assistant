"""Tests for the real ClaudeLLM implementation against a fake Anthropic SDK client."""

import json

import pytest

from research_assistant.llm import (
    FALLBACK_BETA,
    ClaudeLLM,
    LLMError,
    LLMRefusalError,
    ToolOutcome,
    ToolSpec,
)
from research_assistant.models import FindingDraft, ResearchPlan
from tests.fake_anthropic import FakeAnthropic, make_message, text, thinking, tool_use

PLAN_JSON = json.dumps(
    {
        "objective": "o",
        "subtasks": [{"question": "q", "rationale": "r", "search_queries": ["s"]}],
        "success_criteria": ["c"],
    }
)
FINISH = ToolSpec.from_model("submit_findings", "finish", FindingDraft)
SEARCH = ToolSpec(
    "web_search",
    "search",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)
FINDING = {"summary": "s", "key_points": ["k"], "sources": [], "confidence": 0.7, "gaps": []}


def test_structured_sends_schema_effort_fallbacks_and_cached_system():
    client = FakeAnthropic(lambda p: make_message(text(PLAN_JSON)))
    llm = ClaudeLLM(client)
    plan = llm.structured(
        agent="planner",
        system="sys",
        prompt="p",
        schema=ResearchPlan,
        model="claude-opus-5-5",
        effort="high",
    )
    assert isinstance(plan, ResearchPlan) and plan.subtasks[0].question == "q"

    req = client.requests[0]
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in req and "thinking" not in req
    assert llm.usage.stats.llm_calls == 1 and llm.usage.stats.by_agent["planner"]["calls"] == 1


def test_fallbacks_can_be_disabled():
    client = FakeAnthropic(lambda p: make_message(text("hi")))
    ClaudeLLM(client, enable_fallbacks=False).complete(
        agent="a", system="s", prompt="p", model="m", effort="low"
    )
    assert "fallbacks" not in client.requests[0]


def test_invalid_structured_output_raises():
    client = FakeAnthropic(lambda p: make_message(text('{"objective": 1}')))
    with pytest.raises(LLMError):
        ClaudeLLM(client).structured(
            agent="a", system="s", prompt="p", schema=ResearchPlan, model="m", effort="low"
        )


def test_refusal_is_surfaced():
    client = FakeAnthropic(lambda p: make_message(stop_reason="refusal"))
    with pytest.raises(LLMRefusalError):
        ClaudeLLM(client).complete(agent="a", system="s", prompt="p", model="m", effort="low")


def test_tool_loop_runs_parallel_tools_then_finishes():
    turns = iter(
        [
            make_message(
                thinking(),
                tool_use("web_search", {"query": "a"}),
                tool_use("web_search", {"query": "b"}),
            ),
            make_message(thinking(), tool_use("submit_findings", FINDING)),
        ]
    )
    client = FakeAnthropic(lambda p: next(turns))
    executed = []

    def executor(name, args):
        executed.append(args["query"])
        return ToolOutcome(f"results for {args['query']}")

    result = ClaudeLLM(client).tool_loop(
        agent="researcher",
        system="s",
        prompt="p",
        tools=[SEARCH],
        finish_tool=FINISH,
        executor=executor,
        model="m",
        effort="medium",
        max_tool_calls=5,
    )
    assert result.stopped_reason == "finished" and result.final_input == FINDING
    assert sorted(executed) == ["a", "b"] and result.tool_calls == 2

    second = client.requests[1]["messages"]
    # assistant turn appended verbatim (thinking block first), then ONE user message with both results
    assert second[1]["role"] == "assistant" and second[1]["content"][0].type == "thinking"
    results = second[2]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert all(t["strict"] for t in client.requests[0]["tools"])


def test_tool_loop_enforces_budget_and_reports_tool_errors():
    calls = {"n": 0}

    def handler(params):
        calls["n"] += 1
        if calls["n"] <= 3:
            return make_message(tool_use("web_search", {"query": str(calls["n"])}))
        return make_message(tool_use("submit_findings", FINDING))

    client = FakeAnthropic(handler)

    def executor(name, args):
        raise RuntimeError("network down")

    result = ClaudeLLM(client).tool_loop(
        agent="r",
        system="s",
        prompt="p",
        tools=[SEARCH],
        finish_tool=FINISH,
        executor=executor,
        model="m",
        effort="low",
        max_tool_calls=2,
    )
    assert result.stopped_reason == "finished" and result.tool_calls == 2
    third_turn_results = client.requests[3]["messages"][-1]["content"]
    assert (
        third_turn_results[0]["is_error"] and "budget exhausted" in third_turn_results[0]["content"]
    )
    first_results = client.requests[1]["messages"][-1]["content"]
    assert first_results[0]["is_error"] and "network down" in first_results[0]["content"]


def test_tool_loop_nudges_once_then_gives_up():
    client = FakeAnthropic(lambda p: make_message(text("I think I'm done.")))
    result = ClaudeLLM(client).tool_loop(
        agent="r",
        system="s",
        prompt="p",
        tools=[SEARCH],
        finish_tool=FINISH,
        executor=lambda n, a: ToolOutcome(""),
        model="m",
        effort="low",
        max_tool_calls=3,
    )
    assert result.stopped_reason == "no_finish" and result.final_input is None
    assert len(client.requests) == 2
    assert "submit_findings" in client.requests[1]["messages"][-1]["content"]
