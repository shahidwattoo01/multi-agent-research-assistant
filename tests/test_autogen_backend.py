"""The AutoGen backend, driven by a scripted async Anthropic client (no network)."""

import pytest

pytest.importorskip("autogen_agentchat")
pytest.importorskip("autogen_ext.models.anthropic")

from research_assistant.autogen_backend import AutoGenOrchestrator, _content_key  # noqa: E402
from tests.fake_anthropic import ScriptedAsyncAnthropic  # noqa: E402
from tests.fakes import FakeFetcher, FakeLLM, FakeSearch  # noqa: E402


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _orchestrator(settings, scores):
    orch = AutoGenOrchestrator(settings, FakeLLM(), FakeSearch(), FakeFetcher())
    orch.anthropic_client = ScriptedAsyncAnthropic(scores)
    return orch


def test_autogen_team_runs_with_revision_round(settings):
    orch = _orchestrator(settings, [0.5, 0.9])
    events = []
    report = orch.run("How have renewable energy costs changed?", on_event=events.append)

    assert report.backend == "autogen"
    assert [c.approved for c in report.critiques] == [False, True]
    assert {f.question for f in report.findings} == {
        "How have solar costs changed since 2010?",
        "How have onshore wind costs changed since 2010?",
        "How have battery storage costs changed since 2015?",
    }
    # unverifiable URL cited by the model was stripped
    assert all(s.url == "https://example.org/solar" for f in report.findings for s in f.sources)
    assert {"planner", "researcher", "critic", "writer"} <= set(report.usage.by_agent)
    assert any(e.type == "critique" for e in events)


def test_autogen_approves_first_round(settings):
    report = _orchestrator(settings, [0.95]).run("q?")
    assert len(report.critiques) == 1 and report.critiques[0].approved
    assert len(report.findings) == 2


def test_content_key_matches_dicts_and_objects():
    from anthropic.types import TextBlock, ToolUseBlock

    as_objects = [
        TextBlock(type="text", text="hi"),
        ToolUseBlock(type="tool_use", id="t1", name="x", input={}),
    ]
    as_dicts = [
        {"type": "text", "text": ""},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
    ]
    assert _content_key(as_objects) == _content_key(as_dicts) == ("tools", ("t1",))
    assert _content_key("  done ") == ("text", "done")
