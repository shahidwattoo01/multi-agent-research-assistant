"""End-to-end tests of the LangGraph workflow with scripted agents."""

import json

from research_assistant.agents.critic import apply_review_policy
from research_assistant.graph import ResearchOrchestrator
from research_assistant.models import Critique, CritiqueIssue, PlannedSubTask
from tests.fakes import FakeFetcher, FakeLLM, FakeSearch


def _run(settings, llm, **kw):
    events = []
    orch = ResearchOrchestrator(settings, llm, FakeSearch(), FakeFetcher())
    report = orch.run("How have renewable energy costs changed?", on_event=events.append, **kw)
    return report, events


def test_approved_first_round_skips_revision(settings):
    report, events = _run(settings, FakeLLM(critic_scores=[0.9]))
    assert report.revisions == 0
    assert len(report.findings) == 2  # third planned task was a near-duplicate
    assert report.memory_stats["duplicate_tasks_skipped"] == 1
    types = [e.type for e in events]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert types.count("critique") == 1 and "report" in types


def test_critic_triggers_self_correction_round(settings):
    llm = FakeLLM(critic_scores=[0.4, 0.9])
    report, events = _run(settings, llm)
    assert report.revisions == 1
    assert [c.approved for c in report.critiques] == [False, True]
    revised = [f for f in report.findings if f.revision == 1]
    assert [f.question for f in revised] == ["How have battery storage costs changed since 2015?"]
    assert any(e.type == "revision" for e in events)


def test_revisions_are_bounded(settings):
    settings = settings.model_copy(update={"max_revisions": 1})
    llm = FakeLLM(critic_scores=[0.3])  # never satisfied
    llm_followups = iter(
        ["Battery storage price trend?", "Grid integration costs?", "Hydro costs?"]
    )
    original = llm.structured

    def structured(**kw):
        llm.follow_up = next(llm_followups, "x")
        return original(**kw)

    llm.structured = structured
    report, _ = _run(settings, llm)
    assert len(report.critiques) == 2  # initial review + 1 revision, then forced to write


def test_duplicate_follow_ups_do_not_loop(settings):
    # The critic keeps asking a question that is already in the ledger -> no new work -> write.
    llm = FakeLLM(critic_scores=[0.3], follow_up="How have solar costs changed since 2010?")
    report, events = _run(settings, llm)
    assert len(report.critiques) == 1
    assert len(report.findings) == 2
    assert report.memory_stats["duplicate_tasks_skipped"] == 2


def test_shared_memory_reduces_redundant_tool_work(settings):
    search, fetcher = FakeSearch(), FakeFetcher()
    orch = ResearchOrchestrator(settings, FakeLLM(), search, fetcher)
    report = orch.run("q?")
    # Both researchers issue the same query and fetch the same page: only one real call each.
    assert search.calls == ["renewable energy cost trends"]
    assert fetcher.calls == ["https://example.org/solar"]
    assert report.memory_stats["search_cache_hits"] >= 1


def test_hallucinated_sources_are_removed(settings):
    report, _ = _run(settings, FakeLLM())
    for finding in report.findings:
        assert [s.url for s in finding.sources] == ["https://example.org/solar"]
        assert any("could not be verified" in g for g in finding.gaps)
        assert finding.confidence < 0.8


def test_missing_submission_recovers_via_extraction(settings):
    report, events = _run(settings, FakeLLM(researcher_submits=False))
    assert all(f.summary == "Recovered from transcript." for f in report.findings)
    assert any(e.type == "recovery" for e in events)


def test_report_markdown_and_persistence(settings):
    report, _ = _run(settings, FakeLLM())
    md = report.to_markdown()
    assert md.startswith("# Renewable Energy Cost Trends")
    assert "## Sources" in md and "1. [Solar costs](https://example.org/solar)" in md
    run_file = settings.memory_dir / "runs" / f"{report.run_id}.json"
    saved = json.loads(run_file.read_text())
    assert saved["progress"]["done"] == 2
    assert (settings.memory_dir / "cache.json").exists()
    assert report.usage.llm_calls > 0 and report.usage.tool_calls == 6


def test_sequential_mode(settings):
    settings = settings.model_copy(update={"parallel_research": False})
    report, _ = _run(settings, FakeLLM())
    assert len(report.findings) == 2


def test_review_policy():
    issue = CritiqueIssue(severity="critical", description="x", related_question="overall")
    follow = [PlannedSubTask(question="q", rationale="r", search_queries=[])]
    high_but_critical = Critique(
        score=0.95, approved=True, strengths=[], issues=[issue], follow_up_tasks=follow
    )
    assert not apply_review_policy(high_but_critical, 0.75, final_round=False).approved

    low_no_followups = Critique(
        score=0.2, approved=False, strengths=[], issues=[], follow_up_tasks=[]
    )
    assert apply_review_policy(low_no_followups, 0.75, final_round=False).approved

    final = apply_review_policy(
        Critique(score=0.2, approved=False, strengths=[], issues=[], follow_up_tasks=follow),
        0.75,
        True,
    )
    assert not final.approved and final.follow_up_tasks == []
