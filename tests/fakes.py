"""Deterministic test doubles: a scripted LLM, an in-memory search engine and page fetcher."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from research_assistant.llm import ToolLoopResult, ToolSpec, UsageTracker
from research_assistant.models import (
    Critique,
    CritiqueIssue,
    FindingDraft,
    PlannedSubTask,
    ReportDraft,
    ReportSection,
    ResearchPlan,
    Source,
)
from research_assistant.tools.fetch import FetchedPage, FetchError

PAGES = {
    "https://example.org/solar": (
        "Solar costs",
        "Utility solar LCOE fell 90% between 2010 and 2023. " * 5,
    ),
    "https://example.org/wind": ("Wind costs", "Onshore wind LCOE fell about 70% since 2010. " * 5),
    "https://example.org/storage": (
        "Storage",
        "Battery pack prices fell to $139/kWh in 2023. " * 400,
    ),
}


class FakeSearch:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, max_results: int) -> list[Source]:
        self.calls.append(query)
        return [
            Source(url=url, title=title, snippet=text[:60]) for url, (title, text) in PAGES.items()
        ][:max_results]


class FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        if url not in PAGES:
            raise FetchError(f"404 {url}")
        title, text = PAGES[url]
        return FetchedPage(url=url, title=title, text=text)


class _Usage:
    input_tokens = 100
    output_tokens = 50
    cache_read_input_tokens = 0


class FakeLLM:
    """Implements the LLMClient protocol with scripted, schema-aware responses.

    ``critic_scores`` controls the self-correction loop: the critic returns these scores in order
    (the last value repeats), with a follow-up task whenever the score is below threshold.
    """

    def __init__(
        self,
        critic_scores: list[float] | None = None,
        follow_up: str = "How have battery storage costs changed since 2015?",
        researcher_submits: bool = True,
        researcher_tool_script: list[tuple[str, dict[str, Any]]] | None = None,
        cite_url: str = "https://example.org/solar",
    ) -> None:
        self.usage = UsageTracker()
        self.critic_scores = critic_scores or [0.9]
        self.follow_up = follow_up
        self.researcher_submits = researcher_submits
        self.tool_script = researcher_tool_script or [
            ("web_search", {"query": "renewable energy cost trends"}),
            ("fetch_page", {"url": cite_url, "focus": "cost decline"}),
            ("recall_memory", {"query": "energy costs"}),
        ]
        self.cite_url = cite_url
        self.calls: list[tuple[str, str]] = []
        self._critic_round = 0
        self._lock = threading.Lock()

    def _log(self, agent: str, kind: str) -> None:
        with self._lock:
            self.calls.append((agent, kind))
        self.usage.record(agent, _Usage())

    def complete(self, *, agent, system, prompt, model, effort) -> str:
        self._log(agent, "complete")
        return "Condensed: battery pack prices fell to $139/kWh in 2023."

    def structured(self, *, agent, system, prompt, schema, model, effort):
        self._log(agent, f"structured:{schema.__name__}")
        if schema is ResearchPlan:
            return ResearchPlan(
                objective="Explain renewable cost trends",
                subtasks=[
                    PlannedSubTask(
                        question="How have solar costs changed since 2010?",
                        rationale="Solar is the largest driver",
                        search_queries=["solar LCOE 2010 2023"],
                    ),
                    PlannedSubTask(
                        question="How have onshore wind costs changed since 2010?",
                        rationale="Wind is the second driver",
                        search_queries=["onshore wind LCOE trend"],
                    ),
                    PlannedSubTask(  # near-duplicate of the first: must be deduplicated
                        question="How have solar costs changed since 2010",
                        rationale="dup",
                        search_queries=["solar cost"],
                    ),
                ],
                success_criteria=["Quantify cost declines"],
            )
        if schema is Critique:
            with self._lock:
                idx = min(self._critic_round, len(self.critic_scores) - 1)
                self._critic_round += 1
            score = self.critic_scores[idx]
            low = score < 0.75
            return Critique(
                score=score,
                approved=not low,
                strengths=["Quantified"],
                issues=[
                    CritiqueIssue(
                        severity="major", description="Storage missing", related_question="overall"
                    )
                ]
                if low
                else [],
                follow_up_tasks=[
                    PlannedSubTask(
                        question=self.follow_up, rationale="gap", search_queries=["battery prices"]
                    )
                ]
                if low
                else [],
            )
        if schema is ReportDraft:
            return ReportDraft(
                title="Renewable Energy Cost Trends",
                executive_summary="Costs fell sharply [1].",
                sections=[ReportSection(heading="Solar", content="Solar fell 90% [1].")],
                conclusion="Renewables are now cheap.",
                limitations=["Fake data"],
            )
        if schema is FindingDraft:
            return FindingDraft(
                summary="Recovered from transcript.",
                key_points=["recovered"],
                sources=[],
                confidence=0.3,
                gaps=["no submission"],
            )
        raise AssertionError(f"unexpected schema {schema}")

    def tool_loop(
        self,
        *,
        agent: str,
        system: str,
        prompt: str,
        tools: list[ToolSpec],
        finish_tool: ToolSpec,
        executor: Callable,
        model: str,
        effort: str,
        max_tool_calls: int,
        on_tool_call=None,
    ) -> ToolLoopResult:
        self._log(agent, "tool_loop")
        assert {t.name for t in tools} == {"web_search", "fetch_page", "recall_memory"}
        outputs = []
        for name, args in self.tool_script[:max_tool_calls]:
            if on_tool_call:
                on_tool_call(name, args)
            outputs.append(executor(name, args).content)
            self.usage.record_tool_call()
        if not self.researcher_submits:
            return ToolLoopResult(None, len(outputs), "\n".join(outputs), "no_finish")
        return ToolLoopResult(
            {
                "summary": "Costs fell substantially.",
                "key_points": ["Solar LCOE fell 90% 2010-2023"],
                "sources": [
                    {"url": self.cite_url, "title": "Solar costs"},
                    {"url": "https://hallucinated.example/never-seen", "title": "Fake"},
                ],
                "confidence": 0.8,
                "gaps": [],
            },
            len(outputs),
            "\n".join(outputs),
            "finished",
        )
