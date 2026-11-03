"""LangGraph orchestration.

    START -> planner -> research (fan-out, one branch per pending sub-task) -> critic
                             ^                                                  |
                             +------- follow-up sub-tasks (not approved) -------+
                                                                                 |
                                                         approved / budget hit -> writer -> END

The planner's sub-tasks are dispatched in parallel with LangGraph's ``Send`` API. Findings are
merged back into graph state with an additive reducer. The critic either approves or adds
follow-up sub-tasks to the shared task ledger. Duplicates of earlier sub-tasks are dropped there,
so a revision round only researches what is genuinely new.
"""

from __future__ import annotations

import logging
import operator
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_assistant.agents import (
    CriticAgent,
    PlannerAgent,
    ResearcherAgent,
    SummarizerAgent,
    WriterAgent,
)
from research_assistant.config import Settings, get_settings
from research_assistant.events import Event, EventBus
from research_assistant.llm import ClaudeLLM, LLMClient, UsageTracker
from research_assistant.memory import SharedMemory
from research_assistant.models import (
    Critique,
    Finding,
    ReportDraft,
    ResearchPlan,
    ResearchReport,
    Source,
    SubTask,
)
from research_assistant.tools import (
    HttpPageFetcher,
    PageFetcher,
    ResearchToolbox,
    SearchProvider,
    build_search_provider,
)

logger = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    question: str
    plan: ResearchPlan
    findings: Annotated[list[Finding], operator.add]
    critiques: Annotated[list[Critique], operator.add]
    revision: int
    draft: ReportDraft
    sources: list[Source]


class ResearchTaskInput(TypedDict):
    question: str
    task: SubTask


@dataclass
class ResearchTeam:
    memory: SharedMemory
    planner: PlannerAgent
    researcher: ResearcherAgent
    critic: CriticAgent
    writer: WriterAgent


def build_graph(team: ResearchTeam, settings: Settings):
    """Compile the planner -> researchers -> critic -> writer state graph."""
    memory = team.memory

    def planner_node(state: ResearchState) -> dict[str, Any]:
        plan = team.planner.plan(state["question"])
        memory.add_tasks(plan.subtasks, origin="planner", revision=0)
        return {"plan": plan, "revision": 0}

    def research_node(payload: ResearchTaskInput) -> dict[str, Any]:
        finding = team.researcher.research(payload["task"], payload["question"])
        return {"findings": [finding]}

    def critic_node(state: ResearchState) -> dict[str, Any]:
        revision = state.get("revision", 0)
        critique = team.critic.review(
            state["question"],
            state["plan"],
            state.get("findings", []),
            list(memory.tasks.values()),
            revision,
        )
        if not critique.approved and critique.follow_up_tasks:
            new = memory.add_tasks(critique.follow_up_tasks, origin="critic", revision=revision + 1)
            team.critic.emit(
                "revision",
                f"Round {revision + 2}: {len(new)} new sub-task(s), "
                f"{len(critique.follow_up_tasks) - len(new)} duplicate(s) skipped",
                new_tasks=[t.question for t in new],
            )
        return {"critiques": [critique], "revision": revision + 1}

    def writer_node(state: ResearchState) -> dict[str, Any]:
        critiques = state.get("critiques", [])
        draft, sources = team.writer.write(
            state["question"],
            state["plan"],
            state.get("findings", []),
            critiques[-1] if critiques else None,
        )
        return {"draft": draft, "sources": sources}

    def dispatch(state: ResearchState) -> list[Send] | str:
        critiques = state.get("critiques", [])
        if critiques and critiques[-1].approved:
            return "writer"
        pending = memory.pending_tasks()
        if not pending:
            return "writer" if critiques else "critic"
        return [Send("research", {"question": state["question"], "task": t}) for t in pending]

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch, ["research", "critic", "writer"])
    graph.add_edge("research", "critic")
    graph.add_conditional_edges("critic", dispatch, ["research", "writer"])
    graph.add_edge("writer", END)
    return graph.compile()


class ResearchOrchestrator:
    """Wires settings, LLM, tools and shared memory into a runnable research pipeline."""

    backend = "langgraph"

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        search: SearchProvider,
        fetcher: PageFetcher,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.search = search
        self.fetcher = fetcher

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        search: SearchProvider | None = None,
        fetcher: PageFetcher | None = None,
    ) -> ResearchOrchestrator:
        settings = settings or get_settings()
        return cls(
            settings,
            llm
            or ClaudeLLM(
                max_tokens=settings.max_tokens, enable_fallbacks=settings.enable_fallbacks
            ),
            search or build_search_provider(settings.search_provider, settings.tavily_api_key),
            fetcher or HttpPageFetcher(settings.fetch_timeout, settings.max_page_chars),
        )

    def new_memory(self) -> SharedMemory:
        cache = self.settings.memory_dir / "cache.json" if self.settings.persist_memory else None
        return SharedMemory(
            cache_path=cache, duplicate_threshold=self.settings.duplicate_similarity
        )

    def build_team(self, memory: SharedMemory, events: EventBus) -> ResearchTeam:
        s, llm = self.settings, self.llm
        summarizer = SummarizerAgent(llm, s, events)
        toolbox = ResearchToolbox(
            memory,
            self.search,
            self.fetcher,
            summarizer.summarize,
            events=events,
            max_results=s.search_results,
            summarize_threshold=s.summarize_threshold_chars,
        )
        return ResearchTeam(
            memory=memory,
            planner=PlannerAgent(llm, s, events),
            researcher=ResearcherAgent(llm, s, memory, toolbox, events),
            critic=CriticAgent(llm, s, events),
            writer=WriterAgent(llm, s, events),
        )

    def run(
        self,
        question: str,
        *,
        on_event: Callable[[Event], None] | None = None,
        run_id: str | None = None,
    ) -> ResearchReport:
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        events = EventBus()
        if on_event:
            events.subscribe(on_event)
        memory = self.new_memory()
        team = self.build_team(memory, events)
        graph = build_graph(team, self.settings)

        usage_before = self.llm.usage.snapshot()
        events.emit("run_started", "orchestrator", question, run_id=run_id, backend=self.backend)
        try:
            state: ResearchState = graph.invoke(
                {"question": question},
                config={
                    "max_concurrency": 4 if self.settings.parallel_research else 1,
                    "recursion_limit": 10 + 3 * (self.settings.max_revisions + 1),
                    "run_name": run_id,
                },
            )
        finally:
            self._persist(memory, run_id)

        report = ResearchReport(
            **state["draft"].model_dump(),
            run_id=run_id,
            question=question,
            backend=self.backend,
            sources=state.get("sources", []),
            findings=state.get("findings", []),
            critiques=state.get("critiques", []),
            revisions=max(0, len(state.get("critiques", [])) - 1),
            usage=_usage_delta(usage_before, self.llm.usage),
            memory_stats=dict(memory.stats),
        )
        events.emit("run_finished", "orchestrator", report.title, run_id=run_id)
        return report

    def _persist(self, memory: SharedMemory, run_id: str) -> None:
        if not self.settings.persist_memory:
            return
        try:
            memory.save_cache()
            memory.save_run(self.settings.memory_dir / "runs" / f"{run_id}.json")
        except OSError as exc:
            logger.warning("could not persist memory: %s", exc)


def _usage_delta(before, tracker: UsageTracker):
    after = tracker.snapshot()
    by_agent = {}
    for agent, counts in after.by_agent.items():
        prev = before.by_agent.get(agent, {})
        delta = {k: v - prev.get(k, 0) for k, v in counts.items()}
        if delta.get("calls"):
            by_agent[agent] = delta
    return after.model_copy(
        update={
            "by_agent": by_agent,
            "llm_calls": after.llm_calls - before.llm_calls,
            "input_tokens": after.input_tokens - before.input_tokens,
            "output_tokens": after.output_tokens - before.output_tokens,
            "cache_read_tokens": after.cache_read_tokens - before.cache_read_tokens,
            "tool_calls": after.tool_calls - before.tool_calls,
        }
    )
