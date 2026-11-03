"""Alternative orchestration backend built on Microsoft AutoGen (``autogen-agentchat`` >= 0.7).

Planner, researcher and critic are AutoGen ``AssistantAgent``s in a ``SelectorGroupChat``.
A deterministic ``selector_func`` sets the turn order, so no extra LLM call is spent choosing
the next speaker. The agents share state through the same :class:`SharedMemory` the LangGraph
backend uses, exposed to them as tools:

* planner    -> ``register_plan``            (writes sub-tasks to the ledger)
* researcher -> ``get_pending_tasks``, ``web_search``, ``fetch_page``, ``recall_memory``,
                ``submit_finding``          (reads/updates ledger, publishes findings)
* critic     -> ``submit_review``            (scores; adds de-duplicated follow-up tasks)

The final report is written by the backend-agnostic :class:`WriterAgent` over shared memory, so
both backends produce the same :class:`ResearchReport`.

Install with ``pip install "research-assistant[autogen]"``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from research_assistant.agents import SummarizerAgent, WriterAgent
from research_assistant.agents.critic import MAX_FOLLOW_UPS, apply_review_policy
from research_assistant.config import Settings, get_settings
from research_assistant.events import Event, EventBus
from research_assistant.graph import ResearchOrchestrator, _usage_delta
from research_assistant.llm import LLMRefusalError, UsageTracker
from research_assistant.memory import SharedMemory, normalize_url
from research_assistant.models import (
    Critique,
    CritiqueIssue,
    Finding,
    PlannedSubTask,
    ResearchPlan,
    ResearchReport,
    SourceRef,
    TaskStatus,
)
from research_assistant.tools import ResearchToolbox

logger = logging.getLogger(__name__)

_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


# ---------------------------------------------------------------------------
# Adapter: make AutoGen's Anthropic client speak the current Messages API
# ---------------------------------------------------------------------------


def _block_get(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def _content_key(content: Any) -> tuple | None:
    """Identify an assistant turn by its tool_use ids, or by its text if it has none."""
    if isinstance(content, str):
        return ("text", content.strip())
    if not isinstance(content, list):
        return None
    ids = tuple(_block_get(b, "id") for b in content if _block_get(b, "type") == "tool_use")
    if ids:
        return ("tools", ids)
    text = "".join(_block_get(b, "text") or "" for b in content if _block_get(b, "type") == "text")
    return ("text", text.strip())


class _MessagesAdapter:
    """Wraps ``AsyncAnthropic.messages`` as used by ``AnthropicChatCompletionClient``.

    * drops sampling parameters (current Claude models reject them) and sets ``effort``;
    * replays each assistant turn with its *original* content blocks (thinking blocks
      included). AutoGen rebuilds assistant turns from its own message types, which loses
      thinking blocks and can add empty text blocks. Restoring the exact blocks keeps the
      history append-only, as adaptive thinking with tool use requires;
    * records usage and surfaces refusals as :class:`LLMRefusalError`.
    """

    def __init__(self, inner: Any, *, agent: str, effort: str, usage: UsageTracker) -> None:
        self._inner = inner
        self._agent = agent
        self._effort = effort
        self._usage = usage
        self._originals: dict[tuple, list[Any]] = {}

    async def create(self, **params: Any) -> Any:
        for key in _SAMPLING_PARAMS:
            params.pop(key, None)
        params.setdefault("output_config", {"effort": self._effort})
        params["messages"] = [self._restore(m) for m in params.get("messages", [])]
        response = await self._inner.create(**params)
        if params.get("stream"):
            return response
        self._usage.record(self._agent, response.usage)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(getattr(details, "category", None))
        key = _content_key(list(response.content))
        if key is not None:
            self._originals[key] = list(response.content)
        return response

    def _restore(self, message: Any) -> Any:
        if _block_get(message, "role") != "assistant":
            return message
        key = _content_key(_block_get(message, "content"))
        original = self._originals.get(key) if key else None
        return {"role": "assistant", "content": original} if original else message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ClientAdapter:
    def __init__(self, inner: Any, **kwargs: Any) -> None:
        self._inner = inner
        self.messages = _MessagesAdapter(inner.messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def make_model_client(
    settings: Settings,
    *,
    agent: str,
    model: str,
    effort: str,
    usage: UsageTracker,
    anthropic_client: Any | None = None,
):
    """Build an AutoGen ``AnthropicChatCompletionClient`` wired through :class:`_ClientAdapter`.

    ``anthropic_client`` replaces the underlying ``AsyncAnthropic`` (used for tests).
    """
    from autogen_core.models import ModelFamily, ModelInfo
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient

    client = AnthropicChatCompletionClient(
        model=model,
        max_tokens=settings.max_tokens,
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=False,
            structured_output=False,
            family=ModelFamily.CLAUDE_4_OPUS,
            multiple_system_messages=False,
        ),
    )
    inner = anthropic_client or client._client
    client._client = _ClientAdapter(inner, agent=agent, effort=effort, usage=usage)
    return client


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLANNER = """\
You are the Planner of a research team (Planner, Researcher, Critic). Decompose the user's research
question into at most {max_subtasks} focused, non-overlapping sub-questions answerable from public
web sources. Call `register_plan` exactly once, then reply with a one-paragraph summary of the plan.
"""

_RESEARCHER = """\
You are the Researcher of a research team. Call `get_pending_tasks` to see the sub-tasks assigned
to you. For EACH pending task: search the web, read the most relevant pages with `fetch_page`
(snippets alone are not evidence), use `recall_memory` to reuse teammates' findings, then call
`submit_finding` with a grounded summary, key facts and the URLs you actually read. Budget about
{per_task} tool calls per task. When no tasks are pending, reply with a brief status summary.
"""

_CRITIC = """\
You are the Critic of a research team. Call `get_findings` to read every published finding, then
review them against the research question:
coverage, evidence quality, consistency and gaps. Then call `submit_review` exactly once:
score 0.0-1.0; approve when score >= {threshold} and nothing critical is missing; otherwise give at
most {max_follow_ups} follow-up questions that target genuine gaps and do NOT repeat sub-questions
already investigated. After the tool call, reply with a short verdict.
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AutoGenOrchestrator(ResearchOrchestrator):
    backend = "autogen"
    anthropic_client: Any | None = None  # optional AsyncAnthropic override (tests, proxies)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **kwargs: Any) -> AutoGenOrchestrator:
        return super().from_settings(settings or get_settings(), **kwargs)  # type: ignore[return-value]

    def run(
        self,
        question: str,
        *,
        on_event: Callable[[Event], None] | None = None,
        run_id: str | None = None,
    ) -> ResearchReport:
        return asyncio.run(self.arun(question, on_event=on_event, run_id=run_id))

    async def arun(
        self,
        question: str,
        *,
        on_event: Callable[[Event], None] | None = None,
        run_id: str | None = None,
    ) -> ResearchReport:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import ExternalTermination, MaxMessageTermination
        from autogen_agentchat.messages import BaseChatMessage
        from autogen_agentchat.teams import SelectorGroupChat

        s = self.settings
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        events = EventBus()
        if on_event:
            events.subscribe(on_event)
        memory = self.new_memory()
        usage_before = self.llm.usage.snapshot()
        toolbox = ResearchToolbox(
            memory,
            self.search,
            self.fetcher,
            SummarizerAgent(self.llm, s, events).summarize,
            events=events,
            max_results=s.search_results,
            summarize_threshold=s.summarize_threshold_chars,
        )
        state = _RunState()
        stop = ExternalTermination()
        tools = _build_tools(question, s, memory, toolbox, events, state, stop)

        def client(agent: str, model: str, effort: str):
            return make_model_client(
                s,
                agent=agent,
                model=model,
                effort=effort,
                usage=self.llm.usage,
                anthropic_client=self.anthropic_client,
            )

        planner = AssistantAgent(
            "planner",
            client("planner", s.model, s.planner_effort),
            tools=[tools["register_plan"]],
            system_message=_PLANNER.format(max_subtasks=s.max_subtasks),
            description="Decomposes the question into sub-tasks.",
            max_tool_iterations=3,
            reflect_on_tool_use=True,
        )
        researcher = AssistantAgent(
            "researcher",
            client("researcher", s.worker_model, s.researcher_effort),
            tools=[
                tools[n]
                for n in (
                    "get_pending_tasks",
                    "web_search",
                    "fetch_page",
                    "recall_memory",
                    "submit_finding",
                )
            ],
            system_message=_RESEARCHER.format(per_task=s.max_tool_calls_per_task),
            description="Researches pending sub-tasks with web tools.",
            max_tool_iterations=min(60, (s.max_tool_calls_per_task + 2) * s.max_subtasks),
            reflect_on_tool_use=True,
        )
        critic = AssistantAgent(
            "critic",
            client("critic", s.model, s.critic_effort),
            tools=[tools["get_findings"], tools["submit_review"]],
            system_message=_CRITIC.format(
                threshold=s.approval_threshold, max_follow_ups=MAX_FOLLOW_UPS
            ),
            description="Reviews findings and requests follow-up research.",
            max_tool_iterations=4,
            reflect_on_tool_use=True,
        )

        def select(messages) -> str | None:
            last = next(
                (m.source for m in reversed(messages) if isinstance(m, BaseChatMessage)), "user"
            )
            if last == "user":
                return "planner"
            if last in ("planner", "critic"):
                if memory.pending_tasks():
                    events.emit(
                        "agent_started",
                        "researcher",
                        f"{len(memory.pending_tasks())} pending tasks",
                    )
                    return "researcher"
                return "critic"
            if last == "researcher":
                state.researcher_turns += 1
                if memory.pending_tasks() and state.researcher_turns < 2 * (s.max_revisions + 2):
                    return "researcher"
                return "critic"
            return None

        team = SelectorGroupChat(
            [planner, researcher, critic],
            model_client=client("selector", s.model, "low"),  # unused: selector_func decides
            selector_func=select,
            termination_condition=stop | MaxMessageTermination(8 * (s.max_revisions + 2)),
            allow_repeated_speaker=True,
        )

        events.emit("run_started", "orchestrator", question, run_id=run_id, backend=self.backend)
        try:
            async for message in team.run_stream(task=question):
                if isinstance(message, BaseChatMessage) and message.source != "user":
                    text = message.to_text()
                    events.emit("message", message.source, text[:300])
            plan = state.plan or ResearchPlan(objective=question, subtasks=[], success_criteria=[])
            findings = memory.all_findings()
            critique = state.critiques[-1] if state.critiques else None
            writer = WriterAgent(self.llm, s, events)
            draft, sources = await asyncio.to_thread(
                writer.write, question, plan, findings, critique
            )
        finally:
            self._persist(memory, run_id)

        report = ResearchReport(
            **draft.model_dump(),
            run_id=run_id,
            question=question,
            backend=self.backend,
            sources=sources,
            findings=findings,
            critiques=state.critiques,
            revisions=max(0, len(state.critiques) - 1),
            usage=_usage_delta(usage_before, self.llm.usage),
            memory_stats=dict(memory.stats),
        )
        events.emit("run_finished", "orchestrator", report.title, run_id=run_id)
        return report


class _RunState:
    def __init__(self) -> None:
        self.plan: ResearchPlan | None = None
        self.critiques: list[Critique] = []
        self.researcher_turns = 0


def _findings_brief(memory: SharedMemory) -> str:
    lines = []
    for f in memory.all_findings():
        lines.append(
            f"- Q: {f.question}\n  A: {f.summary}\n  confidence={f.confidence:.2f}; "
            f"sources={len(f.sources)}; gaps={'; '.join(f.gaps) or 'none'}"
        )
    return "\n".join(lines) or "(no findings)"


def _build_tools(
    question: str,
    s: Settings,
    memory: SharedMemory,
    toolbox: ResearchToolbox,
    events: EventBus,
    state: _RunState,
    stop: Any,
) -> dict[str, Any]:
    """Create the async tool functions handed to AutoGen agents (schemas come from type hints)."""

    async def register_plan(
        objective: str, sub_questions: list[str], success_criteria: list[str]
    ) -> str:
        """Register the research plan: the objective, the sub-questions to research, and success criteria."""
        planned = [
            PlannedSubTask(question=q, rationale="", search_queries=[])
            for q in sub_questions[: s.max_subtasks]
        ]
        state.plan = ResearchPlan(
            objective=objective, subtasks=planned, success_criteria=success_criteria
        )
        created = memory.add_tasks(planned, origin="planner")
        events.emit(
            "plan",
            "planner",
            f"Planned {len(created)} sub-tasks",
            subtasks=[t.question for t in created],
        )
        return f"Registered {len(created)} sub-tasks ({len(planned) - len(created)} duplicates skipped)."

    async def get_pending_tasks() -> str:
        """List sub-tasks that still need research, with their task ids."""
        pending = memory.pending_tasks()
        if not pending:
            return "No pending tasks."
        return "\n".join(f"{t.id}: {t.question}" for t in pending)

    async def web_search(query: str) -> str:
        """Search the web; returns titles, URLs and snippets."""
        events.emit("tool_call", "researcher", f"web_search({query!r})", tool="web_search")
        return (await asyncio.to_thread(toolbox.web_search, query)).content

    async def fetch_page(url: str, focus: str) -> str:
        """Read a web page; long pages are condensed to what is relevant to `focus`."""
        events.emit("tool_call", "researcher", f"fetch_page({url!r})", tool="fetch_page")
        return (await asyncio.to_thread(toolbox.fetch_page, url, focus)).content

    async def recall_memory(query: str) -> str:
        """Look up findings already published by teammates."""
        return toolbox.recall_memory(query).content

    async def submit_finding(
        task_id: str,
        summary: str,
        key_points: list[str],
        source_urls: list[str],
        confidence: float,
        gaps: list[str],
    ) -> str:
        """Publish the finding for one sub-task (use the id from get_pending_tasks) and mark it done."""
        task = memory.tasks.get(task_id)
        if task is None:
            return f"Unknown task id {task_id!r}. Call get_pending_tasks for valid ids."
        known = memory.known_sources()
        refs = [
            SourceRef(url=u, title=known.get(normalize_url(u), ""))
            for u in source_urls
            if normalize_url(u) in known
        ]
        dropped = len(source_urls) - len(refs)
        if dropped:
            gaps = [*gaps, f"{dropped} cited source(s) could not be verified and were removed."]
        finding = Finding(
            task_id=task.id,
            question=task.question,
            summary=summary,
            key_points=key_points,
            sources=refs,
            confidence=confidence,
            gaps=gaps,
            revision=task.revision,
        )
        memory.publish_finding(finding)
        memory.set_status(task.id, TaskStatus.DONE)
        events.emit(
            "finding",
            "researcher",
            f"{task.question} -> confidence {finding.confidence:.2f}",
            task_id=task.id,
        )
        remaining = len(memory.pending_tasks())
        return f"Finding recorded. {remaining} task(s) still pending."

    async def get_findings() -> str:
        """Read all findings published to shared memory, with confidence, source counts and gaps."""
        return _findings_brief(memory)

    async def submit_review(
        score: float,
        approved: bool,
        strengths: list[str],
        issues: list[str],
        follow_up_questions: list[str],
    ) -> str:
        """Submit the review of all findings: score, verdict, issues, and follow-up questions for gaps."""
        revision = len(state.critiques)
        final_round = revision >= s.max_revisions
        critique = apply_review_policy(
            Critique(
                score=score,
                approved=approved,
                strengths=strengths,
                issues=[
                    CritiqueIssue(severity="major", description=i, related_question="overall")
                    for i in issues
                ],
                follow_up_tasks=[
                    PlannedSubTask(question=q, rationale="critic follow-up", search_queries=[])
                    for q in follow_up_questions
                ],
            ),
            s.approval_threshold,
            final_round,
        )
        state.critiques.append(critique)
        new = []
        if not critique.approved:
            new = memory.add_tasks(critique.follow_up_tasks, origin="critic", revision=revision + 1)
        events.emit(
            "critique",
            "critic",
            f"score {critique.score:.2f} - {'approved' if critique.approved else f'{len(new)} new follow-ups'}",
            score=critique.score,
            approved=critique.approved,
        )
        if critique.approved or final_round or not new:
            stop.set()
            return "Review recorded. Research is complete; the writer will now produce the report."
        return f"Review recorded. {len(new)} follow-up task(s) queued for the researcher."

    return {
        fn.__name__: fn
        for fn in (
            register_plan,
            get_pending_tasks,
            web_search,
            fetch_page,
            recall_memory,
            submit_finding,
            get_findings,
            submit_review,
        )
    }
