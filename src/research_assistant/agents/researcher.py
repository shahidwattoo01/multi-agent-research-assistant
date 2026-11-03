from __future__ import annotations

import logging

from research_assistant.agents.base import Agent
from research_assistant.config import Settings
from research_assistant.events import EventBus
from research_assistant.llm import LLMClient, LLMError, ToolSpec
from research_assistant.memory import SharedMemory, normalize_url
from research_assistant.models import Finding, FindingDraft, SubTask, TaskStatus
from research_assistant.prompts import RESEARCHER_SYSTEM
from research_assistant.tools.toolbox import ResearchToolbox

logger = logging.getLogger(__name__)

SUBMIT_FINDINGS = ToolSpec.from_model(
    "submit_findings",
    "Submit your final, source-grounded findings for the sub-question. Call exactly once, last.",
    FindingDraft,
)


class ResearcherAgent(Agent):
    """Investigates one sub-task with web search / page reading and publishes a Finding."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        memory: SharedMemory,
        toolbox: ResearchToolbox,
        events: EventBus | None = None,
    ) -> None:
        super().__init__(llm, settings, events)
        self.memory = memory
        self.toolbox = toolbox

    def research(self, task: SubTask, question: str) -> Finding:
        self.memory.set_status(task.id, TaskStatus.IN_PROGRESS)
        self.emit("task_started", task.question, task_id=task.id, origin=task.origin)
        try:
            finding = self._investigate(task, question)
        except LLMError as exc:
            logger.warning("research failed for %s: %s", task.id, exc)
            self.memory.set_status(task.id, TaskStatus.FAILED, error=str(exc))
            self.emit("task_failed", str(exc), task_id=task.id)
            finding = Finding(
                task_id=task.id,
                question=task.question,
                summary="Research for this sub-question failed.",
                key_points=[],
                sources=[],
                confidence=0.0,
                gaps=[f"Research failed: {exc}"],
                revision=task.revision,
            )
            self.memory.publish_finding(finding)
            return finding

        self.memory.publish_finding(finding)
        self.memory.set_status(task.id, TaskStatus.DONE)
        self.emit(
            "finding",
            f"{task.question} -> confidence {finding.confidence:.2f}, {len(finding.sources)} sources",
            task_id=task.id,
            confidence=finding.confidence,
            sources=[s.url for s in finding.sources],
        )
        return finding

    def _investigate(self, task: SubTask, question: str) -> Finding:
        queries = "\n".join(f"- {q}" for q in task.search_queries) or "- (choose your own)"
        prompt = (
            f"<main_question>{question}</main_question>\n"
            f"<sub_question>{task.question}</sub_question>\n"
            f"<why_it_matters>{task.rationale}</why_it_matters>\n"
            f"<suggested_queries>\n{queries}\n</suggested_queries>\n"
            f"<tool_budget>{self.settings.max_tool_calls_per_task} tool calls</tool_budget>\n\n"
            "Investigate the sub-question, then call submit_findings."
        )

        result = self.llm.tool_loop(
            agent=self.name,
            system=RESEARCHER_SYSTEM,
            prompt=prompt,
            tools=self.toolbox.specs,
            finish_tool=SUBMIT_FINDINGS,
            executor=lambda name, args: self.toolbox.execute(name, args, task_id=task.id),
            model=self.settings.worker_model,
            effort=self.settings.researcher_effort,
            max_tool_calls=self.settings.max_tool_calls_per_task,
            on_tool_call=lambda name, args: self.emit(
                "tool_call", f"{name}({_short(args)})", task_id=task.id, tool=name, args=args
            ),
        )

        if result.final_input is not None:
            draft = FindingDraft.model_validate(result.final_input)
        else:
            # Self-correction: the loop ended without a submission -> extract from the transcript.
            self.emit(
                "recovery", "No submission; extracting findings from transcript", task_id=task.id
            )
            draft = self.llm.structured(
                agent=self.name,
                system=RESEARCHER_SYSTEM,
                prompt=(
                    f"<sub_question>{task.question}</sub_question>\n<research_notes>\n"
                    f"{result.transcript_text or '(no notes)'}\n</research_notes>\n\n"
                    "Produce findings from these notes only. Lower confidence if evidence is thin."
                ),
                schema=FindingDraft,
                model=self.settings.worker_model,
                effort=self.settings.researcher_effort,
            )

        draft = self._verify_sources(draft)
        return Finding(
            **draft.model_dump(),
            task_id=task.id,
            question=task.question,
            tool_calls=result.tool_calls,
            revision=task.revision,
        )

    def _verify_sources(self, draft: FindingDraft) -> FindingDraft:
        """Drop citations of URLs no tool ever returned (guards against hallucinated sources)."""
        known = self.memory.known_sources()
        kept = [s for s in draft.sources if normalize_url(s.url) in known]
        dropped = len(draft.sources) - len(kept)
        if not dropped:
            return draft
        gaps = [*draft.gaps, f"{dropped} cited source(s) could not be verified and were removed."]
        penalty = 0.15 * dropped / max(1, len(draft.sources))
        return draft.model_copy(
            update={
                "sources": kept,
                "gaps": gaps,
                "confidence": max(0.0, draft.confidence - penalty),
            }
        )


def _short(args: dict, limit: int = 80) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= limit else text[: limit - 1] + "…"
