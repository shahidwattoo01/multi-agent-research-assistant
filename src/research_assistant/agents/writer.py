from __future__ import annotations

from research_assistant.agents.base import Agent, collect_sources, render_findings, render_plan
from research_assistant.memory import normalize_url
from research_assistant.models import Critique, Finding, ReportDraft, ResearchPlan, Source
from research_assistant.prompts import WRITER_SYSTEM


class WriterAgent(Agent):
    """Synthesizes approved findings into a cited report."""

    name = "writer"

    def write(
        self,
        question: str,
        plan: ResearchPlan,
        findings: list[Finding],
        critique: Critique | None,
    ) -> tuple[ReportDraft, list[Source]]:
        self.emit("agent_started", f"Writing report from {len(findings)} findings")
        sources = collect_sources(findings)
        index = {normalize_url(s.url): i for i, s in enumerate(sources, start=1)}
        source_list = "\n".join(f"[{i}] {s.title} — {s.url}" for i, s in enumerate(sources, 1))
        notes = ""
        if critique is not None and critique.issues:
            notes = "\n".join(f"- ({i.severity}) {i.description}" for i in critique.issues)
            notes = f"<reviewer_notes>\n{notes}\n</reviewer_notes>\n"

        draft = self.llm.structured(
            agent=self.name,
            system=WRITER_SYSTEM,
            prompt=(
                f"<research_question>{question}</research_question>\n{render_plan(plan)}\n"
                f"<sources>\n{source_list or '(none)'}\n</sources>\n"
                f"<findings>\n{render_findings(findings, index)}\n</findings>\n{notes}\n"
                "Write the report."
            ),
            schema=ReportDraft,
            model=self.settings.model,
            effort=self.settings.writer_effort,
        )
        self.emit("report", f"Report drafted: {draft.title}", sections=len(draft.sections))
        return draft, sources
