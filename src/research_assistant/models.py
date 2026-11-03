"""Typed data contracts exchanged between agents.

Every LLM-produced object is validated against one of these models, so agents
communicate through explicit schemas instead of free-form text.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlannedSubTask(BaseModel):
    """A sub-question proposed by the planner (or by the critic as a follow-up)."""

    question: str = Field(description="A focused, independently researchable sub-question.")
    rationale: str = Field(description="Why answering this helps answer the main question.")
    search_queries: list[str] = Field(
        description="2-4 concrete web search queries that would surface evidence."
    )


class ResearchPlan(BaseModel):
    objective: str = Field(
        description="One-sentence restatement of what the research must deliver."
    )
    subtasks: list[PlannedSubTask]
    success_criteria: list[str] = Field(
        description="Checkable criteria a complete answer must satisfy."
    )


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class SubTask(BaseModel):
    """A sub-task tracked in the shared task ledger."""

    id: str = Field(default_factory=lambda: _new_id("task"))
    question: str
    rationale: str = ""
    search_queries: list[str] = Field(default_factory=list)
    origin: Literal["planner", "critic"] = "planner"
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    revision: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


class Source(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""


class SourceRef(BaseModel):
    url: str = Field(description="URL of a page that supports the finding.")
    title: str = Field(description="Page title.")


class FindingDraft(BaseModel):
    """What the researcher submits via its ``submit_findings`` tool."""

    summary: str = Field(
        description="3-6 sentence answer to the sub-question, grounded in sources."
    )
    key_points: list[str] = Field(
        description="Atomic, verifiable facts with numbers/dates where possible."
    )
    sources: list[SourceRef] = Field(description="Pages actually read that support the key points.")
    confidence: float = Field(
        description="0.0-1.0 confidence that the summary is correct and complete."
    )
    gaps: list[str] = Field(description="What could not be verified or remains open.")

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return _clamp_unit(value)


class Finding(FindingDraft):
    id: str = Field(default_factory=lambda: _new_id("finding"))
    task_id: str
    question: str
    agent: str = "researcher"
    tool_calls: int = 0
    revision: int = 0
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------


class CritiqueIssue(BaseModel):
    severity: Literal["minor", "major", "critical"]
    description: str
    related_question: str = Field(description="The sub-question this issue concerns, or 'overall'.")


class Critique(BaseModel):
    score: float = Field(description="0.0-1.0 overall quality: coverage, evidence, consistency.")
    approved: bool = Field(description="True if the findings are good enough to write the report.")
    strengths: list[str]
    issues: list[CritiqueIssue]
    follow_up_tasks: list[PlannedSubTask] = Field(
        description="New sub-questions that would close the most important gaps. Empty if approved."
    )

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, value: float) -> float:
        return _clamp_unit(value)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    heading: str
    content: str = Field(description="Markdown body. Cite sources inline as [n].")


class ReportDraft(BaseModel):
    title: str
    executive_summary: str = Field(
        description="Short answer to the research question, with citations [n]."
    )
    sections: list[ReportSection]
    conclusion: str
    limitations: list[str]


class UsageStats(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls: int = 0
    by_agent: dict[str, dict[str, int]] = Field(default_factory=dict)


class ResearchReport(ReportDraft):
    run_id: str
    question: str
    backend: str = "langgraph"
    sources: list[Source] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    revisions: int = 0
    usage: UsageStats = Field(default_factory=UsageStats)
    memory_stats: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def final_score(self) -> float | None:
        return self.critiques[-1].score if self.critiques else None

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"> **Research question:** {self.question}", ""]
        lines += ["## Executive summary", "", self.executive_summary.strip(), ""]
        for section in self.sections:
            lines += [f"## {section.heading}", "", section.content.strip(), ""]
        lines += ["## Conclusion", "", self.conclusion.strip(), ""]
        if self.limitations:
            lines += ["## Limitations", ""] + [f"- {item}" for item in self.limitations] + [""]
        if self.sources:
            lines += ["## Sources", ""]
            lines += [
                f"{i}. [{s.title or s.url}]({s.url})" for i, s in enumerate(self.sources, start=1)
            ]
            lines.append("")
        score = f"{self.final_score:.2f}" if self.final_score is not None else "n/a"
        lines += [
            "---",
            f"*Run `{self.run_id}` · backend: {self.backend} · critic score: {score} · "
            f"revisions: {self.revisions} · findings: {len(self.findings)} · "
            f"LLM calls: {self.usage.llm_calls} · tokens in/out: "
            f"{self.usage.input_tokens}/{self.usage.output_tokens}*",
        ]
        return "\n".join(lines) + "\n"
