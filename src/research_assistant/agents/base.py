from __future__ import annotations

from research_assistant.config import Settings
from research_assistant.events import EventBus
from research_assistant.llm import LLMClient
from research_assistant.memory import normalize_url
from research_assistant.models import Finding, ResearchPlan, Source


class Agent:
    """Common wiring for all agents: an LLM client, settings and the event bus."""

    name: str = "agent"

    def __init__(self, llm: LLMClient, settings: Settings, events: EventBus | None = None) -> None:
        self.llm = llm
        self.settings = settings
        self.events = events or EventBus()

    def emit(self, type: str, message: str, **data) -> None:
        self.events.emit(type, self.name, message, **data)


def render_plan(plan: ResearchPlan) -> str:
    criteria = "\n".join(f"- {c}" for c in plan.success_criteria)
    return f"<objective>{plan.objective}</objective>\n<success_criteria>\n{criteria}\n</success_criteria>"


def render_findings(findings: list[Finding], source_index: dict[str, int] | None = None) -> str:
    blocks = []
    for f in findings:
        points = "\n".join(f"  - {p}" for p in f.key_points)
        if source_index is not None:
            refs = ", ".join(
                f"[{source_index[normalize_url(s.url)]}] {s.title}"
                for s in f.sources
                if normalize_url(s.url) in source_index
            )
        else:
            refs = ", ".join(s.url for s in f.sources)
        gaps = "; ".join(f.gaps) or "none"
        blocks.append(
            f'<finding question="{f.question}" confidence="{f.confidence:.2f}">\n'
            f"<summary>{f.summary}</summary>\n<key_points>\n{points}\n</key_points>\n"
            f"<sources>{refs or 'none'}</sources>\n<gaps>{gaps}</gaps>\n</finding>"
        )
    return "\n\n".join(blocks) if blocks else "(no findings)"


def collect_sources(findings: list[Finding]) -> list[Source]:
    """Deduplicated, ordered source list across all findings."""
    seen: dict[str, Source] = {}
    for f in findings:
        for ref in f.sources:
            key = normalize_url(ref.url)
            if key not in seen:
                seen[key] = Source(url=ref.url, title=ref.title)
    return list(seen.values())
