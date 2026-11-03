from __future__ import annotations

from research_assistant.agents.base import Agent
from research_assistant.prompts import SUMMARIZER_SYSTEM


class SummarizerAgent(Agent):
    """Condenses long pages into focused notes so researchers' context stays small."""

    name = "summarizer"

    def summarize(self, text: str, focus: str) -> str:
        return self.llm.complete(
            agent=self.name,
            system=SUMMARIZER_SYSTEM,
            prompt=f"<focus>{focus}</focus>\n\n<page>\n{text}\n</page>",
            model=self.settings.worker_model,
            effort=self.settings.summarizer_effort,
        )
