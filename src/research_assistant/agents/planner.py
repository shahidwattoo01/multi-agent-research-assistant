from __future__ import annotations

from research_assistant.agents.base import Agent
from research_assistant.models import ResearchPlan
from research_assistant.prompts import PLANNER_SYSTEM


class PlannerAgent(Agent):
    """Decomposes the research question into independent, searchable sub-questions."""

    name = "planner"

    def plan(self, question: str) -> ResearchPlan:
        self.emit("agent_started", "Decomposing the research question")
        plan = self.llm.structured(
            agent=self.name,
            system=PLANNER_SYSTEM.format(max_subtasks=self.settings.max_subtasks),
            prompt=f"<research_question>\n{question}\n</research_question>\n\nCreate the research plan.",
            schema=ResearchPlan,
            model=self.settings.model,
            effort=self.settings.planner_effort,
        )
        plan.subtasks = plan.subtasks[: self.settings.max_subtasks]
        self.emit(
            "plan",
            f"Planned {len(plan.subtasks)} sub-tasks",
            objective=plan.objective,
            subtasks=[s.question for s in plan.subtasks],
        )
        return plan
