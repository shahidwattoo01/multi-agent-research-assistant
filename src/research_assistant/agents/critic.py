from __future__ import annotations

from research_assistant.agents.base import Agent, render_findings, render_plan
from research_assistant.models import Critique, Finding, ResearchPlan, SubTask
from research_assistant.prompts import CRITIC_SYSTEM

MAX_FOLLOW_UPS = 3


class CriticAgent(Agent):
    """Reviews findings against the plan and drives the self-correction loop."""

    name = "critic"

    def review(
        self,
        question: str,
        plan: ResearchPlan,
        findings: list[Finding],
        investigated: list[SubTask],
        revision: int,
    ) -> Critique:
        self.emit("agent_started", f"Reviewing {len(findings)} findings (round {revision + 1})")
        already = "\n".join(f"- {t.question} [{t.status.value}]" for t in investigated)
        final_round = revision >= self.settings.max_revisions
        prompt = (
            f"<research_question>{question}</research_question>\n{render_plan(plan)}\n"
            f"<already_investigated>\n{already}\n</already_investigated>\n"
            f"<findings>\n{render_findings(findings)}\n</findings>\n"
            f"<review_round>{revision + 1} of {self.settings.max_revisions + 1}</review_round>\n\n"
            + (
                "This is the final round: no more research will run. Score honestly; "
                "leave follow_up_tasks empty."
                if final_round
                else "Review the findings."
            )
        )
        critique = self.llm.structured(
            agent=self.name,
            system=CRITIC_SYSTEM.format(
                threshold=self.settings.approval_threshold, max_follow_ups=MAX_FOLLOW_UPS
            ),
            prompt=prompt,
            schema=Critique,
            model=self.settings.model,
            effort=self.settings.critic_effort,
        )
        critique = apply_review_policy(critique, self.settings.approval_threshold, final_round)
        verdict = "approved" if critique.approved else f"{len(critique.follow_up_tasks)} follow-ups"
        self.emit(
            "critique",
            f"score {critique.score:.2f} - {verdict}",
            score=critique.score,
            approved=critique.approved,
            issues=[i.description for i in critique.issues],
            follow_ups=[t.question for t in critique.follow_up_tasks],
        )
        return critique


def apply_review_policy(critique: Critique, threshold: float, final_round: bool) -> Critique:
    """Apply the approval rule in code so it does not depend on model self-consistency.

    Approve iff score >= threshold and no critical issue. Follow-ups are only kept when the
    critique is not approved and another round is allowed; a rejection with nothing actionable
    to research is treated as approval.
    """
    has_critical = any(i.severity == "critical" for i in critique.issues)
    approved = critique.score >= threshold and not has_critical
    follow_ups = [] if (approved or final_round) else critique.follow_up_tasks[:MAX_FOLLOW_UPS]
    if not approved and not follow_ups and not final_round:
        approved = True
    return critique.model_copy(update={"approved": approved, "follow_up_tasks": follow_ups})
