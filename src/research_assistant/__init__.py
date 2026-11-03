"""Multi-agent research assistant.

Planner, researcher and critic agents collaborate over a LangGraph workflow
(or, optionally, an AutoGen group chat) to complete multi-step research tasks.
"""

from research_assistant.config import Settings, get_settings
from research_assistant.models import ResearchReport

__all__ = ["ResearchReport", "Settings", "get_settings", "run_research"]
__version__ = "1.0.0"


def run_research(question: str, **kwargs):
    """Convenience wrapper: run the LangGraph pipeline end to end and return a ResearchReport."""
    from research_assistant.graph import ResearchOrchestrator

    return ResearchOrchestrator.from_settings(**kwargs).run(question)
