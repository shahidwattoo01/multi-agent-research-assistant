"""Run one research task programmatically and print live agent events.

python examples/quickstart.py "How do solid-state batteries compare with lithium-ion today?"
"""

import sys

from research_assistant.config import get_settings
from research_assistant.factory import build_orchestrator


def main() -> None:
    question = (
        " ".join(sys.argv[1:]) or "What are the main techniques for reducing LLM hallucinations?"
    )
    settings = get_settings().model_copy(update={"max_subtasks": 4, "max_revisions": 1})
    orchestrator = build_orchestrator("langgraph", settings)

    report = orchestrator.run(
        question, on_event=lambda e: print(f"[{e.agent:>10}] {e.type:<12} {e.message}")
    )

    print("\n" + report.to_markdown())
    print("Per-agent usage:", report.usage.by_agent)
    print("Memory stats:", report.memory_stats)


if __name__ == "__main__":
    main()
