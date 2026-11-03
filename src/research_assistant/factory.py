"""Single place that turns (backend name, settings overrides) into a ready orchestrator."""

from __future__ import annotations

from typing import Any, Literal

from research_assistant.config import Settings, get_settings
from research_assistant.graph import ResearchOrchestrator

Backend = Literal["langgraph", "autogen"]
BACKENDS: tuple[str, ...] = ("langgraph", "autogen")


def build_orchestrator(
    backend: Backend = "langgraph",
    settings: Settings | None = None,
    **overrides: Any,
) -> ResearchOrchestrator:
    settings = settings or get_settings()
    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean:
        settings = settings.model_copy(update=clean)
    if backend == "langgraph":
        return ResearchOrchestrator.from_settings(settings)
    if backend == "autogen":
        try:
            from research_assistant.autogen_backend import AutoGenOrchestrator
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                'The AutoGen backend needs extra packages: pip install "research-assistant[autogen]"'
            ) from exc
        return AutoGenOrchestrator.from_settings(settings)
    raise ValueError(f"Unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
