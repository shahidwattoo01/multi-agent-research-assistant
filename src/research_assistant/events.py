"""A tiny thread-safe event bus used for live progress output, tracing and SSE streaming."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("research_assistant")


class Event(BaseModel):
    type: str  # e.g. "plan", "task_started", "tool_call", "finding", "critique", "report", "error"
    agent: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: list[Event] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Subscriber) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def emit(self, type: str, agent: str, message: str, **data: Any) -> Event:
        event = Event(type=type, agent=agent, message=message, data=data)
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        logger.debug("[%s] %s: %s", agent, type, message)
        for fn in subscribers:
            try:
                fn(event)
            except Exception:  # a broken subscriber must never break the run
                logger.exception("event subscriber failed")
        return event

    @property
    def history(self) -> list[Event]:
        with self._lock:
            return list(self._history)
