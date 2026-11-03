"""REST API: submit research jobs, poll status, stream agent events over Server-Sent Events.

POST /research                 {"question": "...", "backend": "langgraph"} -> 202 {run_id}
GET  /research/{run_id}        status + report JSON when finished
GET  /research/{run_id}/report.md
GET  /research/{run_id}/events text/event-stream of agent events
GET  /health
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from research_assistant import __version__
from research_assistant.events import Event
from research_assistant.factory import build_orchestrator
from research_assistant.graph import ResearchOrchestrator
from research_assistant.models import ResearchReport

RunStatus = Literal["queued", "running", "completed", "failed"]
OrchestratorFactory = Callable[..., ResearchOrchestrator]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    backend: Literal["langgraph", "autogen"] = "langgraph"
    max_subtasks: int | None = Field(None, ge=1, le=20)
    max_revisions: int | None = Field(None, ge=0, le=10)


class RunInfo(BaseModel):
    run_id: str
    status: RunStatus
    question: str
    backend: str
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    progress: list[Event] = Field(default_factory=list)
    report: ResearchReport | None = None


class RunManager:
    def __init__(self, factory: OrchestratorFactory, max_workers: int = 2) -> None:
        self.factory = factory
        self.runs: dict[str, RunInfo] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")

    def submit(self, req: ResearchRequest) -> RunInfo:
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        info = RunInfo(
            run_id=run_id,
            status="queued",
            question=req.question,
            backend=req.backend,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self.runs[run_id] = info
        self._pool.submit(self._execute, run_id, req)
        return info

    def _execute(self, run_id: str, req: ResearchRequest) -> None:
        info = self.runs[run_id]
        info.status = "running"
        try:
            orchestrator = self.factory(
                req.backend, max_subtasks=req.max_subtasks, max_revisions=req.max_revisions
            )
            info.report = orchestrator.run(
                req.question, on_event=info.progress.append, run_id=run_id
            )
            info.status = "completed"
        except Exception as exc:  # surface any failure to the client
            info.error = f"{type(exc).__name__}: {exc}"
            info.status = "failed"
        finally:
            info.finished_at = datetime.now(timezone.utc)

    def get(self, run_id: str) -> RunInfo:
        info = self.runs.get(run_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Unknown run {run_id}")
        return info


def create_app(factory: OrchestratorFactory = build_orchestrator) -> FastAPI:
    api = FastAPI(
        title="Multi-Agent Research Assistant",
        version=__version__,
        description="Planner, researcher and critic agents that research a question and write a cited report.",
    )
    manager = RunManager(factory)
    api.state.manager = manager

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @api.post("/research", status_code=202)
    def submit(req: ResearchRequest) -> dict[str, str]:
        info = manager.submit(req)
        return {"run_id": info.run_id, "status": info.status}

    @api.get("/research")
    def list_runs() -> list[dict[str, str]]:
        return [
            {"run_id": r.run_id, "status": r.status, "question": r.question}
            for r in manager.runs.values()
        ]

    @api.get("/research/{run_id}")
    def status(run_id: str, include_events: bool = False) -> RunInfo:
        info = manager.get(run_id)
        return info if include_events else info.model_copy(update={"progress": []})

    @api.get("/research/{run_id}/report.md", response_class=PlainTextResponse)
    def report_markdown(run_id: str) -> str:
        info = manager.get(run_id)
        if info.report is None:
            raise HTTPException(status_code=409, detail=f"Run is {info.status}")
        return info.report.to_markdown()

    @api.get("/research/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        info = manager.get(run_id)

        async def stream():
            sent = 0
            while True:
                pending = info.progress[sent:]
                for event in pending:
                    yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
                sent += len(pending)
                if info.status in ("completed", "failed") and sent >= len(info.progress):
                    yield f'event: end\ndata: {{"status": "{info.status}"}}\n\n'
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return api


app = create_app()
