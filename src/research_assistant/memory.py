"""Shared memory used by every agent in a run.

It has three parts:

* **Task ledger** -- every sub-task with its status. Agents read and update it to track
  progress, and new sub-tasks are checked against it so near-duplicates are never researched twice.
* **Knowledge cache** -- search results and fetched/summarized pages, keyed by normalized
  query/URL. Repeated lookups by any agent come from the cache. It is persisted to disk so later
  runs reuse it too.
* **Blackboard** -- findings published by researchers, which other agents can look up
  through ``recall`` (the researcher's ``recall_memory`` tool).

All public methods are thread-safe because researchers run in parallel.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from research_assistant.models import Finding, PlannedSubTask, Source, SubTask, TaskStatus

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
    "what", "which", "who", "how", "why", "when", "does", "do", "did", "with", "by", "at", "from",
    "its", "it", "this", "that", "as", "be", "vs", "versus", "about",
}  # fmt: skip


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if t not in _STOPWORDS}


def similarity(a: str, b: str) -> float:
    """Blend of token-set Jaccard and character-sequence ratio, in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    seq = SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
    return max(jaccard, 0.5 * jaccard + 0.5 * seq)


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


class PageRecord(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    summaries: dict[str, str] = Field(default_factory=dict)  # focus -> summary
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SharedMemory:
    def __init__(self, cache_path: Path | None = None, duplicate_threshold: float = 0.82) -> None:
        self._lock = threading.RLock()
        self.cache_path = cache_path
        self.duplicate_threshold = duplicate_threshold
        self.tasks: dict[str, SubTask] = {}
        self.findings: dict[str, Finding] = {}
        self._searches: dict[str, list[Source]] = {}
        self._pages: dict[str, PageRecord] = {}
        self.stats: Counter[str] = Counter()
        if cache_path is not None:
            self._load_cache()

    # ------------------------------------------------------------------ ledger

    def find_duplicate(self, question: str) -> SubTask | None:
        with self._lock:
            best: tuple[float, SubTask | None] = (0.0, None)
            for task in self.tasks.values():
                if task.status == TaskStatus.FAILED:
                    continue
                score = similarity(question, task.question)
                if score > best[0]:
                    best = (score, task)
            return best[1] if best[0] >= self.duplicate_threshold else None

    def add_tasks(
        self, planned: list[PlannedSubTask], *, origin: str, revision: int = 0
    ) -> list[SubTask]:
        """Register sub-tasks, skipping near-duplicates of existing ones. Returns the new tasks."""
        created: list[SubTask] = []
        with self._lock:
            for item in planned:
                duplicate = self.find_duplicate(item.question)
                if duplicate is not None:
                    self.stats["duplicate_tasks_skipped"] += 1
                    logger.info("skip duplicate task %r (matches %s)", item.question, duplicate.id)
                    continue
                task = SubTask(
                    question=item.question,
                    rationale=item.rationale,
                    search_queries=item.search_queries,
                    origin=origin,  # type: ignore[arg-type]
                    revision=revision,
                )
                self.tasks[task.id] = task
                created.append(task)
            self.stats["tasks_created"] += len(created)
        return created

    def set_status(self, task_id: str, status: TaskStatus, error: str | None = None) -> SubTask:
        with self._lock:
            task = self.tasks[task_id]
            update: dict[str, Any] = {"status": status, "updated_at": datetime.now(timezone.utc)}
            if status == TaskStatus.IN_PROGRESS:
                update["attempts"] = task.attempts + 1
            if error is not None:
                update["error"] = error
            self.tasks[task_id] = task = task.model_copy(update=update)
            return task

    def pending_tasks(self) -> list[SubTask]:
        with self._lock:
            return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    def progress(self) -> dict[str, int]:
        with self._lock:
            counts = Counter(t.status.value for t in self.tasks.values())
            return {s.value: counts.get(s.value, 0) for s in TaskStatus}

    # -------------------------------------------------------------- blackboard

    def publish_finding(self, finding: Finding) -> None:
        with self._lock:
            self.findings[finding.id] = finding

    def all_findings(self) -> list[Finding]:
        with self._lock:
            return sorted(self.findings.values(), key=lambda f: f.created_at)

    def recall(self, query: str, limit: int = 3, exclude_task: str | None = None) -> list[Finding]:
        with self._lock:
            scored = [
                (similarity(query, f"{f.question} {f.summary} {' '.join(f.key_points)}"), f)
                for f in self.findings.values()
                if f.task_id != exclude_task
            ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [f for score, f in scored[:limit] if score > 0.05]

    # --------------------------------------------------------- knowledge cache

    def get_search(self, query: str) -> list[Source] | None:
        with self._lock:
            hit = self._searches.get(normalize_text(query))
            self.stats["search_cache_hits" if hit is not None else "search_cache_misses"] += 1
            return hit

    def put_search(self, query: str, results: list[Source]) -> None:
        with self._lock:
            self._searches[normalize_text(query)] = results

    def get_page(self, url: str) -> PageRecord | None:
        with self._lock:
            hit = self._pages.get(normalize_url(url))
            self.stats["page_cache_hits" if hit is not None else "page_cache_misses"] += 1
            return hit

    def put_page(self, record: PageRecord) -> None:
        with self._lock:
            self._pages[normalize_url(record.url)] = record

    def add_page_summary(self, url: str, focus: str, summary: str) -> None:
        with self._lock:
            record = self._pages.get(normalize_url(url))
            if record is not None:
                record.summaries[normalize_text(focus)] = summary

    def known_sources(self) -> dict[str, str]:
        """Map normalized URL -> title for every page or search hit seen in this memory."""
        with self._lock:
            known = {key: rec.title for key, rec in self._pages.items()}
            for results in self._searches.values():
                for src in results:
                    known.setdefault(normalize_url(src.url), src.title)
            return known

    def cache_size(self) -> dict[str, int]:
        with self._lock:
            return {"searches": len(self._searches), "pages": len(self._pages)}

    # ------------------------------------------------------------- persistence

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tasks": [t.model_dump(mode="json") for t in self.tasks.values()],
                "findings": [f.model_dump(mode="json") for f in self.findings.values()],
                "progress": self.progress(),
                "stats": dict(self.stats),
            }

    def save_run(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def save_cache(self) -> None:
        if self.cache_path is None:
            return
        with self._lock:
            payload = {
                "searches": {
                    q: [s.model_dump(mode="json") for s in results]
                    for q, results in self._searches.items()
                },
                "pages": {u: p.model_dump(mode="json") for u, p in self._pages.items()},
            }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.cache_path)

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._searches = {
                q: [Source.model_validate(s) for s in results]
                for q, results in payload.get("searches", {}).items()
            }
            self._pages = {
                u: PageRecord.model_validate(p) for u, p in payload.get("pages", {}).items()
            }
            logger.info(
                "loaded memory cache: %d searches, %d pages", len(self._searches), len(self._pages)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("ignoring unreadable memory cache %s: %s", self.cache_path, exc)

    def clear_cache(self) -> None:
        with self._lock:
            self._searches.clear()
            self._pages.clear()
        if self.cache_path is not None and self.cache_path.exists():
            self.cache_path.unlink()
