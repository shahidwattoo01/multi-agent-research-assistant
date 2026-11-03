"""The researcher's tool surface. Every call goes through shared memory first.

The same :class:`ResearchToolbox` backs both orchestration backends: LangGraph calls it via
Claude tool use, AutoGen wraps its methods as ``FunctionTool``s.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from research_assistant.events import EventBus
from research_assistant.llm import ToolOutcome, ToolSpec
from research_assistant.memory import PageRecord, SharedMemory, normalize_text
from research_assistant.tools.fetch import FetchError, PageFetcher
from research_assistant.tools.search import SearchError, SearchProvider

Summarizer = Callable[[str, str], str]  # (page_text, focus) -> summary


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query. Be specific: names, years, technical terms.")


class FetchPageInput(BaseModel):
    url: str = Field(description="Absolute http(s) URL to read.")
    focus: str = Field(description="What information you need from this page.")


class RecallMemoryInput(BaseModel):
    query: str = Field(description="Topic to look up in teammates' published findings.")


TOOL_SPECS = [
    ToolSpec.from_model(
        "web_search",
        "Search the web. Returns titles, URLs and snippets. Results are cached in shared memory.",
        WebSearchInput,
    ),
    ToolSpec.from_model(
        "fetch_page",
        "Read a web page. Long pages are condensed to the parts relevant to `focus`.",
        FetchPageInput,
    ),
    ToolSpec.from_model(
        "recall_memory",
        "Look up findings already published by other researchers on the team.",
        RecallMemoryInput,
    ),
]


class ResearchToolbox:
    def __init__(
        self,
        memory: SharedMemory,
        search: SearchProvider,
        fetcher: PageFetcher,
        summarizer: Summarizer,
        *,
        events: EventBus | None = None,
        max_results: int = 5,
        summarize_threshold: int = 6000,
    ) -> None:
        self.memory = memory
        self.search_provider = search
        self.fetcher = fetcher
        self.summarizer = summarizer
        self.events = events or EventBus()
        self.max_results = max_results
        self.summarize_threshold = summarize_threshold

    @property
    def specs(self) -> list[ToolSpec]:
        return TOOL_SPECS

    def execute(
        self, name: str, args: dict[str, Any], *, task_id: str | None = None
    ) -> ToolOutcome:
        if name == "web_search":
            return self.web_search(args["query"])
        if name == "fetch_page":
            return self.fetch_page(args["url"], args.get("focus", ""))
        if name == "recall_memory":
            return self.recall_memory(args["query"], exclude_task=task_id)
        return ToolOutcome(f"Unknown tool: {name}", is_error=True)

    # ------------------------------------------------------------------ tools

    def web_search(self, query: str) -> ToolOutcome:
        cached = self.memory.get_search(query)
        if cached is not None:
            results, note = cached, "(from shared memory -- this query was already run)\n"
        else:
            try:
                results = self.search_provider.search(query, self.max_results)
            except SearchError as exc:
                return ToolOutcome(str(exc), is_error=True)
            self.memory.put_search(query, results)
            note = ""
        if not results:
            return ToolOutcome(f"No results for {query!r}. Try different wording.")
        lines = [f"{note}Results for {query!r}:"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r.title}\n   URL: {r.url}\n   {r.snippet}")
        return ToolOutcome("\n".join(lines))

    def fetch_page(self, url: str, focus: str) -> ToolOutcome:
        record = self.memory.get_page(url)
        if record is None:
            try:
                page = self.fetcher.fetch(url)
            except FetchError as exc:
                return ToolOutcome(str(exc), is_error=True)
            record = PageRecord(url=page.url, title=page.title, text=page.text)
            self.memory.put_page(record)
            if page.url != url:  # also index under the requested URL (pre-redirect)
                self.memory.put_page(record.model_copy(update={"url": url}))
            origin = ""
        else:
            origin = "(from shared memory) "

        header = f"{origin}Page: {record.title or '(untitled)'}\nURL: {record.url}\n\n"
        if len(record.text) <= self.summarize_threshold:
            return ToolOutcome(header + record.text)

        key = normalize_text(focus)
        summary = record.summaries.get(key)
        if summary is None:
            self.events.emit("summarize", "summarizer", f"Condensing {record.url}", url=record.url)
            summary = self.summarizer(record.text, focus or "key facts")
            self.memory.add_page_summary(url, focus, summary)
        return ToolOutcome(header + f"Summary focused on {focus!r}:\n{summary}")

    def recall_memory(self, query: str, exclude_task: str | None = None) -> ToolOutcome:
        hits = self.memory.recall(query, exclude_task=exclude_task)
        if not hits:
            return ToolOutcome("No related findings from teammates yet.")
        blocks = []
        for f in hits:
            sources = ", ".join(s.url for s in f.sources[:3])
            points = "\n".join(f"  - {p}" for p in f.key_points[:6])
            blocks.append(
                f"Sub-question: {f.question}\nSummary: {f.summary}\nKey points:\n{points}\n"
                f"Sources: {sources}"
            )
        return ToolOutcome("\n\n".join(blocks))
