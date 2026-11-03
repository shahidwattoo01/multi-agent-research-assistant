"""Web search providers. DuckDuckGo needs no API key; Tavily is used when a key is configured."""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from research_assistant.models import Source

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    pass


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[Source]: ...


class DuckDuckGoSearch:
    name = "duckduckgo"

    def search(self, query: str, max_results: int) -> list[Source]:
        from ddgs import DDGS

        try:
            rows = DDGS().text(query, max_results=max_results) or []
        except Exception as exc:  # ddgs raises a variety of transport/ratelimit errors
            raise SearchError(f"DuckDuckGo search failed: {exc}") from exc
        return [
            Source(url=r["href"], title=r.get("title", ""), snippet=r.get("body", ""))
            for r in rows
            if r.get("href")
        ]


class TavilySearch:
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> list[Source]:
        try:
            response = httpx.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "max_results": max_results, "search_depth": "basic"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError(f"Tavily search failed: {exc}") from exc
        return [
            Source(url=r["url"], title=r.get("title", ""), snippet=r.get("content", ""))
            for r in response.json().get("results", [])
            if r.get("url")
        ]


def build_search_provider(provider: str, tavily_api_key: str | None = None) -> SearchProvider:
    if provider == "tavily":
        if not tavily_api_key:
            raise ValueError("search_provider=tavily requires TAVILY_API_KEY")
        return TavilySearch(tavily_api_key)
    return DuckDuckGoSearch()
