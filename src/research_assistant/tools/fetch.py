"""Fetch a web page and extract readable text."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from bs4 import BeautifulSoup

_USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchAssistant/1.0; +https://github.com/) "
    "AppleWebKit/537.36 (KHTML, like Gecko)"
)
_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]


class FetchError(RuntimeError):
    pass


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def pdf_to_text(data: bytes, max_pages: int = 40) -> tuple[str, str]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    except PdfReadError as exc:
        raise FetchError(f"Unreadable PDF: {exc}") from exc
    title = str((reader.metadata or {}).get("/Title", "") or "")
    return title, re.sub(r"[ \t]+", " ", "\n".join(pages)).strip()


class HttpPageFetcher:
    def __init__(self, timeout: float = 20.0, max_chars: int = 60000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    def fetch(self, url: str) -> FetchedPage:
        if not url.startswith(("http://", "https://")):
            raise FetchError(f"Unsupported URL scheme: {url}")
        try:
            response = httpx.get(
                url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "text/html,application/pdf,text/plain;q=0.9",
                },
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"Could not fetch {url}: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type or str(response.url).lower().endswith(".pdf"):
            title, text = pdf_to_text(response.content)
        elif "html" in content_type:
            title, text = html_to_text(response.text)
        elif content_type.startswith("text/") or "json" in content_type:
            title, text = "", response.text
        else:
            raise FetchError(f"Unsupported content type {content_type!r} at {url}")

        if not text.strip():
            raise FetchError(f"No readable text at {url}")
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[... page truncated ...]"
        return FetchedPage(url=str(response.url), title=title, text=text)
