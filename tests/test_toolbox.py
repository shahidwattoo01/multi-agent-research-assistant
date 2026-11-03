from research_assistant.memory import SharedMemory
from research_assistant.tools.fetch import html_to_text
from research_assistant.tools.toolbox import TOOL_SPECS, ResearchToolbox
from tests.fakes import FakeFetcher, FakeSearch


def _toolbox(threshold=1000):
    summaries = []

    def summarizer(text, focus):
        summaries.append(focus)
        return f"summary for {focus}"

    tb = ResearchToolbox(
        SharedMemory(), FakeSearch(), FakeFetcher(), summarizer, summarize_threshold=threshold
    )
    return tb, summaries


def test_tool_specs_are_strict():
    for spec in TOOL_SPECS:
        api = spec.to_api()
        assert api["strict"] is True
        assert api["input_schema"]["additionalProperties"] is False


def test_search_is_cached_in_shared_memory():
    tb, _ = _toolbox()
    first = tb.execute("web_search", {"query": "solar costs"})
    second = tb.execute("web_search", {"query": "Solar costs"})
    assert "example.org/solar" in first.content
    assert "from shared memory" in second.content
    assert tb.search_provider.calls == ["solar costs"]


def test_short_pages_returned_verbatim_long_pages_summarized_once_per_focus():
    tb, summaries = _toolbox()
    short = tb.execute("fetch_page", {"url": "https://example.org/solar", "focus": "costs"})
    assert "LCOE fell 90%" in short.content

    long1 = tb.execute("fetch_page", {"url": "https://example.org/storage", "focus": "prices"})
    long2 = tb.execute("fetch_page", {"url": "https://example.org/storage", "focus": "Prices"})
    assert "summary for prices" in long1.content
    assert "from shared memory" in long2.content
    assert summaries == ["prices"]
    assert tb.fetcher.calls == ["https://example.org/solar", "https://example.org/storage"]


def test_fetch_errors_are_reported_not_raised():
    tb, _ = _toolbox()
    out = tb.execute("fetch_page", {"url": "https://missing.example", "focus": "x"})
    assert out.is_error and "404" in out.content


def test_unknown_tool():
    tb, _ = _toolbox()
    assert tb.execute("rm_rf", {}).is_error


def test_html_to_text_strips_boilerplate():
    html = (
        "<html><head><title>T</title><script>x()</script></head><body><nav>menu</nav>"
        "<article><h1>Head</h1><p>Body text</p></article><footer>f</footer></body></html>"
    )
    title, text = html_to_text(html)
    assert title == "T"
    assert "Body text" in text and "menu" not in text and "x()" not in text
