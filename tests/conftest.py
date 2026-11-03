import pytest

from research_assistant.config import Settings
from tests.fakes import FakeFetcher, FakeLLM, FakeSearch


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=tmp_path / "memory",
        persist_memory=True,
        max_revisions=2,
        summarize_threshold_chars=1000,
        _env_file=None,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_search() -> FakeSearch:
    return FakeSearch()


@pytest.fixture
def fake_fetcher() -> FakeFetcher:
    return FakeFetcher()
