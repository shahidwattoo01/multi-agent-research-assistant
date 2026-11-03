"""Runtime configuration, loaded from environment variables (prefix ``RA_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Models -------------------------------------------------------------
    model: str = Field("claude-opus-5-5", description="Model used by planner, critic and writer.")
    worker_model: str = Field(
        "claude-opus-5-5", description="Model used by the researcher and page summarizer."
    )
    planner_effort: Effort = "high"
    researcher_effort: Effort = "medium"
    critic_effort: Effort = "high"
    writer_effort: Effort = "high"
    summarizer_effort: Effort = "low"
    max_tokens: int = 16000
    enable_fallbacks: bool = Field(
        True, description="Opt into server-side refusal fallbacks (fallbacks='default')."
    )

    # --- Workflow limits ----------------------------------------------------
    max_subtasks: int = Field(6, ge=1, le=20)
    max_revisions: int = Field(2, ge=0, le=10, description="Critic-driven self-correction rounds.")
    max_tool_calls_per_task: int = Field(8, ge=1, le=50)
    approval_threshold: float = Field(0.75, ge=0.0, le=1.0)
    parallel_research: bool = True

    # --- Tools --------------------------------------------------------------
    search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: str | None = Field(None, validation_alias="TAVILY_API_KEY")
    search_results: int = Field(5, ge=1, le=20)
    fetch_timeout: float = 20.0
    summarize_threshold_chars: int = Field(
        6000, description="Pages longer than this are condensed by the summarizer before use."
    )
    max_page_chars: int = 60000

    # --- Memory -------------------------------------------------------------
    memory_dir: Path = Path(".research_memory")
    persist_memory: bool = True
    duplicate_similarity: float = Field(
        0.82,
        ge=0.0,
        le=1.0,
        description="Similarity above which two sub-tasks count as duplicates.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
