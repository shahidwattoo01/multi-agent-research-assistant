"""Command-line interface: ``research-assistant run "your question"``."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from research_assistant import __version__
from research_assistant.config import get_settings
from research_assistant.events import Event
from research_assistant.factory import BACKENDS, build_orchestrator
from research_assistant.llm import LLMError

app = typer.Typer(
    help="Multi-agent research assistant (planner, researchers, critic, writer).",
    no_args_is_help=True,
    add_completion=False,
)
memory_app = typer.Typer(help="Inspect or clear the persistent shared-memory cache.")
app.add_typer(memory_app, name="memory")

console = Console()

_AGENT_STYLES = {
    "planner": "bold cyan",
    "researcher": "green",
    "summarizer": "dim green",
    "critic": "bold magenta",
    "writer": "bold blue",
    "orchestrator": "bold white",
}
_QUIET_TYPES = {"message", "agent_started"}


def _print_event(event: Event, verbose: bool) -> None:
    if event.type in _QUIET_TYPES and not verbose:
        return
    style = _AGENT_STYLES.get(event.agent, "white")
    console.print(f"[{style}]{event.agent:>12}[/] [dim]{event.type:<13}[/] {event.message}")
    if event.type == "plan":
        for i, q in enumerate(event.data.get("subtasks", []), 1):
            console.print(f"{'':>27}{i}. {q}")
    if event.type == "critique" and verbose:
        for issue in event.data.get("issues", []):
            console.print(f"{'':>27}[yellow]- {issue}[/]")


def _slug(text: str, limit: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "report"


@app.command()
def run(
    question: str = typer.Argument(..., help="The research question."),
    backend: str = typer.Option(
        "langgraph", "--backend", "-b", help=f"One of: {', '.join(BACKENDS)}"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Markdown report path."),
    json_out: Path | None = typer.Option(
        None, "--json", help="Also write the full report as JSON."
    ),
    max_subtasks: int | None = typer.Option(None, help="Max sub-tasks the planner may create."),
    max_revisions: int | None = typer.Option(None, help="Max critic-driven revision rounds."),
    model: str | None = typer.Option(None, help="Model for planner/critic/writer."),
    worker_model: str | None = typer.Option(None, help="Model for researcher/summarizer."),
    sequential: bool = typer.Option(
        False, "--sequential", help="Research sub-tasks one at a time."
    ),
    no_memory: bool = typer.Option(False, "--no-memory", help="Do not read/write the disk cache."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every agent message."),
    show: bool = typer.Option(True, "--show/--no-show", help="Render the report in the terminal."),
) -> None:
    """Research QUESTION and write a cited Markdown report."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    try:
        orchestrator = build_orchestrator(
            backend,  # type: ignore[arg-type]
            max_subtasks=max_subtasks,
            max_revisions=max_revisions,
            model=model,
            worker_model=worker_model,
            parallel_research=False if sequential else None,
            persist_memory=False if no_memory else None,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc

    console.print(Panel.fit(question, title=f"Research · {backend}", border_style="cyan"))
    try:
        report = orchestrator.run(question, on_event=lambda e: _print_event(e, verbose))
    except LLMError as exc:
        console.print(f"[red]LLM error:[/] {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/]")
        raise typer.Exit(130) from None

    output = output or Path("reports") / f"{_slug(report.title)}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_markdown(), encoding="utf-8")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if show:
        console.rule()
        console.print(Markdown(report.to_markdown()))
        console.rule()
    _print_summary(report)
    console.print(f"[green]Report saved to[/] {output}")


def _print_summary(report) -> None:
    table = Table(title="Run summary", show_header=False, box=None)
    score = f"{report.final_score:.2f}" if report.final_score is not None else "n/a"
    rows = [
        ("Critic score", score),
        ("Revision rounds", str(report.revisions)),
        ("Findings / sources", f"{len(report.findings)} / {len(report.sources)}"),
        ("LLM calls / tool calls", f"{report.usage.llm_calls} / {report.usage.tool_calls}"),
        ("Tokens in / out", f"{report.usage.input_tokens:,} / {report.usage.output_tokens:,}"),
        (
            "Memory hits (search/page)",
            f"{report.memory_stats.get('search_cache_hits', 0)} / "
            f"{report.memory_stats.get('page_cache_hits', 0)}",
        ),
        ("Duplicate tasks skipped", str(report.memory_stats.get("duplicate_tasks_skipped", 0))),
    ]
    for k, v in rows:
        table.add_row(f"[dim]{k}[/]", v)
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
) -> None:
    """Start the REST API (requires the `api` extra)."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print('[red]Install the API extra: pip install "research-assistant[api]"[/]')
        raise typer.Exit(2) from exc
    uvicorn.run("research_assistant.api:app", host=host, port=port)


@memory_app.command("stats")
def memory_stats() -> None:
    """Show what the persistent cache holds."""
    from research_assistant.memory import SharedMemory

    settings = get_settings()
    path = settings.memory_dir / "cache.json"
    mem = SharedMemory(cache_path=path)
    runs = list((settings.memory_dir / "runs").glob("*.json"))
    size = mem.cache_size()
    console.print(
        f"Cache: {path} - {size['searches']} searches, {size['pages']} pages; "
        f"{len(runs)} saved runs"
    )


@memory_app.command("clear")
def memory_clear(yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation.")) -> None:
    """Delete the persistent search/page cache."""
    from research_assistant.memory import SharedMemory

    path = get_settings().memory_dir / "cache.json"
    if not yes and not typer.confirm(f"Delete {path}?"):
        raise typer.Abort()
    SharedMemory(cache_path=path).clear_cache()
    console.print("Cache cleared.")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
