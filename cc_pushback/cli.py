"""The ``cc-pushback`` command-line interface: scan, stats, list, and view-samples."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import get_args

import anyio
import click
from cc_transcript import CLAUDE_PROJECTS_DIR

from cc_pushback.models import SourceKind
from cc_pushback.report import Sample, build_summary, render_html
from cc_pushback.scan import scan as run_scan
from cc_pushback.serve import serve
from cc_pushback.store import FeedbackStore

SOURCE_KINDS = get_args(SourceKind)


def coro[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, R]:
    """Adapts an async command body into the sync callback Click expects."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return anyio.run(functools.partial(fn, *args, **kwargs))

    return wrapper


@click.group()
@click.version_option(package_name="cc-pushback")
def main() -> None:
    """Collect developer pushback signals from existing Claude Code transcripts."""


@main.command()
@click.option(
    "--transcripts",
    "transcripts",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Transcript directories to scan. Defaults to ~/.claude/projects.",
)
@click.option("--full", is_flag=True, help="Re-scan every transcript, ignoring recorded mtimes.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def scan(transcripts: tuple[Path, ...], full: bool, db: Path | None) -> None:
    """Scan transcripts for feedback, incrementally.

    Each transcript is parsed only when new or modified since the last scan, and
    every candidate is inserted with ``INSERT OR IGNORE`` keyed by a content
    digest, so re-running ``scan`` over unchanged inputs is a no-op. Recording a
    file and inserting its candidates commit in one transaction.
    """
    roots = transcripts or (CLAUDE_PROJECTS_DIR,)
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_scan(store, roots, full=full)
    click.echo(f"scanned {report.scanned} files, {report.inserted} new rows")


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def stats(db: Path | None) -> None:
    """Print ingestion counts by source kind and the scanned-file count."""
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await store.stats()
    click.echo(f"total: {report.total}  files: {report.files}")
    for kind, count in report.by_source.items():
        click.echo(f"  {kind}: {count}")


@main.command(name="list")
@click.option(
    "--source",
    "source",
    type=click.Choice(SOURCE_KINDS),
    default=None,
    help="Restrict to one source kind.",
)
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum events to show.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def list_(source: SourceKind | None, limit: int, db: Path | None) -> None:
    """List recent feedback events, newest first."""
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        rows = await store.recent(source_kind=source, limit=limit)
    for row in rows:
        click.echo(f"[{row['source_kind']}] {row['occurred_at']}  {str(row['text'])[:200]}")


@main.command(name="view-samples")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@click.option(
    "--llm/--no-llm",
    default=True,
    show_default=True,
    help="Summarize with the claude CLI when it is on PATH, else use heuristics.",
)
@click.option("--model", default="claude-sonnet-4-6", show_default=True, help="Model for the claude CLI summary.")
@click.option("--port", type=int, default=0, show_default=True, help="Port to serve on; 0 picks a free one.")
@click.option("--open", "open_", is_flag=True, help="Open the page in a browser once serving.")
@coro
async def view_samples(db: Path | None, llm: bool, model: str, port: int, open_: bool) -> None:
    """Render every collected sample into one HTML page and serve it locally.

    The page leads with a corpus summary and highlights, then lists every sample
    with a kind filter, a free-text search, and an expandable context window. It is
    built in memory and served over a transient HTTP server whose URL is printed;
    press Ctrl-C to stop. The summary is written by the ``claude`` CLI when ``--llm``
    is set and ``claude`` is installed, falling back to deterministic heuristics.
    """
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        samples = [Sample.from_row(row) for row in await store.events()]
    summary = await build_summary(samples, use_llm=llm, model=model)
    await serve(render_html(samples, summary).encode("utf-8"), port=port, open_browser=open_)
