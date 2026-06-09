"""The ``cc-pushback`` command-line interface: scan, stats, and list."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import click
from cc_transcript import CLAUDE_PROJECTS_DIR

from cc_pushback.models import SourceKind
from cc_pushback.scan import scan as run_scan
from cc_pushback.store import FeedbackStore

SOURCE_KINDS = get_args(SourceKind)


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
def scan(transcripts: tuple[Path, ...], full: bool, db: Path | None) -> None:
    """Scan transcripts for feedback, incrementally.

    Each transcript is parsed only when new or modified since the last scan, and
    every candidate is inserted with ``INSERT OR IGNORE`` keyed by a content
    digest, so re-running ``scan`` over unchanged inputs is a no-op. Recording a
    file and inserting its candidates commit in one transaction.
    """
    roots = transcripts or (CLAUDE_PROJECTS_DIR,)
    with FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = run_scan(store, roots, full=full)
    click.echo(f"scanned {report.scanned} files, {report.inserted} new rows")
    if report.skipped:
        click.echo(f"skipped {len(report.skipped)} unparseable files")


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
def stats(db: Path | None) -> None:
    """Print ingestion counts by source kind and the scanned-file count."""
    with FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = store.stats()
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
def list_(source: SourceKind | None, limit: int, db: Path | None) -> None:
    """List recent feedback events, newest first."""
    with FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        rows = store.recent(source_kind=source, limit=limit)
    for row in rows:
        click.echo(f"[{row['source_kind']}] {row['occurred_at']}  {str(row['text'])[:200]}")
