from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import click

from cc_pushback.classify import (
    PROMPT_VERSION,
    cheap_pass,
    llm_pass,
    unclassified_events,
)
from cc_pushback.llm import ClaudeBackend, CodexBackend, LlmBackend, TModel
from cc_pushback.models import SourceKind
from cc_pushback.patterns import TAXONOMY_VERSION
from cc_pushback.repo import Repository
from cc_pushback.sources import (
    GitHubReviews,
    Interrupts,
    PlanReviews,
    SupersetIssues,
    TranscriptMessages,
    changed_files,
    changed_issue_files,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from cc_transcript.models import TranscriptEvent

    from cc_pushback.models import FeedbackCandidate
    from cc_pushback.repo import MatchRow
    from cc_pushback.sources import TranscriptSource

__all__ = ["main"]

DEFAULT_TRANSCRIPTS = (Path.home() / ".claude" / "projects",)
SOURCE_KINDS = get_args(SourceKind)
TRANSCRIPT_SOURCES: dict[SourceKind, TranscriptSource] = {
    "transcript_message": TranscriptMessages(),
    "plan_review": PlanReviews(),
    "interrupt_rejection": Interrupts(),
}
BACKENDS: dict[str, type[LlmBackend]] = {"claude": ClaudeBackend, "codex": CodexBackend}
MODELS: tuple[TModel, ...] = get_args(TModel)


def wanted(source_kind: str, selected: Sequence[str]) -> bool:
    return not selected or source_kind in selected


def transcript_candidates(
    path: Path, events: Sequence[TranscriptEvent], selected: Sequence[str]
) -> Iterator[FeedbackCandidate]:
    return (
        candidate
        for kind, source in TRANSCRIPT_SOURCES.items()
        if wanted(kind, selected)
        for candidate in source.candidates_for_file(path, events)
    )


def tally(counts: dict[str, int]) -> str:
    return ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())) or "no new rows"


@click.group()
@click.version_option(package_name="cc-pushback")
def main() -> None:
    """Learn your pushback style from past Claude Code feedback and code reviews.

    Replicate it with a language model.
    """


@main.command()
@click.option(
    "--transcripts",
    "transcripts",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Transcript directories to scan. Defaults to ~/.claude/projects.",
)
@click.option(
    "--issues",
    "issues",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Roots to scan for .context/cleanup/issues.jsonl superset issues.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Choice(SOURCE_KINDS),
    help="Restrict to these source kinds. Defaults to all.",
)
@click.option("--no-github", is_flag=True, help="Skip the GitHub review source.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
def scan(
    transcripts: tuple[Path, ...],
    issues: tuple[Path, ...],
    sources: tuple[str, ...],
    no_github: bool,
    db: Path | None,
) -> None:
    """Scan transcripts, GitHub, and issue files for feedback, incrementally.

    Each file is parsed only when new or modified since the last scan, and every
    candidate is inserted with ``INSERT OR IGNORE`` keyed by a content digest, so
    re-running ``scan`` over unchanged inputs is a no-op. Recording a file and
    inserting its candidates commit in one transaction.
    """
    counts: dict[str, int] = {}
    with Repository.open(db or Repository.default_path()) as repo:
        if any(wanted(kind, sources) for kind in TRANSCRIPT_SOURCES):
            for path, mtime, events in changed_files(repo, transcripts or DEFAULT_TRANSCRIPTS):
                counts["transcripts"] = counts.get("transcripts", 0) + repo.record_file_scan(
                    str(path), mtime, list(transcript_candidates(path, events, sources))
                )

        if wanted("superset_issue", sources):
            for path, mtime in changed_issue_files(repo, issues):
                counts["superset_issue"] = counts.get("superset_issue", 0) + repo.record_file_scan(
                    str(path), mtime, list(SupersetIssues().candidates_for_file(path, mtime))
                )

        if not no_github and wanted("github_review", sources):
            source_key, cursor, candidates = GitHubReviews().candidates(repo, cwd=Path.cwd())
            if source_key:
                counts["github_review"] = repo.advance_github_cursor(source_key, cursor, candidates)

    click.echo(tally(counts))


def novel_names(rows: Sequence[MatchRow]) -> list[str]:
    return sorted({row.pattern_name for row in rows if row.novel})


@main.command()
@click.option(
    "--backend",
    type=click.Choice(list(BACKENDS)),
    default="claude",
    show_default=True,
    help="Language-model CLI backend to classify with.",
)
@click.option(
    "--model",
    type=click.Choice(MODELS),
    default="small",
    show_default=True,
    help="Abstract model size to map onto the backend.",
)
@click.option("--limit", type=int, default=None, help="Maximum events to classify. Defaults to all.")
@click.option("--no-llm", is_flag=True, help="Run only the cheap matcher pass, skipping the language model.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
def classify(backend: str, model: TModel, limit: int | None, no_llm: bool, db: Path | None) -> None:
    """Classify ingested feedback against the pattern taxonomy.

    The cheap matcher pass labels events whose text or structure matches a seed
    pattern, with no language model. Unless ``--no-llm`` is set, every loaded
    event is then classified by the language model, which assigns severity, the
    rule, and any novel pattern the taxonomy does not yet name. Both passes write
    with ``INSERT OR IGNORE`` keyed by the taxonomy and prompt versions, so
    re-running ``classify`` over already-classified events adds nothing.
    """
    with Repository.open(db or Repository.default_path()) as repo:
        events = unclassified_events(
            repo,
            taxonomy_version=TAXONOMY_VERSION,
            prompt_version=PROMPT_VERSION,
            backend=backend,
            limit=limit,
        )
        matcher_rows = cheap_pass(events)
        model_rows = (
            []
            if no_llm
            else asyncio.run(llm_pass(events, backend=BACKENDS[backend](), backend_name=backend, model=model))
        )
        written = repo.save_matches([*matcher_rows, *model_rows])

    click.echo(f"events: {len(events)}  matcher rows: {len(matcher_rows)}  llm rows: {len(model_rows)}  new: {written}")
    if names := novel_names(model_rows):
        click.echo("novel proposals: " + ", ".join(names))


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
def stats(db: Path | None) -> None:
    """Print ingestion counts by source kind, file count, and source cursors."""
    with Repository.open(db or Repository.default_path()) as repo:
        report = repo.stats()
        click.echo(f"total: {report.total}  files: {report.files}")
        for kind, count in report.by_source.items():
            click.echo(f"  {kind}: {count}")
        for key, cursor in report.cursors.items():
            click.echo(f"  cursor {key}: {cursor}")


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
    with Repository.open(db or Repository.default_path()) as repo:
        for row in repo.recent(source_kind=source, limit=limit):
            click.echo(f"[{row['source_kind']}] {row['occurred_at']}  {str(row['text'])[:200]}")
