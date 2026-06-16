"""The ``cc-pushback`` command-line interface: scan, triage, audit, eval, and friends."""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import click
from cc_transcript import CLAUDE_PROJECTS_DIR

from cc_pushback.claude import claude_available
from cc_pushback.dashboard import build_app
from cc_pushback.evaluate import evaluate, flip_report
from cc_pushback.models import PUSHBACK_SOURCE_KINDS, SourceKind
from cc_pushback.report import Sample, build_summary, golden_label, project_label
from cc_pushback.scan import scan as run_scan
from cc_pushback.serve import serve
from cc_pushback.store import FeedbackStore
from cc_pushback.triage import PROMPT_VERSION
from cc_pushback.triage import audit as run_audit
from cc_pushback.triage import triage as run_triage

if TYPE_CHECKING:
    from spawnllm import TModel

SOURCE_KINDS = [*PUSHBACK_SOURCE_KINDS]
TIERS = ["small", "medium", "large"]
PENDING_CAP = 1200


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
    """Print ingestion counts by source kind and triage coverage."""
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await store.stats()
        triaged = await store.triage_stats(prompt_version=PROMPT_VERSION)
    click.echo(f"total: {report.total}  files: {report.files}")
    for kind, count in report.by_source.items():
        click.echo(f"  {kind}: {count}")
    share = f" ({triaged.accepted / triaged.judged:.0%})" if triaged.judged else ""
    click.echo(f"triaged: {triaged.judged}/{triaged.total} (v{PROMPT_VERSION})  accepted: {triaged.accepted}{share}")
    for category, count in triaged.by_category.items():
        click.echo(f"  {category}: {count}")


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


@main.command()
@click.option(
    "--model", "tier", type=click.Choice(TIERS), default="medium", show_default=True, help="Judge model tier."
)
@click.option("--limit", type=int, default=None, help="Judge at most this many rows this pass.")
@click.option("--concurrency", type=int, default=8, show_default=True, help="Maximum concurrent claude subshells.")
@click.option(
    "--refresh-summary",
    "refresh_summary",
    is_flag=True,
    help="Also re-judge rows whose verdict was recorded at summary fidelity.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def triage(tier: TModel, limit: int | None, concurrency: int, refresh_summary: bool, db: Path | None) -> None:
    """Judge every stored candidate lacking a verdict at the current prompt version.

    Incremental and idempotent: verdicts persist per row as soon as each call
    completes, failed rows stay pending and are retried on the next run, and
    re-running over a fully judged corpus is a no-op. With ``--refresh-summary``,
    rows judged at summary fidelity are re-judged; a full-fidelity verdict
    replaces the summary one once the row's window hydrates again.
    """
    from cc_transcript.judge import resolved_model

    from cc_pushback.triage import JUDGE

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        pending = len(
            await store.unjudged(
                role=JUDGE, prompt_version=PROMPT_VERSION, model=resolved_model(tier), refresh_summary=refresh_summary
            )
        )
        if pending > PENDING_CAP:
            raise click.ClickException(f"{pending} pending rows exceeds the {PENDING_CAP} safety cap — wrong DB?")
        click.echo(f"pending: {pending} rows at prompt v{PROMPT_VERSION} ({resolved_model(tier)})")
        report = await run_triage(
            store, tier=tier, limit=limit, concurrency=concurrency, refresh_summary=refresh_summary
        )
    click.echo(f"judged {report.judged} rows ({report.failed} failed), {report.pending} pending")


@main.command()
@click.option("--accepts", type=int, default=60, show_default=True, help="Audit budget for judge-accepted rows.")
@click.option("--rejects", type=int, default=60, show_default=True, help="Audit budget for judge-rejected rows.")
@click.option("--seed", type=int, default=1, show_default=True, help="Deterministic sampling seed (iteration number).")
@click.option(
    "--model", "tier", type=click.Choice(TIERS), default="large", show_default=True, help="Auditor model tier."
)
@click.option("--concurrency", type=int, default=8, show_default=True, help="Maximum concurrent claude subshells.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def audit(tier: TModel, accepts: int, rejects: int, seed: int, concurrency: int, db: Path | None) -> None:
    """Audit a seeded stratified sample of the current prompt version's verdicts.

    The auditor is a stronger model, blind to the judge's verdicts; its labels are
    keyed independently of the judge's prompt version, so they accumulate across
    iterations and re-auditing a sampled row costs nothing.
    """
    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_audit(store, accepts=accepts, rejects=rejects, seed=seed, tier=tier, concurrency=concurrency)
    click.echo(f"audited {report.judged} fresh rows ({report.failed} failed)")


@main.command(name="eval")
@click.option("--seed", type=int, default=1, show_default=True, help="The seed the audit ran with.")
@click.option("--accepts", type=int, default=60, show_default=True, help="The audit's accept budget.")
@click.option("--rejects", type=int, default=60, show_default=True, help="The audit's reject budget.")
@click.option("--compare-to", type=int, default=None, help="Earlier prompt version for flip analysis.")
@click.option("--json", "as_json", is_flag=True, help="Emit the full metrics as JSON.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def eval_(seed: int, accepts: int, rejects: int, compare_to: int | None, as_json: bool, db: Path | None) -> None:
    """Compute the mechanical metrics for the current prompt version. No LLM calls.

    Recomputes everything from raw verdicts: the golden-set gate, audited precision
    and reject contamination over the reproduced uniform core, the cumulative-pool
    secondary estimates, per-kind tables, and (with ``--compare-to``) verdict flips
    against an earlier prompt version.
    """
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        metrics = await evaluate(store, seed=seed, accepts=accepts, rejects=rejects)
        flips = await flip_report(store, from_version=compare_to, to_version=PROMPT_VERSION) if compare_to else None
    if as_json:
        payload = dataclasses.asdict(metrics) | {
            "golden": dataclasses.asdict(metrics.golden)
            | {
                "failures": [
                    dataclasses.asdict(failure) | {"expected": golden_label(failure.expected)}
                    for failure in metrics.golden.failures
                ]
            },
            "precision": metrics.precision,
            "contamination": metrics.contamination,
            "contamination_upper": metrics.contamination_upper,
            "recall_hat": metrics.recall_hat,
            "flips": dataclasses.asdict(flips) if flips else None,
        }
        click.echo(json.dumps(payload, indent=2))
        return
    share = f" ({metrics.accepted / metrics.judged:.0%})" if metrics.judged else ""
    click.echo(
        f"prompt v{metrics.prompt_version}: judged {metrics.judged}/{metrics.total}, accepted {metrics.accepted}{share}"
    )
    click.echo(f"golden: {metrics.golden.passed}/{metrics.golden.total} (sha256 {metrics.golden.sha256[:12]})")
    for failure in metrics.golden.failures:
        why = f" — {failure.rationale}" if failure.rationale else ""
        click.echo(
            f"  FAIL expected {golden_label(failure.expected)}, got {failure.category}{why}: {failure.text[:120]}"
        )
    core_a, core_r = metrics.core_accepts, metrics.core_rejects
    click.echo(
        f"precision (core): {core_a.hits}/{core_a.audited}"
        + (f" = {p:.3f}" if (p := metrics.precision) is not None else "")
    )
    upper = f" (95% upper {u:.3f})" if (u := metrics.contamination_upper) is not None else ""
    click.echo(
        f"contamination (core): {core_r.hits}/{core_r.audited}"
        + (f" = {c:.3f}{upper}" if (c := metrics.contamination) is not None else "")
    )
    if (recall := metrics.recall_hat) is not None:
        click.echo(f"recall_hat: {recall:.3f}")
    pool_a, pool_r = metrics.pool_accepts, metrics.pool_rejects
    click.echo(f"pool: accepts {pool_a.hits}/{pool_a.audited}, rejects {pool_r.hits}/{pool_r.audited}")
    for kind, (judged, accepted) in sorted(metrics.by_kind.items()):
        click.echo(f"  {kind}: {accepted}/{judged} accepted")
    click.echo(f"disagreements: {len(metrics.disagreements)}")
    for item in metrics.disagreements:
        click.echo(
            f"  [{item.source_kind}] judge={item.judge_category} ({item.judge_rationale}) "
            f"auditor={item.auditor_category} ({item.auditor_rationale}): {item.text[:120]}"
        )
    if flips is not None:
        rate = f" ({r:.0%})" if (r := flips.rate) is not None else ""
        click.echo(f"flips vs v{compare_to}: {len(flips.flips)}/{flips.common}{rate}")
        for flip in flips.flips:
            click.echo(f"  {flip.from_category} -> {flip.to_category}: {flip.text[:120]}")


@main.command()
@click.option(
    "--model", "tier", type=click.Choice(TIERS), default="medium", show_default=True, help="Refiner model tier."
)
@click.option("--limit", type=int, default=None, help="Refine at most this many events this pass.")
@click.option("--concurrency", type=int, default=8, show_default=True, help="Maximum concurrent claude subshells.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def refine(tier: TModel, limit: int | None, concurrency: int, db: Path | None) -> None:
    """Refine every accepted pushback event into atomic training pairs.

    Incremental and idempotent: pairs commit per event as soon as each call
    completes, failed events stay pending and are retried on the next run, and
    re-running over a fully refined corpus is a no-op.
    """
    from cc_transcript.judge import resolved_model

    from cc_pushback.refine import PROMPT_VERSION as REFINE_VERSION
    from cc_pushback.refine import refine as run_refine

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        pending = len(await store.unrefined(prompt_version=REFINE_VERSION, model=resolved_model(tier)))
        if pending > PENDING_CAP:
            raise click.ClickException(f"{pending} pending events exceeds the {PENDING_CAP} safety cap — wrong DB?")
        click.echo(f"pending: {pending} events at refine v{REFINE_VERSION} ({resolved_model(tier)})")
        report = await run_refine(store, tier=tier, limit=limit, concurrency=concurrency)
    click.echo(
        f"refined {report.refined} events into {report.pairs} pairs ({report.failed} failed), {report.pending} pending"
    )


@main.command()
@click.option(
    "--model", "tier", type=click.Choice(TIERS), default="medium", show_default=True, help="Linking model tier."
)
@click.option("--limit", type=int, default=None, help="Enrich at most this many pairs this pass.")
@click.option("--concurrency", type=int, default=8, show_default=True, help="Maximum concurrent claude subshells.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def enrich(tier: TModel, limit: int | None, concurrency: int, db: Path | None) -> None:
    """Ground every refined pair in the code it complains about.

    Harvests candidate incorrect edits and their later corrections (from the
    session, or from git history) around each pair's pushback anchor, then has an
    LLM pick the one edit the complaint faults, copied verbatim. Expired
    transcripts and editless windows persist free ``no_code`` rows with no LLM
    call. Incremental and idempotent: evidence persists per pair as soon as each
    row resolves, failed pairs stay pending and are retried on the next run, and
    a refine re-run resurfaces its new pairs here automatically.
    """
    from cc_transcript.evidence import EXTRACTOR_VERSION
    from cc_transcript.judge import resolved_model

    from cc_pushback.enrich import ENRICH_VERSION
    from cc_pushback.enrich import enrich as run_enrich

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        pending = len(
            await store.unenriched(
                enrich_version=ENRICH_VERSION, enrich_model=resolved_model(tier), extractor_version=EXTRACTOR_VERSION
            )
        )
        if pending > PENDING_CAP:
            raise click.ClickException(f"{pending} pending pairs exceeds the {PENDING_CAP} safety cap — wrong DB?")
        click.echo(f"pending: {pending} pairs at enrich v{ENRICH_VERSION} ({resolved_model(tier)})")
        report = await run_enrich(store, tier=tier, limit=limit, concurrency=concurrency)
    click.echo(
        f"enriched {report.enriched} pairs ({report.code} code, {report.no_code} no_code, "
        f"{report.git} git-sourced, {report.failed} failed), {report.pending} pending"
    )


@main.command()
@click.option("--jsonl", is_flag=True, help="Emit full pairs as JSON lines for fine-tuning export.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-pushback/feedback.db.",
)
@coro
async def pairs(jsonl: bool, db: Path | None) -> None:
    """Print the refined training pairs — the pipeline's deliverable.

    Each pair is one atomic complaint: a faithful re-synthesis of what Claude did,
    the verbatim user excerpt, and the distilled one-sentence complaint.
    """
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        rows = await store.pairs()
    for row in rows:
        if jsonl:
            click.echo(json.dumps(row | {"project": project_label(str(row["origin_path"] or ""))}))
        else:
            click.echo(f"[{row['category']}] {str(row['action'])[:80]} -> {str(row['complaint'])[:100]}")


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
@click.option("--open", "open_", is_flag=True, help="Open the dashboard in a browser once serving.")
@coro
async def view_samples(db: Path | None, llm: bool, model: str, port: int, open_: bool) -> None:
    """Serve the training-pairs dashboard: refined pairs and their full lineage.

    Opens an interactive dashboard listing the refined pairs (the pipeline's
    deliverable) and every candidate behind them, with a detail pane that walks one
    candidate's lineage — detector hit, judge verdicts across versions, the auditor's
    agreement, the refiner's atomic split, and the golden gate. It is served over a
    transient HTTP server whose URL is printed; press Ctrl-C to stop. The corpus
    narrative is written by the ``claude`` CLI when ``--llm`` is set and ``claude`` is
    installed, falling back to deterministic heuristics.
    """
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        samples = [Sample.from_row(row) for row in await store.candidates()]
        summary = await build_summary(samples, use_llm=llm, model=model)
        await serve(build_app(store, summary=summary), port=port, open_browser=open_)
