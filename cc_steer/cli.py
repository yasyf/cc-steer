"""The ``cc-steer`` command-line interface: scan, triage, audit, eval, and friends."""

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

from cc_steer import hooks as hook_wiring
from cc_steer import launchd, registry
from cc_steer.claude import claude_available
from cc_steer.dashboard import build_app
from cc_steer.evaluate import evaluate, flip_report
from cc_steer.journal import Journal
from cc_steer.models import STEERING_SOURCE_KINDS, SourceKind
from cc_steer.pipeline import ENRICH_LIMIT, REFINE_LIMIT, TRIAGE_LIMIT, run_pipeline
from cc_steer.report import Sample, build_summary, golden_label, project_label
from cc_steer.scan import scan as run_scan
from cc_steer.serve import serve
from cc_steer.store import FeedbackStore
from cc_steer.triage import PROMPT_VERSION
from cc_steer.triage import audit as run_audit
from cc_steer.triage import triage as run_triage

if TYPE_CHECKING:
    from spawnllm import TModel

SOURCE_KINDS = [*STEERING_SOURCE_KINDS]
TIERS = ["small", "medium", "large"]
DATASET_DIR = Path.home() / ".cc-steer" / "dataset"
sync_option = click.option(
    "--sync/--no-sync",
    default=True,
    show_default=True,
    help="Rebuild the derived dataset and push it to your private HuggingFace repo when the pass changed data.",
)


def coro[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, R]:
    """Adapts an async command body into the sync callback Click expects."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return anyio.run(functools.partial(fn, *args, **kwargs))

    return wrapper


@functools.cache
def hf_repo_id() -> str:
    """Resolves the dataset repo in the authenticated HF user's namespace: ``<hf-user>/cc-steer-traces``."""
    from huggingface_hub import HfApi

    return f"{HfApi().whoami()['name']}/cc-steer-traces"


def _mlx_importable() -> bool:
    """Whether the ``mlx`` extra is installed."""
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


async def sync_dataset(store: FeedbackStore) -> None:
    """Rebuilds the derived dataset and pushes every config to the user's private HF repo."""
    from cc_steer.export import export as run_export

    click.echo(f"syncing dataset to {(repo_id := hf_repo_id())}")
    report = await run_export(store, out=DATASET_DIR, push_to=repo_id)
    click.echo("synced " + "  ".join(f"{config} {sum(splits.values())}" for config, splits in report.counts.items()))


@click.group()
@click.version_option(package_name="cc-steer")
def main() -> None:
    """Collect developer steering signals from existing Claude Code transcripts."""


@main.command()
@click.option(
    "--transcripts",
    "transcripts",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Transcript directories to scan. Defaults to ~/.claude/projects.",
)
@click.option(
    "--findings",
    "findings",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directories to search for superset issues.jsonl findings files. May be repeated.",
)
@click.option("--full", is_flag=True, help="Re-scan every transcript, ignoring recorded mtimes.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@sync_option
@coro
async def scan(
    transcripts: tuple[Path, ...], findings: tuple[Path, ...], full: bool, db: Path | None, sync: bool
) -> None:
    """Scan transcripts for feedback, incrementally.

    Each transcript is parsed only when new or modified since the last scan, and
    every candidate is inserted with ``INSERT OR IGNORE`` keyed by a content
    digest, so re-running ``scan`` over unchanged inputs is a no-op. Recording a
    file and inserting its candidates commit in one transaction. With ``--findings``,
    superset ``issues.jsonl`` files under the given directories are anchored to the
    closest session and recorded through the same idempotent insert. A pass that
    changes data syncs the dataset to HuggingFace; ``--no-sync`` skips it.
    """
    roots = transcripts or (CLAUDE_PROJECTS_DIR,)
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_scan(store, roots, findings_dirs=findings, full=full)
        click.echo(f"scanned {report.scanned} files, {report.inserted} new rows")
        if sync and report.inserted:
            await sync_dataset(store)


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
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
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
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
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@sync_option
@coro
async def triage(
    tier: TModel, limit: int | None, concurrency: int, refresh_summary: bool, db: Path | None, sync: bool
) -> None:
    """Judge every stored candidate lacking a verdict at the current prompt version.

    Incremental and idempotent: verdicts persist per row as soon as each call
    completes, failed rows stay pending and are retried on the next run, and
    re-running over a fully judged corpus is a no-op. With ``--refresh-summary``,
    rows judged at summary fidelity are re-judged; a full-fidelity verdict
    replaces the summary one once the row's window hydrates again. A pass that
    changes data syncs the dataset to HuggingFace; ``--no-sync`` skips it.
    """
    from cc_transcript.judge import resolved_model

    from cc_steer.triage import JUDGE

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        pending = len(await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION, refresh_summary=refresh_summary))
        click.echo(f"pending: {pending} rows at prompt v{PROMPT_VERSION} ({resolved_model(tier)})")
        report = await run_triage(
            store, tier=tier, limit=limit, concurrency=concurrency, refresh_summary=refresh_summary
        )
        click.echo(f"judged {report.judged} rows ({report.failed} failed), {report.pending} pending")
        if sync and report.judged:
            await sync_dataset(store)


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
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@sync_option
@coro
async def audit(
    tier: TModel, accepts: int, rejects: int, seed: int, concurrency: int, db: Path | None, sync: bool
) -> None:
    """Audit a seeded stratified sample of the current prompt version's verdicts.

    The auditor is a stronger model, blind to the judge's verdicts; its labels are
    keyed independently of the judge's prompt version, so they accumulate across
    iterations and re-auditing a sampled row costs nothing. A pass that changes
    data syncs the dataset to HuggingFace; ``--no-sync`` skips it.
    """
    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_audit(store, accepts=accepts, rejects=rejects, seed=seed, tier=tier, concurrency=concurrency)
        click.echo(f"audited {report.judged} fresh rows ({report.failed} failed)")
        if sync and report.judged:
            await sync_dataset(store)


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
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
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
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@sync_option
@coro
async def refine(tier: TModel, limit: int | None, concurrency: int, db: Path | None, sync: bool) -> None:
    """Refine every accepted steering event into atomic training pairs.

    Incremental and idempotent: pairs commit per event as soon as each call
    completes, failed events stay pending and are retried on the next run, and
    re-running over a fully refined corpus is a no-op. A pass that changes data
    syncs the dataset to HuggingFace; ``--no-sync`` skips it.
    """
    from cc_transcript.judge import resolved_model

    from cc_steer.refine import PROMPT_VERSION as REFINE_VERSION
    from cc_steer.refine import refine as run_refine

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        pending = len(await store.unrefined(prompt_version=REFINE_VERSION, model=resolved_model(tier)))
        click.echo(f"pending: {pending} events at refine v{REFINE_VERSION} ({resolved_model(tier)})")
        report = await run_refine(store, tier=tier, limit=limit, concurrency=concurrency)
        click.echo(
            f"refined {report.refined} events into {report.pairs} pairs ({report.failed} failed), "
            f"{report.pending} pending"
        )
        if sync and report.refined:
            await sync_dataset(store)


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
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@sync_option
@coro
async def enrich(tier: TModel, limit: int | None, concurrency: int, db: Path | None, sync: bool) -> None:
    """Ground every refined pair in the code evidence behind it.

    Hands each pair's steering anchor and direction to cc-transcript's shared
    correction extractor, which harvests the candidate edits and their later
    corrections (from the session, or from git history), picks the one the direction
    faults — an LLM call when a backend is ready, the best-overlap candidate
    otherwise — and appends it to the shared ``corrections`` ledger. Anchors that
    yield no correction (expired transcripts, editless windows) cost no LLM call.
    Incremental and idempotent: a pair settles once its anchor carries a ledger row,
    a failure aborts the pass loudly (corrections already appended to the ledger
    persist, so a re-run resumes), and a refine re-run resurfaces its new pairs here
    automatically. A pass that changes data syncs the dataset to HuggingFace;
    ``--no-sync`` skips it.
    """
    from cc_transcript.corrections import CorrectionLog

    from cc_steer.enrich import enrich as run_enrich

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        pending = len(await store.unenriched(CorrectionLog.open()))
        click.echo(f"pending: {pending} pairs")
        report = await run_enrich(store, tier=tier, limit=limit, concurrency=concurrency)
        click.echo(
            f"enriched {report.enriched} pairs ({report.corrections} corrections, {report.skipped} skipped), "
            f"{report.pending} pending"
        )
        click.echo(f"recorded {report.corrections} corrections to the shared ledger (~/.cc-transcript/corrections.db)")
        if sync and report.corrections:
            await sync_dataset(store)


@main.command()
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=DATASET_DIR,
    show_default=True,
    help="Directory to write the per-config parquet files and dataset card into.",
)
@click.option(
    "--repo-id",
    default=None,
    show_default="<hf-user>/cc-steer-traces",
    help="HuggingFace dataset repo to push to.",
)
@click.option(
    "--push/--no-push", default=False, show_default=True, help="Push every config to --repo-id as a private dataset."
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@coro
async def export(out: Path, repo_id: str | None, push: bool, db: Path | None) -> None:
    """Export the steering lineage as a HuggingFace dataset.

    Builds the canonical ``traces`` config — one row per judged event, carrying
    the context, judge and auditor verdicts, refined pairs, and code evidence —
    plus the TRL-ready ``sft``, ``dpo``, and ``kto`` projections. Both source
    databases are read-only; every config lands as per-split parquet under
    ``--out`` next to a generated dataset card, and ``--push`` uploads every
    config to a private dataset in your HF namespace (created on first push),
    ``--repo-id`` overriding the target.
    """
    from cc_steer.export import export as run_export

    push_to = (repo_id or hf_repo_id()) if push else None
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_export(store, out=out, push_to=push_to)
    for config, splits in report.counts.items():
        click.echo(f"{config}: " + "  ".join(f"{split} {count}" for split, count in splits.items()))
    click.echo(f"wrote {report.out}" + (f", pushed to {push_to}" if report.pushed else ""))


@main.command()
@click.option("--jsonl", is_flag=True, help="Emit full pairs as JSON lines for fine-tuning export.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@coro
async def pairs(jsonl: bool, db: Path | None) -> None:
    """Print the refined training pairs — the pipeline's deliverable.

    Each pair is one atomic direction: a faithful re-synthesis of what Claude did,
    the verbatim user excerpt, and the distilled one-sentence direction.
    """
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        rows = await store.pairs()
    for row in rows:
        if jsonl:
            click.echo(json.dumps(row | {"project": project_label(str(row["origin_path"] or ""))}))
        else:
            click.echo(f"[{row['category']}] {str(row['action'])[:80]} -> {str(row['direction'])[:100]}")


@main.command(name="sample-negatives")
@click.option("--seed", type=int, default=1, show_default=True, help="Deterministic sampling seed.")
@click.option("--sessions", type=int, default=400, show_default=True, help="Maximum transcripts to parse this pass.")
@click.option("--per-session", type=int, default=20, show_default=True, help="Random negatives per transcript.")
@click.option("--resample", is_flag=True, help="Revisit sessions that already carry random samples.")
@click.option(
    "--transcripts",
    "transcripts",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Transcript roots to mine. Defaults to ~/.claude/projects plus the mirror corpus.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@coro
async def sample_negatives_(
    seed: int, sessions: int, per_session: int, resample: bool, transcripts: tuple[Path, ...], db: Path | None
) -> None:
    """Sample gate training windows: rewound positives, hard and random negatives.

    Positive windows and hard negatives are recomputed from the judged corpus and
    deduped by key; random negatives parse a budgeted, seed-deterministic batch of
    transcripts that carry none yet, excluding anything near a detected event, so
    repeated passes extend coverage. No LLM calls.
    """
    from cc_steer.negatives import sample_negatives as run_sample
    from cc_steer.pipeline import scan_roots

    roots = transcripts or scan_roots()
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_sample(
            store, roots, seed=seed, sessions=sessions, per_session=per_session, resample=resample
        )
        click.echo(
            "  ".join(f"{kind} +{count}" for kind, count in report.inserted.items())
            + f" ({report.sessions_sampled} transcripts parsed)"
        )
        totals = await store.gate_sample_stats()
        click.echo("total: " + "  ".join(f"{kind} {count}" for kind, count in totals.items()))


@main.command(name="index")
@click.option(
    "--model", default="voyage-4-large", show_default=True, help="Embedding model for the exemplar index."
)
@click.option("--batch", type=int, default=32, show_default=True, help="Encode batch size.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@coro
async def index_(model: str, batch: int, db: Path | None) -> None:
    """Embed the accepted steering exemplars for the frontier refiner's retrieval.

    Incremental by content digest — only exemplars whose rendered context changed
    re-embed. Train-split events only, so evaluation retrieval is never
    contaminated. Requires the ``embed`` extra.
    """
    from cc_steer.exemplars import build_index

    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await build_index(store, model=model, batch=batch)
        click.echo(f"embedded {report.embedded}, current {report.current}, eligible {report.total}")


@main.group(name="hooks")
def hooks_group() -> None:
    """Manage the global SessionEnd hook that feeds continual collection."""


@hooks_group.command(name="install")
@click.option(
    "--prefix",
    default=hook_wiring.DEFAULT_PREFIX,
    show_default=True,
    help="Command prefix the hook invokes cc-steer with.",
)
@click.option(
    "--settings",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Settings file. Defaults to ~/.claude/settings.json.",
)
def hooks_install(prefix: str, settings: Path | None) -> None:
    """Wire an async SessionEnd scan into the user-level Claude Code settings.

    Every session on the machine then feeds the store incrementally as it ends;
    the LLM stages and the HF sync stay with the scheduled pipeline. Idempotent:
    re-running updates the one cc-steer-owned group in place and preserves every
    other hook untouched.
    """
    result = hook_wiring.install(settings, prefix=prefix)
    click.echo(f"{result}: {hook_wiring.scan_command(prefix)!r} on {hook_wiring.EVENT}")


@hooks_group.command(name="uninstall")
@click.option(
    "--settings",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Settings file. Defaults to ~/.claude/settings.json.",
)
def hooks_uninstall(settings: Path | None) -> None:
    """Remove the SessionEnd scan hook, leaving every other hook untouched."""
    click.echo(hook_wiring.uninstall(settings))


@hooks_group.command(name="status")
@click.option(
    "--settings",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Settings file. Defaults to ~/.claude/settings.json.",
)
def hooks_status(settings: Path | None) -> None:
    """Show the scan command currently wired at SessionEnd, if any."""
    command = hook_wiring.installed_command(settings)
    click.echo(f"installed: {command!r}" if command else "not installed")


@main.group(name="models")
def models_group() -> None:
    """Inspect and flip the registry of lab-trained model versions."""


@models_group.command(name="list")
@click.argument("component", required=False)
def models_list(component: str | None) -> None:
    """List registered versions, oldest first; the promoted one is marked ``*``.

    With no COMPONENT, every component in the registry is listed.
    """
    names = [component] if component else registry.components()
    if not names:
        click.echo("no registered models")
        return
    for name in names:
        promoted = registry.current(name)
        rows = registry.versions(name)
        if not rows:
            click.echo(f"{name}: no registered versions")
            continue
        for info in rows:
            marker = " *" if promoted is not None and info.version == promoted.version else ""
            metrics = info.metadata.get("metrics")
            pr_auc = f"  pr_auc={value:.4f}" if isinstance(metrics, dict) and (value := metrics.get("pr_auc")) else ""
            click.echo(f"{name} {info.version}{marker}{pr_auc}")


@models_group.command(name="promote")
@click.argument("component")
@click.argument("version")
def models_promote(component: str, version: str) -> None:
    """Atomically point ``current`` at VERSION (full name or its ``v<NNN>`` prefix)."""
    try:
        registry.promote(component, version)
    except registry.RegistryError as error:
        raise click.ClickException(str(error)) from error
    promoted = registry.current(component)
    assert promoted is not None
    click.echo(f"promoted {component} {promoted.version}")


@models_group.command(name="rollback")
@click.argument("component")
def models_rollback(component: str) -> None:
    """Flip ``current`` back to the version registered before it."""
    try:
        info = registry.rollback(component)
    except registry.RegistryError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"rolled back {component} to {info.version}")


@main.group(name="pipeline")
def pipeline_group() -> None:
    """Run the collection stages as one budgeted, schedulable pass."""


@pipeline_group.command(name="run")
@click.option("--weekly", is_flag=True, help="Also run the auditor and the mechanical eval this pass.")
@click.option("--auto-weekly", is_flag=True, help="Treat Sunday runs as weekly; the launchd agent's mode.")
@click.option("--push/--no-push", default=True, show_default=True, help="Push the export to HuggingFace.")
@click.option("--triage-limit", type=int, default=TRIAGE_LIMIT, show_default=True, help="Judge at most this many.")
@click.option("--refine-limit", type=int, default=REFINE_LIMIT, show_default=True, help="Refine at most this many.")
@click.option("--enrich-limit", type=int, default=ENRICH_LIMIT, show_default=True, help="Enrich at most this many.")
@click.option("--concurrency", type=int, default=8, show_default=True, help="Maximum concurrent claude subshells.")
@click.option(
    "--journal-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Repository whose cc-notes journal records this pass.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@coro
async def pipeline_run(
    weekly: bool,
    auto_weekly: bool,
    push: bool,
    triage_limit: int,
    refine_limit: int,
    enrich_limit: int,
    concurrency: int,
    journal_repo: Path | None,
    db: Path | None,
) -> None:
    """Run one budgeted pass over every stage: scan, triage, refine, enrich, export.

    A weekly pass adds the auditor and the mechanical eval before the export. A
    stage failure is recorded and skipped past so later stages still run, a
    failed HF push downgrades to a local-only export, and with ``--journal-repo``
    the pass appends its one-line summary to that repo's cc-notes journal. Exits
    nonzero when any stage failed.
    """
    from datetime import date

    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    is_weekly = weekly or (auto_weekly and date.today().weekday() == 6)
    push_to = None
    if push:
        try:
            push_to = hf_repo_id()
        except Exception as error:  # noqa: BLE001 — no HF auth downgrades to a local export
            click.echo(f"push disabled (HF auth unavailable: {type(error).__name__})", err=True)
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        report = await run_pipeline(
            store,
            out=DATASET_DIR,
            push_to=push_to,
            weekly=is_weekly,
            triage_limit=triage_limit,
            refine_limit=refine_limit,
            enrich_limit=enrich_limit,
            concurrency=concurrency,
        )
    for outcome in report.outcomes:
        click.echo(("FAIL " if not outcome.ok else "") + f"{outcome.stage}: {outcome.summary}")
    if journal_repo is not None:
        line = ("weekly | " if is_weekly else "") + report.summary_line()
        if not Journal(journal_repo).append(line):
            click.echo("journal: not recorded (cc-notes missing or repo uninitialized)", err=True)
    if report.failed:
        raise click.ClickException(f"stages failed: {', '.join(report.failed)}")


@pipeline_group.command(name="install-launchd")
@click.option(
    "--prefix",
    default=hook_wiring.DEFAULT_PREFIX,
    show_default=True,
    help="Command prefix the agent invokes cc-steer with.",
)
@click.option(
    "--journal-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Repository whose cc-notes journal records each pass.",
)
@click.option("--hour", type=int, default=3, show_default=True, help="Local hour the nightly pass fires at.")
@click.option(
    "--retrain/--no-retrain",
    "retrain",
    default=True,
    show_default=True,
    help="Also install the weekly gate-retrain agent, run through the lab checkout.",
)
@click.option(
    "--lab",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    show_default="~/Code/cc-steer-lab",
    help="The cc-steer-lab checkout the retrain agent runs harness.retrain in.",
)
@click.option("--retrain-hour", type=int, default=4, show_default=True, help="Local hour the Sunday retrain fires at.")
@click.option(
    "--watch/--no-watch",
    "watch",
    default=True,
    show_default=True,
    help="Also install the always-on shadow watch daemon under KeepAlive.",
)
def pipeline_install_launchd(
    prefix: str, journal_repo: Path | None, hour: int, retrain: bool, lab: Path | None, retrain_hour: int, watch: bool
) -> None:
    """Schedule the pass nightly — plus the weekly model retrain and the shadow watcher — via macOS LaunchAgents.

    The pipeline agent covers both collection cadences: it runs ``pipeline run
    --auto-weekly``, so the Sunday pass folds in the auditor and eval. The
    retrain agent runs the lab's ``harness.retrain`` every Sunday, refreshing
    the promoted gate model when the training data moved (``--no-retrain``
    skips it). The watch agent runs ``cc-steer watch`` continuously under
    ``KeepAlive`` so a fail-fast crash respawns (``--no-watch`` skips it). Logs
    land under ``~/.cc-steer/logs/``. Re-running replaces the agents in place.
    """
    path = launchd.install(prefix, journal_repo, hour=hour)
    click.echo(f"installed {launchd.LABEL} ({path}): nightly {hour:02d}:00, weekly audit on Sundays")
    if watch:
        watch_path = launchd.install_watch(prefix)
        click.echo(f"installed {launchd.WATCH_LABEL} ({watch_path}): always-on shadow watcher (KeepAlive)")
    if not retrain:
        return
    lab_dir = lab or launchd.LAB_DIR
    if not lab_dir.is_dir():
        raise click.ClickException(f"no cc-steer-lab checkout at {lab_dir}; pass --lab or --no-retrain")
    retrain_path = launchd.install_retrain(lab_dir, hour=retrain_hour)
    click.echo(f"installed {launchd.RETRAIN_LABEL} ({retrain_path}): Sundays {retrain_hour:02d}:00, gate retrain")


@pipeline_group.command(name="uninstall-launchd")
def pipeline_uninstall_launchd() -> None:
    """Unload and remove the nightly pipeline, weekly retrain, and shadow watch LaunchAgents."""
    click.echo(f"{launchd.LABEL}: " + ("removed" if launchd.uninstall() else "not installed"))
    click.echo(f"{launchd.WATCH_LABEL}: " + ("removed" if launchd.uninstall_watch() else "not installed"))
    click.echo(f"{launchd.RETRAIN_LABEL}: " + ("removed" if launchd.uninstall_retrain() else "not installed"))


@main.command(name="watch")
@click.option(
    "--shadow/--live",
    "shadow_mode",
    default=True,
    show_default=True,
    help="Delivery mode; live delivery is not yet implemented.",
)
@click.option(
    "--root",
    "roots",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Transcript roots to tail. Defaults to ~/.claude/projects.",
)
@click.option(
    "--gate",
    "gate_kind",
    type=click.Choice(["heuristic", "lexical"]),
    default=None,
    show_default="lexical when a gate model is promoted, else heuristic",
    help="Stage-1 gate: the lab-trained lexical model (from the registry) or the turn-floor heuristic.",
)
@click.option(
    "--gate-threshold",
    type=float,
    default=None,
    show_default="the trained threshold (lexical) or 0.5 (heuristic)",
    help="Stage-1 gate score below which a turn is suppressed.",
)
@click.option(
    "--drafter",
    "drafter_kind",
    type=click.Choice(["auto", "spawn", "mlx"]),
    default="auto",
    show_default=True,
    help="Stage-2 drafter: the local trained watcher (mlx) or the claude CLI (spawn); "
    "auto picks mlx when a watcher model is promoted and the mlx extra is installed.",
)
@click.option(
    "--stage2-threshold",
    type=float,
    default=None,
    show_default="the promoted watcher's budget threshold",
    help="Local drafter abstain threshold on P(NO_STEER); ignored for the spawn drafter.",
)
@click.option(
    "--refiner",
    "refiner_kind",
    type=click.Choice(["auto", "spawn", "none"]),
    default="auto",
    show_default=True,
    help="Stage 3: the claude CLI refiner or none (a fired draft ships as-is); "
    "auto disables it for the mlx drafter (two-stage, per E2) and keeps it for spawn.",
)
@click.option(
    "--debounce",
    type=float,
    default=2.0,
    show_default=True,
    help="Seconds a session must stay quiet before its last completed turn is evaluated.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@click.option(
    "--shadow-db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Shadow ledger path. Defaults to ~/.cc-steer/shadow.db.",
)
@coro
async def watch_(
    shadow_mode: bool,
    roots: tuple[Path, ...],
    gate_kind: str | None,
    gate_threshold: float | None,
    drafter_kind: str,
    stage2_threshold: float | None,
    refiner_kind: str,
    debounce: float,
    db: Path | None,
    shadow_db: Path | None,
) -> None:
    """Tail live transcripts and run the steering cascade in shadow mode.

    Every open session is followed as it writes; each time one goes quiet
    after completing a turn, the cascade — stage-1 gate, drafting model,
    optional exemplar-conditioned refiner — decides whether the user would have
    steered right there. Stage 1 defaults to the promoted lexical gate from the
    model registry, thresholded at its trained 2 fires/100 turns budget
    (``--gate-threshold`` overrides); without a promoted version it falls back
    to the turn-floor heuristic. Stage 2 defaults to the promoted local watcher
    (the mlx extra) when one exists, abstaining at its trained budget threshold
    on P(NO_STEER); stage 3 is then disabled — the E2-validated two-stage
    configuration — unless ``--refiner spawn`` re-enables it. Proposals land in
    the shadow ledger (``cc-steer shadow report`` measures them); no session is
    ever touched. Exemplar retrieval needs the ``embed`` extra and a built
    index (``cc-steer index``); without an index the watcher still runs, with
    stage 3 unconditioned. Runs until interrupted.
    """
    from cc_steer.exemplars import load_index, query_encoder
    from cc_steer.watcher.cascade import Cascade, Drafter, Gate, HeuristicGate, Refiner, SpawnDrafter, SpawnRefiner
    from cc_steer.watcher.daemon import Watcher
    from cc_steer.watcher.delivery import ShadowDelivery
    from cc_steer.watcher.gate import LexicalGate
    from cc_steer.watcher.types import CascadeConfig

    if not shadow_mode:
        raise click.ClickException("live delivery is not yet implemented; run without --live to shadow")
    if gate_kind is None:
        gate_kind = "lexical" if registry.current("gate") is not None else "heuristic"
        if gate_kind == "heuristic":
            click.echo("no promoted gate model; falling back to the heuristic gate", err=True)
    if drafter_kind == "auto":
        drafter_kind = "mlx" if registry.current("watcher") is not None and _mlx_importable() else "spawn"
        if drafter_kind == "spawn":
            click.echo("no promoted watcher model (or mlx extra missing); drafting via the claude CLI", err=True)
    if refiner_kind == "auto":
        refiner_kind = "none" if drafter_kind == "mlx" else "spawn"
    if (drafter_kind == "spawn" or refiner_kind == "spawn") and not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")

    drafter: Drafter
    stage2_model = "medium"
    render_version = 1
    if drafter_kind == "mlx":
        from cc_steer.watcher.drafter_mlx import MlxDrafter

        try:
            mlx_drafter = MlxDrafter(threshold=stage2_threshold)
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error
        drafter = mlx_drafter
        stage2_model = mlx_drafter.base_model
        stage2_threshold = mlx_drafter.threshold
        render_version = mlx_drafter.render_version
        click.echo(
            f"drafter: mlx {mlx_drafter.version.version} "
            f"(P(NO_STEER) abstain threshold {mlx_drafter.threshold:.4f} [{mlx_drafter.operating_point}], "
            f"render v{render_version})"
        )
    else:
        drafter = SpawnDrafter(model=stage2_model)
        stage2_threshold = None
        click.echo("drafter: spawn (claude CLI, medium tier)")

    gate: Gate
    if gate_kind == "lexical":
        try:
            lexical = LexicalGate()
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error
        gate = lexical
        resolved_gate_threshold = gate_threshold if gate_threshold is not None else lexical.threshold
        gate_banner = f"gate: lexical {lexical.version.version} (threshold {resolved_gate_threshold:.4f})"
    else:
        resolved_gate_threshold = gate_threshold if gate_threshold is not None else 0.5
        gate_banner = f"gate: heuristic (threshold {resolved_gate_threshold})"
    config = CascadeConfig(
        gate_threshold=resolved_gate_threshold,
        stage2_model=stage2_model,
        stage2_threshold=stage2_threshold,
        drafter_kind=drafter_kind,
        render_version=render_version,
    )
    if gate_kind != "lexical":
        gate = HeuristicGate(min_turns=config.min_turns)
    click.echo(gate_banner)

    refiner: Refiner | None = SpawnRefiner(tier=config.stage3_tier) if refiner_kind == "spawn" else None
    click.echo(f"refiner: {refiner_kind}" + (" (fired drafts ship as-is)" if refiner is None else ""))
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        encoder = None
        if refiner is not None:
            keys, _ = await load_index(store, model=config.embed_model)
            if keys:
                try:
                    encoder = query_encoder(config.embed_model)
                except RuntimeError as error:
                    raise click.ClickException(str(error)) from error
            else:
                click.echo(
                    "retrieval disabled: the exemplar index is empty — run `cc-steer index` to enable it", err=True
                )
        cascade = Cascade(
            gate=gate,
            drafter=drafter,
            refiner=refiner,
            store=store,
            config=config,
            encoder=encoder,
        )
        async with await ShadowDelivery.open(shadow_db) as delivery:
            watcher = Watcher(cascade, delivery, roots=roots or (CLAUDE_PROJECTS_DIR,), debounce_s=debounce)
            click.echo(
                f"watching {len(watcher.roots)} root(s) in shadow mode; "
                f"proposals land in {shadow_db or ShadowDelivery.default_path()}"
            )
            await watcher.run()


@main.group(name="shadow")
def shadow_group() -> None:
    """Analyze the live watcher's shadow-mode proposals."""


@shadow_group.command(name="report")
@click.option(
    "--window",
    type=int,
    default=30,
    show_default=True,
    help="Minutes after a proposal within which a real intervention counts as a hit.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the summary as JSON.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@click.option(
    "--shadow-db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Shadow ledger path. Defaults to ~/.cc-steer/shadow.db.",
)
@click.option(
    "--journal-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Repository whose cc-notes journal records this report.",
)
@coro
async def shadow_report(
    window: int, as_json: bool, db: Path | None, shadow_db: Path | None, journal_repo: Path | None
) -> None:
    """Join shadow proposals against the interventions users actually made.

    Feedback events carry no turn index, so the join is time within a session:
    a steer is a HIT when the same session shows a real feedback event within
    ``--window`` minutes after the proposal fired, and a nuisance candidate
    otherwise. Also reports stage-2/3 abstention rates, per-category hit
    counts, the drafter's sentinel-probability distribution, proposals per
    session, and the sessions that produced proposals. No LLM calls.
    """
    from cc_steer.watcher.delivery import ShadowDelivery
    from cc_steer.watcher.shadow import intervention_rows, summarize

    async with await ShadowDelivery.open(shadow_db) as ledger:
        proposals = await ledger.proposals()
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        interventions = await intervention_rows(store)
    summary = summarize(proposals, interventions, window_minutes=window)
    payload = dataclasses.asdict(summary) | {"proposals_per_session": summary.proposals_per_session}
    if as_json:
        click.echo(json.dumps(payload))
        return
    click.echo(f"sessions with proposals: {summary.sessions}")
    per = f" ({summary.proposals_per_session:.1f}/session)" if summary.sessions else ""
    click.echo(f"proposals: {summary.proposals}{per}")
    stage2 = f" ({summary.stage2_abstained / summary.proposals:.0%})" if summary.proposals else ""
    click.echo(f"stage-2 abstained: {summary.stage2_abstained}/{summary.proposals}{stage2}")
    drafted = summary.proposals - summary.stage2_abstained
    stage3 = f" ({summary.stage3_abstained / drafted:.0%})" if drafted else ""
    click.echo(f"stage-3 abstained: {summary.stage3_abstained}/{drafted}{stage3}")
    click.echo(f"steers: {summary.steers} — hits {summary.hits}, nuisance {summary.nuisance} ({window}m window)")
    if summary.hit_categories:
        by_count = sorted(summary.hit_categories.items(), key=lambda item: -item[1])
        click.echo("hit categories: " + ", ".join(f"{category} {count}" for category, count in by_count))
    if (stats := summary.sentinel_probs) is not None:
        deciles = " ".join(f"{p:.3f}" for p in stats.deciles)
        click.echo(f"sentinel P(NO_STEER): n={stats.n} mean={stats.mean:.3f} deciles=[{deciles}]")
    if journal_repo is not None:
        line = f"shadow report | {json.dumps(payload, sort_keys=True)}"
        if not Journal(journal_repo, title="cc-steer shadow reports", tag="shadow").append(line):
            click.echo("journal: not recorded (cc-notes missing or repo uninitialized)", err=True)


@main.command(name="view-samples")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.cc-steer/feedback.db.",
)
@click.option("--model", default="claude-sonnet-4-6", show_default=True, help="Model for the claude CLI summary.")
@click.option("--port", type=int, default=0, show_default=True, help="Port to serve on; 0 picks a free one.")
@click.option("--open", "open_", is_flag=True, help="Open the dashboard in a browser once serving.")
@coro
async def view_samples(db: Path | None, model: str, port: int, open_: bool) -> None:
    """Serve the training-pairs dashboard: refined pairs and their full lineage.

    Opens an interactive dashboard listing the refined pairs (the pipeline's
    deliverable) and every candidate behind them, with a detail pane that walks one
    candidate's lineage — detector hit, judge verdicts across versions, the auditor's
    agreement, the refiner's atomic split, and the golden gate. It is served over a
    transient HTTP server whose URL is printed; press Ctrl-C to stop. The corpus
    narrative is written by the ``claude`` CLI.
    """
    if not claude_available():
        raise click.ClickException("the claude CLI is not on PATH")
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        samples = [Sample.from_row(row) for row in await store.candidates()]
        if not samples:
            raise click.ClickException("no judged samples to serve")
        summary = await build_summary(samples, model=model)
        await serve(build_app(store, summary=summary), port=port, open_browser=open_)
