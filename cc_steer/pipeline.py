"""The scheduled pipeline: every collection stage chained under budgets.

One pass sequences scan → triage → refine → enrich → export; a weekly pass adds
the auditor and the mechanical eval between enrich and export. Each stage is
budgeted so an unattended run can never exhaust an account, and a stage failure
is recorded and skipped past rather than aborting the pass — except the export
push, whose failure downgrades to a local-only export.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript import CLAUDE_PROJECTS_DIR

from cc_steer.enrich import enrich as run_enrich
from cc_steer.evaluate import evaluate
from cc_steer.export import export as run_export
from cc_steer.negatives import sample_negatives as run_negatives
from cc_steer.refine import refine as run_refine
from cc_steer.scan import scan as run_scan
from cc_steer.triage import audit as run_audit
from cc_steer.triage import triage as run_triage
from cc_steer.watcher.reactions import attribute_reactions

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from cc_steer.store import FeedbackStore

MIRRORS_DIR = Path.home() / ".cc-steer" / "mirrors"
TRIAGE_LIMIT = 400
REFINE_LIMIT = 400
ENRICH_LIMIT = 200
NEGATIVE_SESSIONS = 200
AUDIT_BUDGET = 60


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """One stage's one-line result.

    Attributes:
        stage: The stage name.
        summary: A one-line human-readable result.
        ok: Whether the stage completed without error.
    """

    stage: str
    summary: str
    ok: bool = True


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """The outcome of one pipeline pass."""

    outcomes: tuple[StageOutcome, ...]

    @property
    def failed(self) -> tuple[str, ...]:
        """The names of the stages that errored this pass."""
        return tuple(outcome.stage for outcome in self.outcomes if not outcome.ok)

    def summary_line(self) -> str:
        """The whole pass as one journal-ready line."""
        parts = (f"{o.stage}: {'FAILED — ' if not o.ok else ''}{o.summary}" for o in self.outcomes)
        return " | ".join(parts)


def weekly_seed(today: date | None = None) -> int:
    """A deterministic audit seed for the current ISO week, fresh samples each week."""
    year, week, _ = (today or date.today()).isocalendar()
    return year * 100 + week


def scan_roots() -> tuple[Path, ...]:
    """The live projects dir plus the rsync'd mirror corpus when present."""
    roots = [CLAUDE_PROJECTS_DIR]
    if MIRRORS_DIR.is_dir():
        roots.append(MIRRORS_DIR)
    return tuple(roots)


async def run_pipeline(
    store: FeedbackStore,
    *,
    out: Path,
    push_to: str | None,
    weekly: bool = False,
    triage_limit: int | None = TRIAGE_LIMIT,
    refine_limit: int | None = REFINE_LIMIT,
    enrich_limit: int | None = ENRICH_LIMIT,
    audit_seed: int | None = None,
    concurrency: int = 8,
) -> PipelineReport:
    """Runs one budgeted pass over every stage, collecting per-stage outcomes.

    Args:
        store: The feedback store every stage reads and writes.
        out: The directory the dataset export lands in.
        push_to: The HF dataset repo to push to, or ``None`` to export locally only.
        weekly: When set, also run the auditor and the mechanical eval.
        triage_limit: Judge at most this many rows.
        refine_limit: Refine at most this many events.
        enrich_limit: Enrich at most this many pairs.
        audit_seed: The audit's sampling seed; defaults to the current ISO week.
        concurrency: Maximum concurrent LLM subshells per stage.

    Returns:
        The :class:`PipelineReport` with one :class:`StageOutcome` per stage run.
    """
    seed = audit_seed if audit_seed is not None else weekly_seed()
    outcomes: list[StageOutcome] = []

    async def stage(name: str, run: Callable[[], Awaitable[str]]) -> None:
        try:
            outcomes.append(StageOutcome(stage=name, summary=await run()))
        except Exception as error:  # noqa: BLE001 — an unattended pass records and moves on
            outcomes.append(StageOutcome(stage=name, summary=f"{type(error).__name__}: {error}"[:200], ok=False))

    async def scan_() -> str:
        report = await run_scan(store, scan_roots())
        return f"scanned {report.scanned} files, {report.inserted} new rows"

    async def triage_() -> str:
        report = await run_triage(store, tier="large", limit=triage_limit, concurrency=concurrency)
        return f"judged {report.judged} ({report.failed} failed), {report.pending} pending"

    async def refine_() -> str:
        report = await run_refine(store, tier="large", limit=refine_limit, concurrency=concurrency)
        return f"refined {report.refined} into {report.pairs} pairs ({report.failed} failed), {report.pending} pending"

    async def enrich_() -> str:
        report = await run_enrich(store, tier="medium", limit=enrich_limit, concurrency=concurrency)
        return (
            f"enriched {report.enriched} (+{report.corrections} corrections, {report.failed} failed), "
            f"{report.pending} pending"
        )

    async def negatives_() -> str:
        report = await run_negatives(store, scan_roots(), seed=seed, sessions=NEGATIVE_SESSIONS)
        counts = "  ".join(f"{kind} +{n}" for kind, n in report.inserted.items())
        return f"{counts} ({report.sessions_sampled} transcripts parsed)"

    async def audit_() -> str:
        report = await run_audit(
            store, accepts=AUDIT_BUDGET, rejects=AUDIT_BUDGET, seed=seed, tier="large", concurrency=concurrency
        )
        return f"audited {report.judged} fresh rows ({report.failed} failed, seed {seed})"

    async def eval_() -> str:
        metrics = await evaluate(store, seed=seed, accepts=AUDIT_BUDGET, rejects=AUDIT_BUDGET)
        precision = f"{p:.3f}" if (p := metrics.precision) is not None else "n/a"
        contamination = f"{c:.3f}" if (c := metrics.contamination) is not None else "n/a"
        return (
            f"golden {metrics.golden.passed}/{metrics.golden.total}, "
            f"precision {precision}, contamination {contamination}"
        )

    async def reactions_() -> str:
        report = await attribute_reactions(store)
        return report.summary_line() if report.total else "no delivered steers to attribute"

    async def export_() -> str:
        try:
            report = await run_export(store, out=out, push_to=push_to)
        except Exception as error:  # noqa: BLE001 — a failed push must not lose the local export
            if push_to is None:
                raise
            report = await run_export(store, out=out, push_to=None)
            counts = "  ".join(f"{config} {sum(splits.values())}" for config, splits in report.counts.items())
            return f"{counts} (push to {push_to} FAILED: {type(error).__name__}, exported locally)"
        counts = "  ".join(f"{config} {sum(splits.values())}" for config, splits in report.counts.items())
        return counts + (f" pushed to {push_to}" if report.pushed else "")

    await stage("scan", scan_)
    await stage("triage", triage_)
    await stage("refine", refine_)
    await stage("enrich", enrich_)
    await stage("negatives", negatives_)
    if weekly:
        await stage("audit", audit_)
        await stage("eval", eval_)
    await stage("reactions", reactions_)
    await stage("export", export_)
    return PipelineReport(outcomes=tuple(outcomes))
