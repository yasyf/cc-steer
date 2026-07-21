"""The promotion gate: free-metric bars over the frozen eval, and the retrain journal.

A candidate is promoted only when it beats the incumbent on metrics that cost nothing
to compute — no frontier judging. The watcher bar (:func:`watcher_promotable`) reads
the corrected paired gate (:func:`corrected_gate`) under the instrument card's paired
DeLong rule (:func:`cc_steer.instrument.paired_verdict`): no actionable sentinel-AUC
regression, the fire budget held at matched fires, and coverage wins at least matching
losses on the warranted prose-corrective rows. The gate bar (:func:`gate_promotable`) is the
lexical gate's rule: beat the incumbent's PR-AUC without regressing recall at the
alert budget. :func:`should_retrain` decides whether a pass trains at all — forced,
no incumbent, or the training data moved.

Every pass — skip, reject, promote — appends one line through :func:`journal`, the
single writer, to ``<state_dir>/retrain/journal.jsonl`` and returns the one-line
verdict for stdout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from athome.train.gate import (
    GateResult,
    corrected_gate,
    matched_fire_mask,
    sentinel_auc,
    sign_test_p,
    threshold_for_budget,
)

from cc_steer.journal import Journal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cc_steer import registry
    from cc_steer.instrument import Comparison

STATE_DIR: Path = Path.home() / ".cc-steer"
JOURNAL_NAME = "journal.jsonl"
RETRAIN_LOG_TITLE = "cc-steer retrain journal"
RETRAIN_LOG_LABEL = "retrain"
PR_AUC_KEY = "pr_auc"
RECALL_KEY = "recall_at_2per100_viewratio_proxy"

# Gate statistics live in athome.train.gate (higher-is-fire); re-exported so callers keep the
# promotion import path and orient their no-fire probabilities (fire = 1 - prob) at the call site.
__all__ = [
    "PR_AUC_KEY",
    "RECALL_KEY",
    "RETRAIN_LOG_LABEL",
    "RETRAIN_LOG_TITLE",
    "GateResult",
    "Verdict",
    "corrected_gate",
    "gate_promotable",
    "journal",
    "matched_fire_mask",
    "sentinel_auc",
    "should_retrain",
    "sign_test_p",
    "threshold_for_budget",
    "watcher_promotable",
]


@dataclass(frozen=True, slots=True)
class Verdict:
    """One promotion decision: whether to promote and the human-readable reason."""

    promote: bool
    reason: str


def watcher_promotable(result: GateResult, comparison: Comparison) -> Verdict:
    """The watcher bar: the judged fold when it exists, else the free-metric bar, AUC term card-governed.

    ``comparison`` is the instrument-card verdict (:func:`cc_steer.instrument.paired_verdict`) over
    the same two fire-score vectors the gate compared. Its two-part rule replaces the point AUC beat
    everywhere: only an actionable regression refuses on AUC, and a sub-threshold delta is instrument
    noise — reported as within the noise floor, never a win, a loss, or a rejection. Once the
    harmful-fire judging has run (``result.harmful_favors_incumbent`` is not ``None``) the verdict
    folds significant coverage, the fire budget at matched fires, and no actionable AUC regression
    with the judged term — recomposed here from the gate's components because
    :attr:`GateResult.promote` bakes in the point AUC comparison this rule supersedes. While judging
    is pending it falls back to the free-metric bar: no actionable AUC regression, the fire budget
    held, and coverage wins at least matching losses. Fails closed on a non-finite AUC: a NaN on
    either side is a degenerate score, never a beat.
    """
    if not all(
        np.isfinite(value)
        for value in (result.cell_auc, result.incumbent_auc, result.coverage_wins, result.coverage_losses)
    ):
        return Verdict(
            False,
            f"non-finite metric: candidate AUC {result.cell_auc}, incumbent AUC {result.incumbent_auc}, "
            f"coverage {result.coverage_wins}/{result.coverage_losses}",
        )
    regressed = comparison.actionable and comparison.delta < 0
    auc_line = f"AUC {comparison.auc_b:.4f} vs incumbent {comparison.auc_a:.4f}: {comparison.verdict}"
    if result.harmful_favors_incumbent is not None:
        promote = (
            result.coverage_sig and result.budget_held and not regressed and not result.harmful_favors_incumbent
        )
        return Verdict(
            promote,
            f"judged gate {'promotes' if promote else 'refuses'}: "
            f"harmful_favors_incumbent={result.harmful_favors_incumbent}, coverage "
            f"{result.coverage_wins}/{result.coverage_losses} (sig {result.coverage_sig}), "
            f"budget {'held' if result.budget_held else 'exceeded'}, {auc_line}",
        )
    if regressed:
        return Verdict(False, auc_line)
    if not result.budget_held:
        return Verdict(False, "fire budget exceeded at matched fires")
    if result.coverage_wins < result.coverage_losses:
        return Verdict(False, f"coverage losses {result.coverage_losses} > wins {result.coverage_wins}")
    return Verdict(
        True, f"{auc_line}; budget held, coverage {result.coverage_wins} >= {result.coverage_losses}"
    )


def gate_promotable(candidate: Mapping[str, float], incumbent: Mapping[str, float] | None) -> Verdict:
    """The lexical gate bar: beat the incumbent's PR-AUC without regressing recall at the alert budget.

    Fails closed on a non-finite metric, and reads the incumbent's metrics by direct index so a
    missing key raises loud (a corrupt incumbent record must never silently lose to any candidate).
    """
    if incumbent is None:
        return Verdict(True, "no incumbent")
    metrics = (candidate[PR_AUC_KEY], candidate[RECALL_KEY], incumbent[PR_AUC_KEY], incumbent[RECALL_KEY])
    if not all(np.isfinite(metric) for metric in metrics):
        return Verdict(False, f"non-finite metric among candidate/incumbent pr_auc & recall: {metrics}")
    if candidate[PR_AUC_KEY] <= incumbent[PR_AUC_KEY]:
        return Verdict(False, f"pr_auc {candidate[PR_AUC_KEY]:.4f} <= incumbent {incumbent[PR_AUC_KEY]:.4f}")
    if candidate[RECALL_KEY] < incumbent[RECALL_KEY]:
        return Verdict(False, f"recall {candidate[RECALL_KEY]:.4f} < incumbent {incumbent[RECALL_KEY]:.4f}")
    return Verdict(True, f"pr_auc {candidate[PR_AUC_KEY]:.4f} > incumbent {incumbent[PR_AUC_KEY]:.4f}, recall held")


def should_retrain(incumbent: registry.VersionInfo | None, digest: str, *, force: bool) -> bool:
    """Whether the pass trains at all: forced, no incumbent, or the train data moved."""
    return force or incumbent is None or str(incumbent.metadata.get("dataset_digest")) != digest


def journal(
    component: str,
    verdict: str,
    *,
    dataset_digest: str,
    hf_revision: str | None = None,
    metrics: Mapping[str, float] | None = None,
    version: str | None = None,
    state_dir: Path | None = None,
) -> str:
    """Append one retrain-journal line and return the ``<component>: <verdict>`` stdout line.

    The single writer for ``<state_dir>/retrain/journal.jsonl``; ``state_dir`` defaults
    to ``~/.cc-steer``. The same line is mirrored to the ``cc-steer retrain journal``
    cc-notes log in the current repo, degrading silently when cc-notes is missing or the
    repo is uninitialized — the JSONL stays authoritative regardless.
    """
    path = (state_dir or STATE_DIR) / "retrain" / JOURNAL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "component": component,
        "verdict": verdict,
        "dataset_digest": dataset_digest,
        **({"hf_revision": hf_revision} if hf_revision is not None else {}),
        "metrics": dict(metrics or {}),
        "version": version,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    line = f"{component}: {verdict}"
    Journal(Path.cwd(), title=RETRAIN_LOG_TITLE, label=RETRAIN_LOG_LABEL).append(line)
    return line
