"""The promotion gate: free-metric bars over the frozen eval, and the retrain journal.

A candidate is promoted only when it beats the incumbent on metrics that cost nothing
to compute — no frontier judging. The watcher bar (:func:`watcher_promotable`) reads
the corrected paired gate (:func:`corrected_gate`): a strict sentinel-AUC beat, the
fire budget held at matched fires, and coverage wins at least matching losses on the
warranted prose-corrective rows. The gate bar (:func:`gate_promotable`) is the
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

from cc_steer.journal import Journal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cc_steer import registry

STATE_DIR: Path = Path.home() / ".cc-steer"
JOURNAL_NAME = "journal.jsonl"
RETRAIN_LOG_TITLE = "cc-steer retrain journal"
RETRAIN_LOG_LABEL = "retrain"
PR_AUC_KEY = "pr_auc"
RECALL_KEY = "recall_at_2per100_viewratio_proxy"


@dataclass(frozen=True, slots=True)
class Verdict:
    """One promotion decision: whether to promote and the human-readable reason."""

    promote: bool
    reason: str


@dataclass(frozen=True, slots=True)
class GateResult:
    """The corrected paired gate over one (candidate, incumbent) pair at matched budget.

    ``harmful_favors_incumbent`` is ``None`` until harmful-fire warrant judging lands,
    and ``promote`` stays ``None`` while it is pending, so the free components preview
    without a verdict.

    Attributes:
        candidate: The candidate's name.
        incumbent: The incumbent's name.
        coverage_wins: Warranted prose-corrective rows the candidate fires and the incumbent abstains on.
        coverage_losses: Warranted prose-corrective rows the incumbent fires and the candidate abstains on.
        coverage_sign_p: Exact sign-test p over the discordant coverage pairs.
        coverage_sig: The coverage win is significant (``p < 0.05`` and wins > losses).
        budget_held: The candidate's fire count does not exceed the incumbent's.
        cell_auc: The candidate's sentinel AUC.
        incumbent_auc: The incumbent's sentinel AUC.
        auc_not_regressed: The candidate's AUC is at least the incumbent's.
        harmful_favors_incumbent: Whether harmful-fire warrant favours the incumbent, or None while pending.
        promote: The full verdict once harmful judging lands, or None while pending.
    """

    candidate: str
    incumbent: str
    coverage_wins: int
    coverage_losses: int
    coverage_sign_p: float
    coverage_sig: bool
    budget_held: bool
    cell_auc: float
    incumbent_auc: float
    auc_not_regressed: bool
    harmful_favors_incumbent: bool | None
    promote: bool | None


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test over discordant pairs (exact McNemar); 1.0 with no discordant pairs."""
    from scipy.stats import binomtest

    if wins < 0 or losses < 0:
        raise ValueError(f"wins/losses must be >= 0, got {wins}/{losses}")
    if (n := wins + losses) == 0:
        return 1.0
    return float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)


def threshold_for_budget(scores: np.ndarray, *, fires_per_100: float, total_turns: int) -> float:
    """Score threshold whose exceedance count (``score >= threshold``) matches the alert budget.

    Quantile-based and conservative: fires at most ``floor(fires_per_100 * total_turns / 100)``
    times on ``scores``, as many as possible within that budget. Ties that would blow the
    budget push the threshold above the tied value.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if total_turns <= 0:
        raise ValueError(f"total_turns must be > 0, got {total_turns}")
    if fires_per_100 < 0:
        raise ValueError(f"fires_per_100 must be >= 0, got {fires_per_100}")
    budget = int(np.floor(fires_per_100 * total_turns / 100.0))
    scores_sorted = np.sort(scores)
    values = np.unique(scores_sorted)
    count_ge = scores.size - np.searchsorted(scores_sorted, values, side="left")
    within = values[count_ge <= budget]
    if within.size:
        return float(within[0])
    return float(np.nextafter(values[-1], np.inf))


def sentinel_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Sentinel AUC: ROC-AUC of the fire score ``1 - P(NO_STEER)`` against the fire labels."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels.tolist(), (1.0 - probs).tolist()))


def matched_fire_mask(probs: np.ndarray, *, budget_fires: int) -> np.ndarray:
    """Fire mask (``P(NO_STEER) < t``) matched to a fire budget via :func:`threshold_for_budget` on ``1 - probs``."""
    n = probs.size
    threshold = threshold_for_budget(1.0 - probs, fires_per_100=100.0 * budget_fires / n, total_turns=n)
    return (1.0 - probs) >= threshold


def corrected_gate(
    candidate_probs: np.ndarray,
    incumbent_probs: np.ndarray,
    *,
    candidate: str,
    incumbent: str,
    incumbent_threshold: float,
    labels: np.ndarray,
    corrective: np.ndarray,
    prose: np.ndarray,
    harmful_favors_incumbent: bool | None = None,
) -> GateResult:
    """The corrected paired gate, paired on the common eval rows at matched budget.

    Coverage sign-test over warranted prose-corrective rows, budget held, same-render
    AUC not regressed, and harmful-fire not favouring the incumbent. The harmful-fire
    term is judged and passed in from the deferred gate pass; absent it, ``promote``
    stays ``None`` (free preview only). Never favours a candidate on degenerate stats:
    all three free terms must be affirmative.
    """
    incumbent_fires = incumbent_probs < incumbent_threshold
    candidate_fires = matched_fire_mask(candidate_probs, budget_fires=int(incumbent_fires.sum()))
    warranted = corrective & prose
    wins = int((candidate_fires & ~incumbent_fires & warranted).sum())
    losses = int((incumbent_fires & ~candidate_fires & warranted).sum())
    coverage_p = sign_test_p(wins, losses)
    coverage_sig = coverage_p < 0.05 and wins > losses
    budget_held = int(candidate_fires.sum()) <= int(incumbent_fires.sum())
    cell_auc = sentinel_auc(labels, candidate_probs)
    incumbent_auc = sentinel_auc(labels, incumbent_probs)
    auc_not_regressed = cell_auc >= incumbent_auc
    free_pass = coverage_sig and budget_held and auc_not_regressed
    promote = None if harmful_favors_incumbent is None else (free_pass and not harmful_favors_incumbent)
    return GateResult(
        candidate=candidate,
        incumbent=incumbent,
        coverage_wins=wins,
        coverage_losses=losses,
        coverage_sign_p=coverage_p,
        coverage_sig=coverage_sig,
        budget_held=budget_held,
        cell_auc=cell_auc,
        incumbent_auc=incumbent_auc,
        auc_not_regressed=auc_not_regressed,
        harmful_favors_incumbent=harmful_favors_incumbent,
        promote=promote,
    )


def watcher_promotable(result: GateResult) -> Verdict:
    """The free-metric watcher bar: strict AUC beat, budget held, coverage wins >= losses.

    Fails closed on a non-finite AUC: a NaN on either side is a degenerate score, never a beat.
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
    if result.cell_auc <= result.incumbent_auc:
        return Verdict(False, f"candidate AUC {result.cell_auc:.4f} <= incumbent {result.incumbent_auc:.4f}")
    if not result.budget_held:
        return Verdict(False, "fire budget exceeded at matched fires")
    if result.coverage_wins < result.coverage_losses:
        return Verdict(False, f"coverage losses {result.coverage_losses} > wins {result.coverage_wins}")
    return Verdict(
        True,
        f"candidate AUC {result.cell_auc:.4f} > incumbent {result.incumbent_auc:.4f}, budget held, "
        f"coverage {result.coverage_wins} >= {result.coverage_losses}",
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
        "metrics": dict(metrics or {}),
        "version": version,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    line = f"{component}: {verdict}"
    Journal(Path.cwd(), title=RETRAIN_LOG_TITLE, label=RETRAIN_LOG_LABEL).append(line)
    return line
