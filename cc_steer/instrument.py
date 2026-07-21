"""Fast paired DeLong AUC statistics and the instrument-card decision rule.

The DeLong machinery is the Sun & Xu (2014) fast midrank algorithm, ported verbatim
from the E27 full-frame validation harness: ``compute_midrank`` and ``fast_delong``
carry the same tie-averaged midranks and covariance terms, so a single covariance
computation yields both classifiers' AUCs, their DeLong standard errors, and the
covariance-aware standard error of their difference.

On top of that sits the instrument card: :func:`mde` turns a standard error into a
minimum detectable effect and :func:`actionable` applies the card's two-part rule —
the paired 95% CI must exclude zero *and* the observed AUC delta must clear the
frame's MDE — that separates a real steering effect from the instrument's noise floor.
:class:`InstrumentCard` reads the card sidecar at :data:`INSTRUMENT_CARD`, and
:func:`paired_verdict` / :func:`unpaired_verdict` render the rule as a
:class:`Comparison` — the production verdict for any comparative checkpoint AUC:
paired with measured rho whenever both arms' per-row probs are persisted, the card's
unpaired frame floor when a counterpart vector is absent, and a sub-threshold delta
always reported as within the noise floor, never as a win or a loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

Z975 = 1.959963984540054
INSTRUMENT_CARD: Path = Path.home() / ".cc-steer" / "experiments" / "instrument-card-v1.json"


def compute_midrank(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = t + 1
    return out


def fast_delong(preds: npt.NDArray[np.float64], m: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = preds.shape[1] - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_cov(
    labels: npt.ArrayLike, *scores: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    y = np.asarray(labels).astype(int)
    order = np.argsort(-y, kind="stable")
    preds = np.vstack([np.asarray(s, dtype=np.float64) for s in scores])[:, order]
    return fast_delong(preds, int(y.sum()))


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


@dataclass(frozen=True, slots=True)
class PairedDeLong:
    """Result of a paired fast-DeLong comparison of two score vectors on shared labels.

    ``delta`` is ``auc_b - auc_a`` and ``se_delta`` is its covariance-aware DeLong
    standard error, ``sqrt(var_a + var_b - 2 * cov)``; ``ci95`` is the two-sided 95%
    confidence interval on ``delta``.

    Attributes:
        auc_a: DeLong AUC of the first score vector.
        auc_b: DeLong AUC of the second score vector.
        delta: ``auc_b - auc_a``.
        se_a: DeLong standard error of ``auc_a``.
        se_b: DeLong standard error of ``auc_b``.
        cov: Covariance of the two AUC estimates.
        se_delta: Covariance-aware standard error of ``delta``.
        rho: Correlation of the two AUC estimates (NaN when either variance is zero).
        z: ``delta / se_delta`` (NaN when ``se_delta`` is zero).
        ci95: Two-sided 95% confidence interval on ``delta``.
    """

    auc_a: float
    auc_b: float
    delta: float
    se_a: float
    se_b: float
    cov: float
    se_delta: float
    rho: float
    z: float
    ci95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class InstrumentCard:
    """The measurement-instrument card: the frame noise floors comparative verdicts consult.

    Read from the sidecar JSON at :data:`INSTRUMENT_CARD`
    (``~/.cc-steer/experiments/instrument-card-v1.json``) via :meth:`load`. The paired
    path measures its own MDE from the DeLong standard error, so the card matters
    exactly where pairing is impossible: ``mde`` is the unpaired floor of the n=628
    watcher frame — the fallback when a counterpart's per-row probs were never
    persisted — with the other frames' floors under ``stopping``.

    Attributes:
        mde: The unpaired minimum detectable effect of the watcher frame.
        stopping: The per-frame unpaired floors (``mde``, ``mde_gate_frame``, ``mde_golden``).
        mde_paired: The card's paired-MDE reference table with measured-rho examples.
    """

    mde: float
    stopping: Mapping[str, float]
    mde_paired: Mapping[str, object]

    @classmethod
    def load(cls, path: Path | None = None) -> InstrumentCard:
        """Read the card sidecar; ``path`` overrides :data:`INSTRUMENT_CARD`."""
        payload = json.loads((path or INSTRUMENT_CARD).read_text())
        return cls(
            mde=float(payload["mde"]),
            stopping={key: float(value) for key, value in payload["stopping"].items()},
            mde_paired=payload["mde_paired"],
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """A card-governed comparative verdict between two checkpoints' AUCs on one frame.

    ``verdict`` speaks the card's language: an actionable gain or regression names the
    delta, confidence interval, and MDE it cleared; a sub-threshold delta reads
    ``within noise floor (MDE <value>)`` and is never a win, a loss, or a rejection.

    Attributes:
        auc_a: AUC of the first arm (the incumbent or baseline).
        auc_b: AUC of the second arm (the candidate).
        delta: ``auc_b - auc_a``.
        mde: The minimum detectable effect the verdict applied — ``2.8016 * se_delta``
            on the paired path, the card's unpaired frame floor on the fallback.
        actionable: Whether the 95% CI excludes zero and ``|delta| >= mde``.
        verdict: The card-rule verdict line.
        paired: The full paired DeLong record, or ``None`` on the unpaired fallback.
    """

    auc_a: float
    auc_b: float
    delta: float
    mde: float
    actionable: bool
    verdict: str
    paired: PairedDeLong | None


def delong_se(labels: npt.ArrayLike, probs: npt.ArrayLike) -> float:
    """Return the DeLong standard error of a single AUC.

    Args:
        labels: Binary labels (higher score = positive class).
        probs: Per-row scores aligned with ``labels``.

    Returns:
        The DeLong standard error of the AUC of ``probs`` against ``labels``.
    """
    return float(np.sqrt(delong_cov(labels, probs)[1][0, 0]))


def paired_delong(labels: npt.ArrayLike, probs_a: npt.ArrayLike, probs_b: npt.ArrayLike) -> PairedDeLong:
    """Compare two score vectors on identical labels with a paired fast-DeLong test.

    Args:
        labels: Binary labels shared by both score vectors (higher score = positive).
        probs_a: Per-row scores of the first classifier.
        probs_b: Per-row scores of the second classifier.

    Returns:
        A :class:`PairedDeLong` with both AUCs, their delta, the per-AUC and
        covariance-aware standard errors, the pairing correlation, the z-statistic,
        and the 95% confidence interval on the delta.
    """
    aucs, cov = delong_cov(labels, probs_a, probs_b)
    var_a, var_b, cov_ab = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    se_delta = float(np.sqrt(max(var_a + var_b - 2 * cov_ab, 0.0)))
    delta = float(aucs[1] - aucs[0])
    return PairedDeLong(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        delta=delta,
        se_a=float(np.sqrt(var_a)),
        se_b=float(np.sqrt(var_b)),
        cov=cov_ab,
        se_delta=se_delta,
        rho=(cov_ab / denom) if (denom := float(np.sqrt(var_a * var_b))) > 0 else float("nan"),
        z=(delta / se_delta) if se_delta > 0 else float("nan"),
        ci95=(delta - Z975 * se_delta, delta + Z975 * se_delta),
    )


def bootstrap_ci(scores: list[float], labels: list[int], *, iters: int = 2000, seed: int = 1729) -> tuple[float, float]:
    """Nonparametric bootstrap 95% CI of the AUC (E27 harness resampling).

    Resamples row indices with replacement ``iters`` times, keeps resamples that
    retain both classes, and returns the 2.5th and 97.5th percentiles of the
    bootstrap AUC distribution.

    Args:
        scores: Per-row scores (higher = positive class).
        labels: Binary labels aligned with ``scores``.
        iters: Number of bootstrap resamples.
        seed: Seed for the resampling RNG.

    Returns:
        The ``(lo, hi)`` bounds of the bootstrap 95% confidence interval.
    """
    rng = Random(seed)
    idx = range(len(scores))
    vals = sorted(
        auc([scores[j] for j in s], [labels[j] for j in s])
        for s in ([rng.choice(idx) for _ in idx] for _ in range(iters))
        if len({labels[j] for j in s}) == 2
    )
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def mde(se: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for a two-sided z-test at the given standard error.

    Computes ``(z_{1-alpha/2} + z_{power}) * se`` with the normal quantiles from the
    standard library; the defaults (``alpha=0.05``, ``power=0.8``) reproduce the
    card's ``2.8016 * se`` constant to four decimals.

    Args:
        se: Standard error of the estimate under test.
        alpha: Two-sided significance level.
        power: Desired statistical power.

    Returns:
        The smallest effect the test can resolve at ``alpha`` and ``power``.
    """
    nd = NormalDist()
    return (nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) * se


def actionable(delta: float, se_delta: float, frame_mde: float) -> bool:
    """Apply the instrument card's two-part actionability rule to an AUC delta.

    The delta is actionable when its two-sided 95% confidence interval excludes zero
    *and* its magnitude clears the frame's minimum detectable effect.

    Args:
        delta: Observed AUC delta.
        se_delta: Covariance-aware standard error of ``delta``.
        frame_mde: The frame's minimum detectable effect.

    Returns:
        ``True`` when the delta is both significant and large enough to act on.
    """
    lo, hi = delta - Z975 * se_delta, delta + Z975 * se_delta
    return bool((lo > 0 or hi < 0) and abs(delta) >= frame_mde)


def paired_verdict(paired: PairedDeLong) -> Comparison:
    """The card rule over a paired DeLong record — the production comparative verdict.

    The MDE is measured, not assumed: ``2.8016`` times the record's covariance-aware
    ``se_delta``, which already carries the observed pairing correlation. The verdict
    line names the delta, CI, rho, and MDE when actionable, and reports a
    sub-threshold delta as within the noise floor — never a win or a loss.

    Args:
        paired: The paired comparison from :func:`paired_delong` over both arms'
            persisted per-row probs on the identical frame.

    Returns:
        The :class:`Comparison` carrying the applied MDE and the verdict line.
    """
    floor = mde(paired.se_delta)
    lo, hi = paired.ci95
    return Comparison(
        auc_a=paired.auc_a,
        auc_b=paired.auc_b,
        delta=paired.delta,
        mde=floor,
        actionable=(is_actionable := actionable(paired.delta, paired.se_delta, floor)),
        verdict=(
            f"actionable {'gain' if paired.delta > 0 else 'regression'} (delta {paired.delta:+.4f}, "
            f"95% CI [{lo:+.4f}, {hi:+.4f}], rho {paired.rho:.4f}, paired MDE {floor:.4f})"
            if is_actionable
            else f"within noise floor (MDE {floor:.4f})"
        ),
        paired=paired,
    )


def unpaired_verdict(auc_a: float, auc_b: float, *, frame_mde: float) -> Comparison:
    """The card rule when a counterpart's per-row probs were never persisted.

    With no pairing there is no measured rho, so the card's unpaired frame floor is
    the MDE. ``|delta| >= frame_mde`` already implies the unpaired 95% CI excludes
    zero (the MDE sits at ``2.8016`` standard errors, the CI bound at ``1.96``), so
    the card's two-part rule reduces to the MDE test.

    Args:
        auc_a: Point AUC of the first arm (the incumbent or baseline).
        auc_b: Point AUC of the second arm (the candidate).
        frame_mde: The frame's unpaired MDE from :class:`InstrumentCard`.

    Returns:
        The :class:`Comparison` with ``paired=None``.
    """
    delta = auc_b - auc_a
    return Comparison(
        auc_a=auc_a,
        auc_b=auc_b,
        delta=delta,
        mde=frame_mde,
        actionable=(is_actionable := abs(delta) >= frame_mde),
        verdict=(
            f"actionable {'gain' if delta > 0 else 'regression'} (delta {delta:+.4f}, unpaired MDE {frame_mde:.4f})"
            if is_actionable
            else f"within noise floor (MDE {frame_mde:.4f})"
        ),
        paired=None,
    )
