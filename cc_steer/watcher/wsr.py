"""Anytime-valid confidence sequences over the live fire-outcome stream.

Once the gate fires live and :mod:`cc_steer.watcher.outcomes` resolves whether the
user really steered near each scored moment, the running question is "how good is
the gate, and how sure are we yet." A fixed-n confidence interval can't answer that
under continuous monitoring — peeking inflates its error. This module implements the
Waudby-Smith--Ramdas predictable-mixture empirical-Bernstein confidence sequence
(*Estimating means of bounded random variables by betting*, JRSS-B 2023): a CI that
holds simultaneously at every sample size, so it may be read after every new outcome
without penalty.

The estimand is always the mean of a bounded ``[0, 1]`` stream — precision
``P(steered | fired)``, recall ``P(fired | steered)``, or the base steer rate
``P(steered)`` — so one CS construction covers them all. Everything here is a pure
function of the observation sequence; :func:`report` folds a fire-outcome pair stream
into the three headline sequences the CLI and dashboard surface.

The construction matches the reference ``confseq.predmix.predmix_empbern_twosided_cs``
bit-for-bit (verified numerically): the lower CS is a predictable-mixture martingale
with regularized empirical-Bernstein bets ``λ_t`` truncated at ``truncation``, and the
two-sided sequence splits ``α`` across a lower bound on ``x`` and an upper bound read
off the lower bound on ``1 - x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_ALPHA = 0.05
DEFAULT_TRUNCATION = 0.5
PRIOR_MEAN = 0.5
PRIOR_VARIANCE = 0.25


class Bounds(NamedTuple):
    """One time step of a two-sided confidence sequence: the closed interval ``[lower, upper]``."""

    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class WsrEstimate:
    """A confidence sequence read at its latest observation.

    ``point`` is the plain sample mean; ``[lower, upper]`` is the anytime-valid
    confidence sequence, which is *not* a Wald interval centered on the mean. With
    running intersection the bounds carry forward the tightest verdict from every
    earlier prefix, so ``point`` may sit outside ``[lower, upper]`` when a run of
    early observations pinned the mean more sharply than the latest ones — the
    bounds, not the point, are the trustworthy anytime-valid claim.

    Attributes:
        n: How many observations the estimate is built on.
        point: The running sample mean, or None when ``n`` is 0.
        lower: The anytime-valid lower confidence bound.
        upper: The anytime-valid upper confidence bound.
    """

    n: int
    point: float | None
    lower: float
    upper: float

    @property
    def width(self) -> float:
        """The interval width — how far the CS still is from a point answer."""
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class FireOutcome:
    """One scored live moment paired with its resolved outcome.

    Attributes:
        fired: Whether the gate cleared its threshold on this moment.
        steered: Whether the user actually steered on or near this moment.
    """

    fired: bool
    steered: bool


@dataclass(frozen=True, slots=True)
class WsrReport:
    """The gate's live quality, each headline as an anytime-valid confidence sequence.

    Attributes:
        total: Scored moments the report covers.
        fires: Moments the gate fired on.
        steers: Moments the user actually steered on.
        alpha: The two-sided error level every sequence holds at.
        precision: ``P(steered | fired)`` — of the fires, how many were real.
        recall: ``P(fired | steered)`` — of the real steers, how many the gate caught.
        steer_rate: ``P(steered)`` — the base rate a random scored moment is a steer.
        fire_rate: ``P(fired)`` — the base rate the gate fires.
    """

    total: int
    fires: int
    steers: int
    alpha: float
    precision: WsrEstimate
    recall: WsrEstimate
    steer_rate: WsrEstimate
    fire_rate: WsrEstimate

    def summary_line(self) -> str:
        return (
            f"wsr n={self.total} fires={self.fires} steers={self.steers} "
            f"precision={fmt(self.precision)} recall={fmt(self.recall)} "
            f"steer_rate={fmt(self.steer_rate)} fire_rate={fmt(self.fire_rate)} (α={self.alpha})"
        )


def fmt(estimate: WsrEstimate) -> str:
    """One estimate as ``point[lower, upper]``, or ``·`` when it rests on no data."""
    point = "·" if estimate.point is None else f"{estimate.point:.3f}"
    return f"{point}[{estimate.lower:.3f}, {estimate.upper:.3f}]"


def psi_e(lam: float) -> float:
    """The empirical-Bernstein variance penalty ``-ln(1 - λ) - λ`` for a bet ``λ ∈ [0, 1)``."""
    return -log(1 - lam) - lam


def eb_lambdas(xs: Sequence[float], *, alpha: float, truncation: float = DEFAULT_TRUNCATION) -> list[float]:
    """The predictable empirical-Bernstein bets ``λ_t``, each fixed before seeing ``x_t``.

    ``λ_t`` scales like ``1/sqrt(t log t · σ̂²_{t-1})`` off the regularized running
    variance through ``t-1`` (seeded at the ``(1/2, 1/4)`` prior), truncated at
    ``truncation`` so the martingale terms stay positive.
    """
    lambdas: list[float] = []
    running = squares = 0.0
    prev_sigma2 = PRIOR_VARIANCE
    for t, x in enumerate(xs, start=1):
        lambdas.append(min(truncation, sqrt(2 * log(1 / alpha) / (t * log(1 + t) * prev_sigma2))))
        running += x
        mu = min((PRIOR_MEAN + running) / (t + 1), 1.0)
        squares += (x - mu) ** 2
        prev_sigma2 = (PRIOR_VARIANCE + squares) / (t + 1)
    return lambdas


def lower_cs(
    xs: Sequence[float], *, alpha: float, truncation: float = DEFAULT_TRUNCATION, running_intersection: bool = True
) -> list[float]:
    """The predictable-mixture empirical-Bernstein lower confidence sequence on ``E[x]``.

    Returns one lower bound per time step; with ``running_intersection`` each bound is
    the tightest valid one seen so far, so the sequence never loosens.
    """
    lambdas = eb_lambdas(xs, alpha=alpha, truncation=truncation)
    log_term = log(1 / alpha)
    bounds: list[float] = []
    sum_lam = sum_lam_x = sum_v_psi = running = 0.0
    mu_prev = floor = 0.0
    for t, (x, lam) in enumerate(zip(xs, lambdas, strict=True), start=1):
        sum_lam += lam
        sum_lam_x += lam * x
        sum_v_psi += (x - mu_prev) ** 2 * psi_e(lam)
        floor = max(floor, raw := max((sum_lam_x - log_term - sum_v_psi) / sum_lam, 0.0))
        bounds.append(floor if running_intersection else raw)
        running += x
        mu_prev = running / t
    return bounds


def confidence_sequence(
    xs: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    truncation: float = DEFAULT_TRUNCATION,
    running_intersection: bool = True,
) -> list[Bounds]:
    """The two-sided empirical-Bernstein confidence sequence on ``E[x]`` for ``x ∈ [0, 1]``.

    The interval at every prefix length covers the true mean with probability at least
    ``1 - alpha`` simultaneously, so it is safe to read after each new observation.

    Args:
        xs: The bounded ``[0, 1]`` observation stream, in arrival order.
        alpha: The two-sided error level, split evenly across the bounds.
        truncation: The cap on each empirical-Bernstein bet.
        running_intersection: Tighten each interval with all earlier ones.

    Returns:
        One :class:`Bounds` per prefix of ``xs``; empty for an empty stream.
    """
    lower = lower_cs(xs, alpha=alpha / 2, truncation=truncation, running_intersection=running_intersection)
    upper = lower_cs(
        [1 - x for x in xs], alpha=alpha / 2, truncation=truncation, running_intersection=running_intersection
    )
    return [Bounds(lo, 1 - up) for lo, up in zip(lower, upper, strict=True)]


def estimate(xs: Sequence[float], *, alpha: float = DEFAULT_ALPHA) -> WsrEstimate:
    """The confidence sequence read at its final observation, plus the running mean.

    An empty stream yields the uninformative ``[0, 1]`` interval with a None point.
    """
    if not (seq := confidence_sequence(xs, alpha=alpha)):
        return WsrEstimate(n=0, point=None, lower=0.0, upper=1.0)
    lo, hi = seq[-1]
    return WsrEstimate(n=len(seq), point=sum(xs) / len(xs), lower=lo, upper=hi)


def bern(flags: Iterable[bool]) -> list[float]:
    """A boolean stream as its ``0.0``/``1.0`` indicator sequence."""
    return [float(flag) for flag in flags]


def report(pairs: Sequence[FireOutcome], *, alpha: float = DEFAULT_ALPHA) -> WsrReport:
    """Fold a chronological fire-outcome stream into the gate's headline confidence sequences.

    Args:
        pairs: Scored moments paired with resolved outcomes, oldest first — order is
            load-bearing, since a confidence sequence reads its stream in arrival order.
        alpha: The two-sided error level every sequence holds at.

    Returns:
        The :class:`WsrReport` with precision, recall, and the two base rates.
    """
    return WsrReport(
        total=len(pairs),
        fires=sum(pair.fired for pair in pairs),
        steers=sum(pair.steered for pair in pairs),
        alpha=alpha,
        precision=estimate(bern(pair.steered for pair in pairs if pair.fired), alpha=alpha),
        recall=estimate(bern(pair.fired for pair in pairs if pair.steered), alpha=alpha),
        steer_rate=estimate(bern(pair.steered for pair in pairs), alpha=alpha),
        fire_rate=estimate(bern(pair.fired for pair in pairs), alpha=alpha),
    )
