from __future__ import annotations

import math

import pytest

from cc_steer.watcher.wsr import (
    Bounds,
    FireOutcome,
    WsrEstimate,
    bern,
    confidence_sequence,
    eb_lambdas,
    estimate,
    lower_cs,
    psi_e,
    report,
)

# --- the empirical-Bernstein primitives ------------------------------------


def test_psi_e_matches_the_closed_form() -> None:
    assert psi_e(0.5) == pytest.approx(-math.log(0.5) - 0.5)
    assert psi_e(0.5) == pytest.approx(0.1931471805599453)
    assert psi_e(0.0) == 0.0


def test_eb_lambdas_clamp_at_truncation_for_short_streams() -> None:
    # sqrt(2 ln(1/alpha) / (1 * ln2 * 1/4)) >> 1/2, so every early bet hits the cap.
    assert eb_lambdas([1.0, 0.0, 1.0], alpha=0.1) == [0.5, 0.5, 0.5]
    assert eb_lambdas([1.0, 1.0], alpha=0.05, truncation=0.1) == [0.1, 0.1]


def test_eb_first_lambda_is_the_hand_value() -> None:
    [lam] = eb_lambdas([1.0], alpha=0.1, truncation=10.0)
    assert lam == pytest.approx(math.sqrt(2 * math.log(1 / 0.1) / (1 * math.log(2) * 0.25)))


# --- the lower confidence sequence, hand-derived ---------------------------


def test_lower_cs_all_ones_is_the_closed_form() -> None:
    # All bets clamp to 1/2, so at t=10: sum_lambda = 5, weighted mean = 1, and only the
    # first term contributes variance (mu_0 = 0): margin = (ln(1/alpha) + psi_E(1/2)) / 5.
    [*_, last] = lower_cs([1.0] * 10, alpha=0.05)
    assert last == pytest.approx(1.0 - (math.log(1 / 0.05) + psi_e(0.5)) / 5.0)
    assert last == pytest.approx(0.3622241092, abs=1e-9)


def test_lower_cs_all_zeros_is_pinned_to_zero() -> None:
    assert lower_cs([0.0] * 10, alpha=0.05) == [0.0] * 10


def test_lower_cs_running_intersection_never_loosens() -> None:
    seq = lower_cs([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], alpha=0.05)
    assert seq == sorted(seq)


# --- the two-sided confidence sequence -------------------------------------


def test_confidence_sequence_all_ones_pins_both_bounds() -> None:
    # Two-sided splits alpha: the lower bound runs at alpha/2 = 0.05.
    cs = confidence_sequence([1.0] * 10, alpha=0.1)
    assert cs[-1] == Bounds(pytest.approx(0.3622241092, abs=1e-9), pytest.approx(1.0))


def test_confidence_sequence_all_zeros_is_the_mirror() -> None:
    cs = confidence_sequence([0.0] * 10, alpha=0.1)
    assert cs[-1] == Bounds(pytest.approx(0.0), pytest.approx(1.0 - 0.3622241092, abs=1e-9))


def test_confidence_sequence_twenty_ones_is_pinned() -> None:
    cs = confidence_sequence([1.0] * 20, alpha=0.05)
    assert cs[-1].lower == pytest.approx(0.6117973365, abs=1e-9)
    assert cs[-1].upper == pytest.approx(1.0)


def test_confidence_sequence_brackets_the_true_mean() -> None:
    # A balanced stream keeps 1/2 strictly inside every interval it produces.
    cs = confidence_sequence([1.0, 0.0] * 40, alpha=0.05)
    assert all(lo <= 0.5 <= hi for lo, hi in cs)
    assert cs[-1].lower > 0.0 and cs[-1].upper < 1.0


def test_confidence_sequence_is_empty_for_no_data() -> None:
    assert confidence_sequence([], alpha=0.05) == []


# --- the estimate wrapper ---------------------------------------------------


def test_estimate_reads_the_final_step_and_running_mean() -> None:
    est = estimate([1.0] * 10, alpha=0.1)
    assert est.n == 10
    assert est.point == pytest.approx(1.0)
    assert est.lower == pytest.approx(0.3622241092, abs=1e-9)
    assert est.width == pytest.approx(est.upper - est.lower)


def test_estimate_of_nothing_is_maximally_uncertain() -> None:
    assert estimate([], alpha=0.05) == WsrEstimate(n=0, point=None, lower=0.0, upper=1.0)


# --- the fire-outcome report ------------------------------------------------

FIRE_HIT = FireOutcome(fired=True, steered=True)
FIRE_MISS = FireOutcome(fired=True, steered=False)
QUIET_STEER = FireOutcome(fired=False, steered=True)
QUIET_CALM = FireOutcome(fired=False, steered=False)


def test_bern_maps_flags_to_indicators() -> None:
    assert bern([True, False, True]) == [1.0, 0.0, 1.0]


def test_report_counts_and_conditions_each_stream() -> None:
    pairs = [FIRE_HIT, FIRE_MISS, QUIET_STEER, QUIET_CALM]
    rep = report(pairs, alpha=0.05)
    assert (rep.total, rep.fires, rep.steers) == (4, 2, 2)
    # precision is P(steered | fired): 1 of 2 fires was a real steer.
    assert rep.precision.n == 2 and rep.precision.point == pytest.approx(0.5)
    # recall is P(fired | steered): 1 of 2 real steers was fired on.
    assert rep.recall.n == 2 and rep.recall.point == pytest.approx(0.5)
    # base rates run over every scored moment.
    assert rep.steer_rate.n == 4 and rep.steer_rate.point == pytest.approx(0.5)
    assert rep.fire_rate.n == 4 and rep.fire_rate.point == pytest.approx(0.5)


def test_report_precision_of_a_gate_that_never_fires_rests_on_no_data() -> None:
    rep = report([QUIET_STEER, QUIET_CALM, QUIET_STEER], alpha=0.05)
    assert rep.fires == 0
    assert rep.precision == WsrEstimate(n=0, point=None, lower=0.0, upper=1.0)
    assert rep.recall.n == 2 and rep.recall.point == pytest.approx(0.0)


def test_report_summary_line_is_scannable() -> None:
    line = report([FIRE_HIT, FIRE_MISS], alpha=0.05).summary_line()
    assert "precision=" in line and "recall=" in line and "n=2" in line
