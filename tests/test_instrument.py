from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cc_steer.instrument import actionable, bootstrap_ci, delong_se, mde, paired_delong

FIXTURE = Path(__file__).parent / "fixtures" / "delong_e27.npz"

# E15 instrument-card cross-check: single-AUC DeLong SE of the v002 fire scores on the
# n=628 watcher frame (E15/out/bootstrap_summary.json ["delong_single_auc_se"]).
E15_DELONG_SE = 0.009981676031609831

# E27/out/fullframe_summary.json ["delong"] — the paired base-vs-steer record. E27's base
# arm reproduces the stored v002 probs exactly, so se_base == E15_DELONG_SE.
E27_SCALARS = {
    "auc_a": 0.9316896744944222,
    "auc_b": 0.9062859762620344,
    "delta": -0.02540369823238775,
    "se_a": 0.009981676031609831,
    "se_b": 0.011941544811239779,
    "cov": 0.00010306931535000547,
    "se_delta": 0.006007971219875388,
    "rho": 0.8646998992070197,
    "z": -4.228332211104475,
}
E27_CI95 = (-0.03717910544349669, -0.013628291021278816)
E27_MDE_PAIRED = 0.016831932169602888


@pytest.fixture(scope="module")
def vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(FIXTURE)
    return data["labels"], data["fire_base"], data["fire_steer"]


def test_fixture_shape(vectors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    labels, fire_base, fire_steer = vectors
    assert labels.shape == fire_base.shape == fire_steer.shape == (628,)
    assert int(labels.sum()) == 335
    assert int((~labels).sum()) == 293


def test_delong_se_reproduces_e15_constant(vectors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    labels, fire_base, _ = vectors
    assert abs(delong_se(labels, fire_base) - E15_DELONG_SE) < 1e-15


@pytest.mark.parametrize("field", list(E27_SCALARS), ids=list(E27_SCALARS))
def test_paired_delong_scalar_fields(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray], field: str
) -> None:
    labels, fire_base, fire_steer = vectors
    value = getattr(paired_delong(labels, fire_base, fire_steer), field)
    assert value == pytest.approx(E27_SCALARS[field], rel=1e-9, abs=1e-12)


def test_paired_delong_ci95(vectors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    labels, fire_base, fire_steer = vectors
    lo, hi = paired_delong(labels, fire_base, fire_steer).ci95
    assert lo == pytest.approx(E27_CI95[0], rel=1e-9, abs=1e-12)
    assert hi == pytest.approx(E27_CI95[1], rel=1e-9, abs=1e-12)


def test_mde_default_reproduces_card_constant() -> None:
    assert round(mde(1.0), 4) == 2.8016
    assert round(mde(E27_SCALARS["se_delta"]), 4) == round(E27_MDE_PAIRED, 4)


@pytest.mark.parametrize(
    "alpha, power, expected",
    [
        (0.05, 0.5, 1.9599639845400534),
        (0.01, 0.8, 3.4174505371218142),
        (0.10, 0.9, 2.9264051924960723),
    ],
    ids=["alpha05_power50", "alpha01_power80", "alpha10_power90"],
)
def test_mde_custom_alpha_power(alpha: float, power: float, expected: float) -> None:
    assert mde(1.0, alpha=alpha, power=power) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "delta, se_delta, frame_mde, expected",
    [
        (-0.02540369823238775, 0.006007971219875388, 0.016831932169602888, True),
        (0.05, 0.01, 0.03, True),
        (0.02, 0.005, 0.045, False),
        (0.05, 0.03, 0.04, False),
    ],
    ids=["e27_actionable", "significant_and_large", "below_mde", "ci_includes_zero"],
)
def test_actionable_card_rule(delta: float, se_delta: float, frame_mde: float, expected: bool) -> None:
    assert actionable(delta, se_delta, frame_mde) is expected


def test_bootstrap_ci_deterministic() -> None:
    scores = [0.55, 0.20, 0.60, 0.35, 0.70, 0.45, 0.52, 0.30, 0.65, 0.25, 0.48, 0.40, 0.75, 0.15, 0.58, 0.33]
    labels = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
    assert bootstrap_ci(scores, labels, iters=2000, seed=1729) == (0.5396825396825397, 1.0)
