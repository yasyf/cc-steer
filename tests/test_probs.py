from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cc_steer import instrument
from cc_steer.retrain.data import DatasetDigest
from cc_steer.retrain.evalset import (
    PROJECTION_AUC,
    PROJECTION_RHO,
    TARGET_MDE,
    ArmComparison,
    EvalFrame,
    LabelRecord,
    compare_arms,
    comparison_path,
    hanley_mcneil_se,
    negatives_for_target_mde,
    plan_rebuild_frame,
    projected_frame_mde,
    resolve_labels,
    write_comparison,
)

if TYPE_CHECKING:
    from pathlib import Path

# labels 4/4; fire scores flipped from P(NO_STEER). One positive sits below a negative on each arm,
# so neither AUC is a degenerate 1.0 and the pairing variance is finite (rho is a real number).
LABELS = np.array([True, True, True, True, False, False, False, False], dtype=bool)
INCUMBENT_FIRE = np.array([0.90, 0.80, 0.70, 0.40, 0.60, 0.50, 0.30, 0.20])
CANDIDATE_FIRE = np.array([0.95, 0.85, 0.70, 0.55, 0.60, 0.45, 0.35, 0.10])


def frame(ids: tuple[str, ...], labels: np.ndarray) -> EvalFrame:
    n = len(ids)
    return EvalFrame(
        ids=ids,
        labels=labels,
        corrective=labels,
        prose=np.ones(n, dtype=bool),
        tails=tuple(f"tail {i}" for i in range(n)),
        digest=DatasetDigest("digest0000000000"),
    )


def eight_row_frame() -> EvalFrame:
    return frame(tuple(f"r{i}" for i in range(8)), LABELS)


class TestCompareArms:
    def test_auc_delta_and_rho_match_paired_delong(self) -> None:
        f = eight_row_frame()
        comparison = compare_arms(f, 1.0 - INCUMBENT_FIRE, 1.0 - CANDIDATE_FIRE, incumbent="v1", candidate="v2")
        auc_a = instrument.auc(INCUMBENT_FIRE.tolist(), [int(x) for x in LABELS])
        auc_b = instrument.auc(CANDIDATE_FIRE.tolist(), [int(x) for x in LABELS])
        assert comparison.paired.auc_a == pytest.approx(auc_a)
        assert comparison.paired.auc_b == pytest.approx(auc_b)
        assert comparison.paired.delta == pytest.approx(auc_b - auc_a)
        assert math.isfinite(comparison.paired.rho)  # finite-variance arms -> a real correlation
        assert comparison.frame_mde == pytest.approx(instrument.mde(comparison.paired.se_delta))

    def test_actionability_folds_ci_and_mde(self) -> None:
        f = eight_row_frame()
        comparison = compare_arms(f, 1.0 - INCUMBENT_FIRE, 1.0 - CANDIDATE_FIRE, incumbent="v1", candidate="v2")
        expected = instrument.actionable(comparison.paired.delta, comparison.paired.se_delta, comparison.frame_mde)
        assert comparison.is_actionable is expected

    def test_metrics_keys_are_flat_and_prefixed(self) -> None:
        comparison = ArmComparison(
            incumbent="v1",
            candidate="v2",
            paired=instrument.paired_delong(LABELS.astype(int), 1.0 - INCUMBENT_FIRE, 1.0 - CANDIDATE_FIRE),
            frame_mde=0.03,
            is_actionable=False,
        )
        metrics = comparison.as_metrics()
        assert set(metrics) == {
            "paired_incumbent_auc",
            "paired_candidate_auc",
            "paired_delta_auc",
            "paired_se_delta",
            "paired_rho",
            "paired_ci_lo",
            "paired_ci_hi",
            "paired_frame_mde",
            "paired_actionable",
        }
        assert metrics["paired_actionable"] == 0.0


class TestWriteComparison:
    def test_persists_both_arms_per_row(self, tmp_path: Path) -> None:
        f = eight_row_frame()
        comparison = compare_arms(f, 1.0 - INCUMBENT_FIRE, 1.0 - CANDIDATE_FIRE, incumbent="v1", candidate="v2")
        path = write_comparison(f, comparison, 1.0 - INCUMBENT_FIRE, 1.0 - CANDIDATE_FIRE, root=tmp_path)
        assert path == comparison_path("v1", "v2", root=tmp_path)
        payload = json.loads(path.read_text())
        assert payload["meta"] == {"dataset_digest": f.digest, "incumbent": "v1", "candidate": "v2"}
        assert payload["probs"]["r0"] == {
            "incumbent": pytest.approx(1.0 - INCUMBENT_FIRE[0]),
            "candidate": pytest.approx(1.0 - CANDIDATE_FIRE[0]),
        }
        assert set(payload["probs"]) == set(f.ids)  # both arms cover every frame row
        assert payload["paired"]["paired_delta_auc"] == pytest.approx(comparison.paired.delta)


class TestResolveLabels:
    def test_human_beats_fable_beats_medium_and_records_guidance(self) -> None:
        records = [
            LabelRecord(id="a", is_steering=True, category="wrong_approach", provenance="fable"),
            LabelRecord(id="a", is_steering=True, category="wrong_approach", provenance="human"),
            LabelRecord(id="a", is_steering=False, category="", provenance="medium_judge"),
        ]
        [resolved], dropped = resolve_labels(records)
        assert dropped == []
        assert (resolved.provenance, resolved.is_steering) == ("human", True)
        assert resolved.guidance is False  # the medium judge disagreed
        assert resolved.agrees_with_guidance is False

    def test_medium_only_is_dropped_never_labels_alone(self) -> None:
        records = [LabelRecord(id="m", is_steering=True, category="direction", provenance="medium_judge")]
        resolved, dropped = resolve_labels(records)
        assert resolved == []
        assert dropped == ["m"]

    def test_fable_without_guidance_has_none_agreement(self) -> None:
        records = [LabelRecord(id="f", is_steering=False, category="", provenance="fable")]
        [resolved], dropped = resolve_labels(records)
        assert (resolved.provenance, resolved.guidance, resolved.agrees_with_guidance) == ("fable", None, None)


class TestProjectedMde:
    def test_hanley_mcneil_matches_pinned_value(self) -> None:
        # A(1-A) + (np-1)(Q1-A^2) + (nn-1)(Q2-A^2) all over np*nn, sqrt'd; pinned at (0.93, 335, 293).
        assert hanley_mcneil_se(0.93, 335, 293) == pytest.approx(0.0104153, abs=1e-6)

    def test_projected_paired_mde_is_pinned_and_under_target(self) -> None:
        projected = projected_frame_mde(335, 293, auc=PROJECTION_AUC, rho=PROJECTION_RHO)
        assert projected == pytest.approx(0.018454, abs=1e-4)
        assert projected < TARGET_MDE  # the 335/293 frame already clears the 0.02 bar

    def test_mde_falls_monotonically_as_negatives_grow(self) -> None:
        assert projected_frame_mde(335, 50) > projected_frame_mde(335, 500) > projected_frame_mde(335, 5000)

    def test_degenerate_counts_are_infinite(self) -> None:
        assert hanley_mcneil_se(0.93, 0, 100) == float("inf")
        assert projected_frame_mde(0, 100) == float("inf")

    def test_negatives_for_target_reaches_the_bar(self) -> None:
        needed = negatives_for_target_mde(335, target_mde=TARGET_MDE)
        assert projected_frame_mde(335, needed) <= TARGET_MDE
        assert projected_frame_mde(335, needed - 1) > TARGET_MDE  # it is the fewest that suffice


class TestPlanRebuildFrame:
    def test_negative_rich_sizing_with_provenance_and_guidance(self) -> None:
        labels = [
            *(
                LabelRecord(id=f"p{i}", is_steering=True, category="wrong_approach", provenance="fable")
                for i in range(4)
            ),
            LabelRecord(id="p0", is_steering=True, category="wrong_approach", provenance="medium_judge"),  # agrees
            LabelRecord(id="p1", is_steering=False, category="", provenance="medium_judge"),  # disagrees, guidance only
            LabelRecord(id="mo", is_steering=True, category="direction", provenance="medium_judge"),  # dropped
            LabelRecord(id="n0", is_steering=False, category="", provenance="human"),  # an authoritative negative
        ]
        pool = [f"rand{i}" for i in range(50)]
        plan = plan_rebuild_frame(labels, pool, seed=1729)
        assert plan.n_pos == 4
        assert plan.medium_only_dropped == 1
        assert plan.n_neg >= plan.n_pos  # negative-rich: at least parity
        assert plan.n_neg == len(plan.negative_ids)
        assert "n0" in plan.negative_ids  # the authoritative negative joins the pool
        assert plan.provenance_counts["fable"] == 4
        assert plan.provenance_counts["human"] == 1  # the one authoritative negative admitted
        assert plan.provenance_counts["structural"] == plan.n_neg - 1
        assert (plan.guidance_agree, plan.guidance_disagree) == (1, 1)  # p0 agrees, p1 disagrees
        assert set(plan.ids) == set(plan.positive_ids) | set(plan.negative_ids)
        assert math.isfinite(plan.projected_mde)

    def test_authoritative_negatives_admitted_before_structural(self) -> None:
        labels = [
            *(LabelRecord(id=f"p{i}", is_steering=True, category="x", provenance="fable") for i in range(4)),
            *(LabelRecord(id=f"hn{i}", is_steering=False, category="", provenance="human") for i in range(3)),
        ]
        plan = plan_rebuild_frame(labels, [f"rand{i}" for i in range(50)], target_mde=0.5, seed=1729)
        assert plan.n_neg == 4  # the ratio floor, well under the 53-row pool
        assert {"hn0", "hn1", "hn2"} <= set(plan.negative_ids)  # every authoritative negative admitted first
        assert plan.provenance_counts["human"] == 3
        assert plan.provenance_counts["structural"] == 1  # only the remainder comes from the pool

    def test_truncated_draw_stays_within_authoritative(self) -> None:
        labels = [
            *(LabelRecord(id=f"p{i}", is_steering=True, category="x", provenance="fable") for i in range(4)),
            *(LabelRecord(id=f"hn{i}", is_steering=False, category="", provenance="human") for i in range(6)),
        ]
        plan = plan_rebuild_frame(labels, [], target_mde=0.5, seed=1729)
        assert plan.n_neg == 4
        assert plan.negative_ids == ("hn0", "hn1", "hn2", "hn5")  # sampled within the authoritative stratum
        assert "structural" not in plan.provenance_counts

    def test_deterministic_in_seed(self) -> None:
        labels = [LabelRecord(id=f"p{i}", is_steering=True, category="x", provenance="fable") for i in range(3)]
        pool = [f"rand{i}" for i in range(40)]
        assert plan_rebuild_frame(labels, pool, seed=7).ids == plan_rebuild_frame(labels, pool, seed=7).ids

    def test_meets_target_at_realistic_scale(self) -> None:
        labels = [LabelRecord(id=f"p{i}", is_steering=True, category="x", provenance="fable") for i in range(335)]
        pool = [f"rand{i}" for i in range(4000)]
        plan = plan_rebuild_frame(labels, pool, seed=1729)
        assert plan.meets_target
        assert plan.projected_mde <= TARGET_MDE
        assert plan.n_neg >= plan.n_pos
