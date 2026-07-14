from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from cc_steer import registry
from cc_steer.retrain.promotion import (
    PR_AUC_KEY,
    RECALL_KEY,
    GateResult,
    Verdict,
    corrected_gate,
    gate_promotable,
    journal,
    matched_fire_mask,
    should_retrain,
    sign_test_p,
    threshold_for_budget,
    watcher_promotable,
)

GOOD = {PR_AUC_KEY: 0.94, RECALL_KEY: 0.30}


def mask(indices: list[int], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[indices] = True
    return out


def gate_result(**overrides: object) -> GateResult:
    fields: dict[str, object] = {
        "candidate": "cand",
        "incumbent": "inc",
        "coverage_wins": 5,
        "coverage_losses": 0,
        "coverage_sign_p": 0.03,
        "coverage_sig": True,
        "budget_held": True,
        "cell_auc": 0.80,
        "incumbent_auc": 0.70,
        "auc_not_regressed": True,
        "harmful_favors_incumbent": None,
        "promote": None,
    }
    return GateResult(**(fields | overrides))  # type: ignore[arg-type]


def version_info(digest: str = "d1") -> registry.VersionInfo:
    return registry.VersionInfo(
        component="watcher", version="v001-20260101-abcdef123456", path=Path("/x"), metadata={"dataset_digest": digest}
    )


class TestMatchedFireMask:
    def test_fires_lowest_prob_within_budget(self) -> None:
        probs = np.array([0.01, 0.02, 0.03, 0.9, 0.8], dtype=np.float64)
        assert matched_fire_mask(probs, budget_fires=2).tolist() == [True, True, False, False, False]

    def test_zero_budget_fires_nothing(self) -> None:
        assert matched_fire_mask(np.array([0.01, 0.5, 0.9]), budget_fires=0).sum() == 0


class TestThresholdForBudget:
    def test_matches_exceedance_count(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95], dtype=np.float64)
        assert threshold_for_budget(scores, fires_per_100=40, total_turns=5) == 0.9

    def test_zero_budget_pushes_above_max(self) -> None:
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        assert threshold_for_budget(scores, fires_per_100=0, total_turns=3) > 0.9


class TestSignTestP:
    def test_no_discordant_pairs_is_one(self) -> None:
        assert sign_test_p(0, 0) == 1.0

    def test_symmetric_split_is_one(self) -> None:
        assert sign_test_p(3, 3) == 1.0

    def test_lopsided_split_is_significant(self) -> None:
        assert sign_test_p(7, 0) < 0.05


def gate_frame() -> dict[str, np.ndarray]:
    n = 20
    cell_probs = np.full(n, 0.9)
    cell_probs[:7] = np.linspace(0.01, 0.07, 7)  # 7 warranted rows are the lowest -> fire at matched budget 7
    cell_probs[7:10] = np.array([0.08, 0.09, 0.10])  # remaining positives low -> perfect AUC, budget-excluded
    cell_probs[10:] = np.linspace(0.5, 0.99, 10)  # negatives high -> never fire
    inc_probs = np.full(n, 0.9)
    inc_probs[10:17] = 0.1  # incumbent fires 7 non-warranted rows -> budget 7, zero warranted coverage
    return {
        "cell_probs": cell_probs,
        "inc_probs": inc_probs,
        "labels": mask(list(range(10)), n),
        "corrective": mask(list(range(10)), n),
        "prose": np.ones(n, dtype=bool),
    }


class TestCorrectedGate:
    def test_dominant_candidate_promotes(self) -> None:
        f = gate_frame()
        result = corrected_gate(
            f["cell_probs"], f["inc_probs"], candidate="cand", incumbent="inc", incumbent_threshold=0.5,
            labels=f["labels"], corrective=f["corrective"], prose=f["prose"], harmful_favors_incumbent=False,
        )
        assert (result.coverage_wins, result.coverage_losses) == (7, 0)
        assert result.coverage_sign_p < 0.05
        assert result.budget_held is True
        assert result.auc_not_regressed is True
        assert result.promote is True
        assert watcher_promotable(result).promote is True

    def test_harmful_pending_leaves_promote_none(self) -> None:
        f = gate_frame()
        result = corrected_gate(
            f["cell_probs"], f["inc_probs"], candidate="cand", incumbent="inc", incumbent_threshold=0.5,
            labels=f["labels"], corrective=f["corrective"], prose=f["prose"],
        )
        assert result.harmful_favors_incumbent is None
        assert result.promote is None

    def test_regressed_auc_blocks_watcher_bar(self) -> None:
        labels = mask([0, 1, 2, 3, 4], 10)
        inc_probs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.6, 0.7, 0.8, 0.9, 0.95])  # AUC 1.0
        cell_probs = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 0.01, 0.02, 0.03, 0.04, 0.05])  # AUC 0.0 (inverted)
        result = corrected_gate(
            cell_probs, inc_probs, candidate="cand", incumbent="inc", incumbent_threshold=0.005,
            labels=labels, corrective=labels, prose=np.ones(10, dtype=bool),
        )
        assert result.incumbent_auc > result.cell_auc
        assert result.auc_not_regressed is False
        assert watcher_promotable(result) == Verdict(False, "candidate AUC 0.0000 <= incumbent 1.0000")


class TestWatcherBar:
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            pytest.param(
                {},
                Verdict(True, "candidate AUC 0.8000 > incumbent 0.7000, budget held, coverage 5 >= 0"),
                id="promote",
            ),
            pytest.param(
                {"cell_auc": 0.70},
                Verdict(False, "candidate AUC 0.7000 <= incumbent 0.7000"),
                id="auc-not-beaten",
            ),
            pytest.param(
                {"budget_held": False},
                Verdict(False, "fire budget exceeded at matched fires"),
                id="budget-exceeded",
            ),
            pytest.param(
                {"coverage_wins": 1, "coverage_losses": 3},
                Verdict(False, "coverage losses 3 > wins 1"),
                id="coverage-losses-exceed-wins",
            ),
        ],
    )
    def test_bar(self, overrides: dict[str, object], expected: Verdict) -> None:
        assert watcher_promotable(gate_result(**overrides)) == expected


class TestGateBar:
    def test_no_incumbent_promotes(self) -> None:
        assert gate_promotable(GOOD, None) == Verdict(True, "no incumbent")

    def test_lower_pr_auc_rejected(self) -> None:
        assert gate_promotable(GOOD | {"pr_auc": 0.90}, GOOD) == Verdict(False, "pr_auc 0.9000 <= incumbent 0.9400")

    def test_equal_pr_auc_rejected(self) -> None:
        assert gate_promotable(GOOD, GOOD) == Verdict(False, "pr_auc 0.9400 <= incumbent 0.9400")

    def test_recall_regression_rejected(self) -> None:
        candidate = GOOD | {PR_AUC_KEY: 0.96, RECALL_KEY: 0.10}
        assert gate_promotable(candidate, GOOD) == Verdict(False, "recall 0.1000 < incumbent 0.3000")

    def test_better_pr_auc_and_held_recall_promotes(self) -> None:
        assert gate_promotable(GOOD | {"pr_auc": 0.96}, GOOD) == Verdict(
            True, "pr_auc 0.9600 > incumbent 0.9400, recall held"
        )


class TestBarsFailClosedOnNaN:
    @pytest.mark.parametrize("field", ["cell_auc", "incumbent_auc", "coverage_wins", "coverage_losses"])
    def test_watcher_bar_rejects_non_finite_metric(self, field: str) -> None:
        verdict = watcher_promotable(gate_result(**{field: float("nan")}))
        assert verdict.promote is False
        assert "non-finite" in verdict.reason

    @pytest.mark.parametrize(
        ("who", "key"),
        [("candidate", PR_AUC_KEY), ("candidate", RECALL_KEY), ("incumbent", PR_AUC_KEY), ("incumbent", RECALL_KEY)],
    )
    def test_gate_bar_rejects_non_finite_metric(self, who: str, key: str) -> None:
        candidate = GOOD | {PR_AUC_KEY: 0.99}  # would otherwise beat the incumbent
        incumbent = dict(GOOD)
        (candidate if who == "candidate" else incumbent)[key] = float("nan")
        verdict = gate_promotable(candidate, incumbent)
        assert verdict.promote is False
        assert "non-finite" in verdict.reason

    def test_gate_bar_missing_incumbent_metric_raises(self) -> None:
        with pytest.raises(KeyError):
            gate_promotable(GOOD | {PR_AUC_KEY: 0.99}, {PR_AUC_KEY: 0.94})  # incumbent record missing recall


class TestShouldRetrain:
    def test_no_incumbent_retrains(self) -> None:
        assert should_retrain(None, "d1", force=False)

    def test_unchanged_digest_skips(self) -> None:
        assert not should_retrain(version_info("d1"), "d1", force=False)

    def test_force_retrains_unchanged(self) -> None:
        assert should_retrain(version_info("d1"), "d1", force=True)

    def test_changed_digest_retrains(self) -> None:
        assert should_retrain(version_info("d1"), "d2", force=False)


class TestJournal:
    def test_appends_exact_json_and_returns_line(self, tmp_path: Path) -> None:
        line = journal(
            "watcher",
            "promoted v005",
            dataset_digest="abc123",
            metrics={"auc": 0.8},
            version="v005",
            state_dir=tmp_path,
        )
        assert line == "watcher: promoted v005"
        entries = [json.loads(row) for row in (tmp_path / "retrain" / "journal.jsonl").read_text().splitlines()]
        assert len(entries) == 1
        entry = entries[0]
        ts = datetime.fromisoformat(entry.pop("ts"))
        assert ts.tzinfo is not None and ts.utcoffset() == timedelta(0)
        assert entry == {
            "component": "watcher",
            "verdict": "promoted v005",
            "dataset_digest": "abc123",
            "metrics": {"auc": 0.8},
            "version": "v005",
        }

    def test_appends_one_line_per_call(self, tmp_path: Path) -> None:
        journal("gate", "skipped", dataset_digest="d1", state_dir=tmp_path)
        journal("gate", "promoted v002", dataset_digest="d2", version="v002", state_dir=tmp_path)
        lines = (tmp_path / "retrain" / "journal.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["metrics"] == {}
