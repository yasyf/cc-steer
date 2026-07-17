from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cc_steer import registry
from cc_steer.retrain.promotion import (
    PR_AUC_KEY,
    RECALL_KEY,
    RETRAIN_LOG_LABEL,
    RETRAIN_LOG_TITLE,
    GateResult,
    Verdict,
    gate_promotable,
    journal,
    should_retrain,
    watcher_promotable,
)

GOOD = {PR_AUC_KEY: 0.94, RECALL_KEY: 0.30}


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

    def test_missing_digest_retrains(self) -> None:
        incumbent = registry.VersionInfo(
            component="watcher", version="v001-20260101-abcdef123456", path=Path("/x"), metadata={}
        )
        assert should_retrain(incumbent, "d1", force=False)


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

    def test_appends_hf_revision_only_when_provided(self, tmp_path: Path) -> None:
        journal("gate", "promoted v002", dataset_digest="d2", hf_revision="sha-gate", state_dir=tmp_path)
        entry = json.loads((tmp_path / "retrain" / "journal.jsonl").read_text())
        ts = datetime.fromisoformat(entry.pop("ts"))
        assert ts.tzinfo is not None and ts.utcoffset() == timedelta(0)
        assert entry == {
            "component": "gate",
            "dataset_digest": "d2",
            "hf_revision": "sha-gate",
            "metrics": {},
            "verdict": "promoted v002",
            "version": None,
        }

    def test_appends_one_line_per_call(self, tmp_path: Path) -> None:
        journal("gate", "skipped", dataset_digest="d1", state_dir=tmp_path)
        journal("gate", "promoted v002", dataset_digest="d2", version="v002", state_dir=tmp_path)
        lines = (tmp_path / "retrain" / "journal.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["metrics"] == {}

    def test_mirrors_line_to_cc_notes_retrain_log(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []
        responses = {
            ("log", "list"): json.dumps([{"id": "rid", "title": RETRAIN_LOG_TITLE}]),
            ("log", "append"): "",
        }

        def fake(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            key = tuple(argv[1:3])
            return subprocess.CompletedProcess(
                argv, returncode=0 if key in responses else 1, stdout=responses.get(key, ""), stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake)
        line = journal("watcher", "promoted v009", dataset_digest="d9", state_dir=tmp_path)
        assert line == "watcher: promoted v009"
        listed = next(argv for argv in calls if argv[1:3] == ["log", "list"])
        assert listed[listed.index("--label") + 1] == RETRAIN_LOG_LABEL
        append = next(argv for argv in calls if argv[1:3] == ["log", "append"])
        assert append[3:] == ["rid", "--entry", "watcher: promoted v009"]
        # journal() stays the sole writer: one JSONL line, the mirror adds no second write.
        assert (tmp_path / "retrain" / "journal.jsonl").read_text().count("\n") == 1
