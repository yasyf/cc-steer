from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cc_steer import registry
from cc_steer.retrain import data, lexical, promotion
from cc_steer.watcher.gate import ARTIFACT_NAME, THRESHOLD_KEY, LexicalGate

if TYPE_CHECKING:
    from pathlib import Path

GATE_COLUMNS = ("id", "text", "label", "kind", "offset_turns", "source_kind", "category", "session_id", "split")


def gate_rows(split: str, *, n_pos: int, n_neg: int) -> list[dict[str, object]]:
    pos = [
        {
            "id": f"{split}-p{i}",
            "text": f"use a frozen dataclass number {i} instead of a plain dict right here",
            "label": True,
            "kind": "positive",
            "offset_turns": 0,
            "source_kind": "transcript_message",
            "category": "wrong_approach",
            "session_id": f"sp{i}",
            "split": split,
        }
        for i in range(n_pos)
    ]
    neg = [
        {
            "id": f"{split}-n{i}",
            "text": f"the build passed cleanly on run {i} and the suite looks totally fine today",
            "label": False,
            "kind": "hard_negative" if i % 2 else "random_negative",
            "offset_turns": 0,
            "source_kind": "question_answer" if i % 2 else "transcript_message",
            "category": "",
            "session_id": f"sn{i}",
            "split": split,
        }
        for i in range(n_neg)
    ]
    return pos + neg


def gate_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.table({column: [row[column] for row in rows] for column in GATE_COLUMNS})


def write_gate_dataset(root: Path) -> Path:
    (root / "gate").mkdir(parents=True, exist_ok=True)
    pq.write_table(gate_table(gate_rows("train", n_pos=20, n_neg=20)), root / "gate" / "train.parquet")
    pq.write_table(gate_table(gate_rows("test", n_pos=12, n_neg=12)), root / "gate" / "test.parquet")
    return root


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return write_gate_dataset(tmp_path / "dataset")


@pytest.fixture
def eval_dir(tmp_path: Path, dataset_dir: Path) -> Path:
    from cc_steer.retrain import evalset

    root = tmp_path / "eval"
    evalset.freeze_eval("gate", dataset_dir=dataset_dir, root=root)
    return root


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    return {"registry": tmp_path / "models", "state": tmp_path / "state"}


class TestGateMetrics:
    def test_emits_exactly_promotion_metric_keys(self) -> None:
        frame = lexical.GateFrame.from_table(gate_table(gate_rows("test", n_pos=8, n_neg=8)))
        probs = np.linspace(0.05, 0.95, len(frame))
        metrics = lexical.gate_metrics(frame, probs, temperature=1.3)
        assert promotion.PR_AUC_KEY == "pr_auc"
        assert promotion.RECALL_KEY == "recall_at_2per100_viewratio_proxy"
        assert promotion.PR_AUC_KEY in metrics
        assert promotion.RECALL_KEY in metrics
        assert lexical.THRESHOLD_METRIC in metrics
        assert 0.0 <= metrics[promotion.PR_AUC_KEY] <= 1.0


class TestRetrainGate:
    def test_promote_then_reject_on_equal_metrics(
        self, dataset_dir: Path, eval_dir: Path, roots: dict[str, Path]
    ) -> None:
        first = lexical.retrain_gate(
            force=True,
            dataset_dir=dataset_dir,
            eval_root=eval_dir,
            registry_root=roots["registry"],
            state_dir=roots["state"],
        )
        assert first.startswith("gate: promoted")
        promoted = registry.current("gate", root=roots["registry"])
        assert promoted is not None
        assert "hf_revision" not in promoted.metadata
        assert len(registry.versions("gate", root=roots["registry"])) == 1

        second = lexical.retrain_gate(
            force=True,
            dataset_dir=dataset_dir,
            eval_root=eval_dir,
            registry_root=roots["registry"],
            state_dir=roots["state"],
        )
        assert second.startswith("gate: rejected")
        assert "<= incumbent" in second
        # A reject never registers a new version, and the incumbent is untouched.
        assert len(registry.versions("gate", root=roots["registry"])) == 1
        assert registry.current("gate", root=roots["registry"]).version == promoted.version
        entries = [
            json.loads(line) for line in (roots["state"] / "retrain" / "journal.jsonl").read_text().splitlines()
        ]
        assert all("hf_revision" not in entry for entry in entries)

    def test_hf_revision_threads_to_registry_and_journal(
        self, dataset_dir: Path, eval_dir: Path, roots: dict[str, Path]
    ) -> None:
        (dataset_dir / data.HF_PUSH_NAME).write_text(
            json.dumps({"hf_revision": "sha-gate", "repo_id": "u/r", "ts": "2026-07-17T00:00:00+00:00"})
        )
        verdict = lexical.retrain_gate(
            force=True,
            dataset_dir=dataset_dir,
            eval_root=eval_dir,
            registry_root=roots["registry"],
            state_dir=roots["state"],
        )
        assert verdict.startswith("gate: promoted")
        current = registry.current("gate", root=roots["registry"])
        assert current is not None
        assert current.metadata["hf_revision"] == "sha-gate"
        entry = json.loads((roots["state"] / "retrain" / "journal.jsonl").read_text())
        assert entry["hf_revision"] == "sha-gate"

    def test_skip_when_digest_unchanged(self, dataset_dir: Path, roots: dict[str, Path]) -> None:
        digest = lexical.gate_train_digest(dataset_dir=dataset_dir)
        info = registry.register("gate", {ARTIFACT_NAME: b"stub"}, {"dataset_digest": digest}, root=roots["registry"])
        registry.promote("gate", info.version, root=roots["registry"])
        verdict = lexical.retrain_gate(
            force=False, dataset_dir=dataset_dir, registry_root=roots["registry"], state_dir=roots["state"]
        )
        assert verdict == f"gate: skipped (no new data at digest {digest})"
        assert (roots["state"] / "retrain" / "journal.jsonl").exists()

    def test_promoted_artifact_loads_through_inference_gate(
        self, dataset_dir: Path, eval_dir: Path, roots: dict[str, Path]
    ) -> None:
        candidate = lexical.train_gate(dataset_dir=dataset_dir, eval_root=eval_dir)
        lexical.register_candidate(candidate, digest="d0", root=roots["registry"])
        gate = LexicalGate(root=roots["registry"])
        text = "use a frozen dataclass instead of a dict"
        # The served score must equal the trainer-side calibrated prob: same vectorizer hstack order and temperature.
        expected = float(lexical.probs_from_logits(candidate.model.logits([text]), candidate.temperature)[0])
        assert gate.score(text) == pytest.approx(expected)
        assert isinstance(gate.threshold, float)
        promoted = registry.current("gate", root=roots["registry"])
        assert THRESHOLD_KEY in promoted.metadata["thresholds"]
        assert (promoted.path / ARTIFACT_NAME).exists()

    def test_corrupt_incumbent_without_metrics_fails_loud(
        self, dataset_dir: Path, eval_dir: Path, roots: dict[str, Path]
    ) -> None:
        info = registry.register("gate", {ARTIFACT_NAME: b"stub"}, {"dataset_digest": "stale"}, root=roots["registry"])
        registry.promote("gate", info.version, root=roots["registry"])
        with pytest.raises(KeyError, match="metrics"):
            lexical.retrain_gate(
                force=True,
                dataset_dir=dataset_dir,
                eval_root=eval_dir,
                registry_root=roots["registry"],
                state_dir=roots["state"],
            )

    def test_fresh_epoch_promotes_over_metricsless_incumbent(
        self, dataset_dir: Path, eval_dir: Path, roots: dict[str, Path]
    ) -> None:
        # A metrics-less incumbent would KeyError on the normal path; fresh-epoch takes the no-incumbent path instead.
        info = registry.register("gate", {ARTIFACT_NAME: b"stub"}, {"dataset_digest": "stale"}, root=roots["registry"])
        registry.promote("gate", info.version, root=roots["registry"])
        verdict = lexical.retrain_gate(
            force=True,
            fresh_epoch=True,
            dataset_dir=dataset_dir,
            eval_root=eval_dir,
            registry_root=roots["registry"],
            state_dir=roots["state"],
        )
        assert verdict.startswith("gate: fresh-epoch promoted")
        assert "(no incumbent)" in verdict
        assert registry.current("gate", root=roots["registry"]).version != info.version

    def test_journals_every_pass(self, dataset_dir: Path, eval_dir: Path, roots: dict[str, Path]) -> None:
        lexical.retrain_gate(
            force=True,
            dataset_dir=dataset_dir,
            eval_root=eval_dir,
            registry_root=roots["registry"],
            state_dir=roots["state"],
        )
        entries = [
            json.loads(line)
            for line in (roots["state"] / "retrain" / "journal.jsonl").read_text().splitlines()
        ]
        assert len(entries) == 1
        assert entries[0]["component"] == "gate"
        assert entries[0]["verdict"].startswith("promoted")
