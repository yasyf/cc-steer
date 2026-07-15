from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId

from cc_steer.rendering import gate_text
from cc_steer.retrain.evalset import (
    GATE_EVAL_NAME,
    MANIFEST_NAME,
    RENDER_VERSION,
    WATCHER_EVAL_NAME,
    EmptyEvalContext,
    EvalFrame,
    FrozenViolationError,
    ProbsStoreError,
    SchemaError,
    freeze_eval,
    load_frozen,
    load_probs,
    probs_path,
    write_probs,
)

if TYPE_CHECKING:
    from pathlib import Path

ROWS = [
    ("r0", True, "wrong_approach", ""),
    ("r1", True, "direction", ""),
    ("r2", True, "wrong_approach", "question_answer"),
    ("r3", False, "", ""),
]


def watcher_test_table(rows: list[tuple[str, bool, str, str]] = ROWS) -> pa.Table:
    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    return pa.table(
        {
            "prompt": pa.array(
                [[{"role": "user", "content": f"context for {rid}"}] for rid, *_ in rows], type=pa.list_(message)
            ),
            "completion": pa.array(
                [[{"role": "assistant", "content": "steer" if label else "NO_STEER"}] for _, label, *_ in rows],
                type=pa.list_(message),
            ),
            "verbatim": ["v" if label else "" for _, label, *_ in rows],
            "label": [label for _, label, *_ in rows],
            "id": [rid for rid, *_ in rows],
            "category": [category for *_, category, _ in rows],
            "source_kind": [source for *_, source in rows],
            "session_id": [f"s{i}" for i in range(len(rows))],
            "split": ["test"] * len(rows),
        }
    )


def write_dataset(root: Path, table: pa.Table) -> Path:
    (root / "watcher").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, root / "watcher" / "test.parquet")
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "dataset", watcher_test_table())


@pytest.fixture
def eval_dir(tmp_path: Path) -> Path:
    return tmp_path / "eval"


class TestFreezeEval:
    def test_round_trip_with_manifest(self, dataset: Path, eval_dir: Path) -> None:
        sha = freeze_eval(dataset_dir=dataset, root=eval_dir)
        assert json.loads((eval_dir / MANIFEST_NAME).read_text()) == {WATCHER_EVAL_NAME: sha}
        assert load_frozen(root=eval_dir).column("id").to_pylist() == ["r0", "r1", "r2", "r3"]

    def test_idempotent_for_identical_bytes(self, dataset: Path, eval_dir: Path) -> None:
        assert freeze_eval(dataset_dir=dataset, root=eval_dir) == freeze_eval(dataset_dir=dataset, root=eval_dir)

    def test_refuses_overwrite_with_changed_content(self, dataset: Path, eval_dir: Path) -> None:
        freeze_eval(dataset_dir=dataset, root=eval_dir)
        original = (eval_dir / WATCHER_EVAL_NAME).read_bytes()
        write_dataset(dataset, watcher_test_table([("z0", True, "wrong_approach", "")]))
        with pytest.raises(FrozenViolationError, match="refusing to overwrite"):
            freeze_eval(dataset_dir=dataset, root=eval_dir)
        assert (eval_dir / WATCHER_EVAL_NAME).read_bytes() == original

    def test_refuses_refreeze_when_file_deleted_but_manifest_survives(self, dataset: Path, eval_dir: Path) -> None:
        freeze_eval(dataset_dir=dataset, root=eval_dir)
        (eval_dir / WATCHER_EVAL_NAME).unlink()  # frozen file gone, manifest entry survives
        write_dataset(dataset, watcher_test_table([("z0", True, "wrong_approach", "")]))  # drifted source
        with pytest.raises(FrozenViolationError, match="refusing to refreeze"):
            freeze_eval(dataset_dir=dataset, root=eval_dir)
        assert not (eval_dir / WATCHER_EVAL_NAME).exists()

    def test_rematerializes_deleted_file_from_identical_source(self, dataset: Path, eval_dir: Path) -> None:
        sha = freeze_eval(dataset_dir=dataset, root=eval_dir)
        (eval_dir / WATCHER_EVAL_NAME).unlink()
        assert freeze_eval(dataset_dir=dataset, root=eval_dir) == sha  # identical source re-materializes cleanly
        assert load_frozen(root=eval_dir).column("id").to_pylist() == ["r0", "r1", "r2", "r3"]

    def test_missing_source_raises(self, tmp_path: Path, eval_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            freeze_eval(dataset_dir=tmp_path / "empty", root=eval_dir)

    def test_rejects_source_missing_required_columns(self, tmp_path: Path, eval_dir: Path) -> None:
        bad = write_dataset(tmp_path / "dataset", pa.table({"id": ["r0"], "label": [True]}))
        with pytest.raises(SchemaError, match="missing columns"):
            freeze_eval(dataset_dir=bad, root=eval_dir)

    def test_rejects_watcher_rows_with_empty_rendered_context(self, tmp_path: Path, eval_dir: Path) -> None:
        message = pa.struct([("role", pa.string()), ("content", pa.string())])
        table = pa.table(
            {
                "prompt": pa.array(
                    [
                        [{"role": "user", "content": "real context"}],
                        [{"role": "assistant", "content": "   "}],  # whitespace only
                        [],  # no messages at all
                    ],
                    type=pa.list_(message),
                ),
                "completion": pa.array([[{"role": "assistant", "content": "steer"}]] * 3, type=pa.list_(message)),
                "verbatim": ["v"] * 3,
                "label": [True, True, True],
                "id": ["good", "blank", "empty"],
                "category": ["wrong_approach"] * 3,
                "source_kind": [""] * 3,
                "session_id": ["s0", "s1", "s2"],
                "split": ["test"] * 3,
            }
        )
        dataset = write_dataset(tmp_path / "dataset", table)
        with pytest.raises(EmptyEvalContext) as excinfo:
            freeze_eval(dataset_dir=dataset, root=eval_dir)
        assert excinfo.value.ids == ("blank", "empty")
        assert "blank" in str(excinfo.value) and "empty" in str(excinfo.value)
        assert not (eval_dir / WATCHER_EVAL_NAME).exists()  # nothing frozen over invalid rows

    def test_rejects_gate_rows_with_empty_text(self, tmp_path: Path, eval_dir: Path) -> None:
        dataset = tmp_path / "dataset"
        (dataset / "gate").mkdir(parents=True)
        empty_text = gate_text(
            ContextWindow(
                anchor=EventRef(SessionId("s1"), EventUuid("u1")),
                before=(TurnRef(role="assistant", refs=(), preview="   ", tool_digests=()),),
                trigger=None,
                after=(),
                fidelity="full",
                preview_chars=200,
            )
        )
        assert empty_text == "<assistant>\n   "
        gate = pa.table(
            {
                "id": ["g0", "g1"],
                "text": ["real context", empty_text],
                "label": [True, False],
                "kind": ["positive", "hard_negative"],
                "offset_turns": [0, 0],
                "source_kind": ["transcript_message", "question_answer"],
                "category": ["wrong_approach", ""],
                "session_id": ["s0", "s1"],
                "split": ["test", "test"],
            }
        )
        pq.write_table(gate, dataset / "gate" / "test.parquet")
        with pytest.raises(EmptyEvalContext) as excinfo:
            freeze_eval("gate", dataset_dir=dataset, root=eval_dir)
        assert excinfo.value.ids == ("g1",)
        assert not (eval_dir / GATE_EVAL_NAME).exists()

    def test_freezes_both_views_and_merges_the_manifest(self, tmp_path: Path, eval_dir: Path) -> None:
        dataset = write_dataset(tmp_path / "dataset", watcher_test_table())
        (dataset / "gate").mkdir(parents=True, exist_ok=True)
        gate = pa.table(
            {
                "id": ["g0", "g1"],
                "text": ["a", "b"],
                "label": [True, False],
                "kind": ["positive", "hard_negative"],
                "offset_turns": [0, 0],
                "source_kind": ["transcript_message", "question_answer"],
                "category": ["wrong_approach", ""],
                "session_id": ["s0", "s1"],
                "split": ["test", "test"],
            }
        )
        pq.write_table(gate, dataset / "gate" / "test.parquet")
        watcher_sha = freeze_eval("watcher", dataset_dir=dataset, root=eval_dir)
        gate_sha = freeze_eval("gate", dataset_dir=dataset, root=eval_dir)
        assert json.loads((eval_dir / MANIFEST_NAME).read_text()) == {
            WATCHER_EVAL_NAME: watcher_sha,
            GATE_EVAL_NAME: gate_sha,
        }
        assert load_frozen("gate", root=eval_dir).column("id").to_pylist() == ["g0", "g1"]
        assert load_frozen("watcher", root=eval_dir).column("id").to_pylist() == ["r0", "r1", "r2", "r3"]


class TestLoadFrozen:
    def test_tampered_file_raises(self, dataset: Path, eval_dir: Path) -> None:
        freeze_eval(dataset_dir=dataset, root=eval_dir)
        pq.write_table(watcher_test_table([("tampered", True, "wrong_approach", "")]), eval_dir / WATCHER_EVAL_NAME)
        with pytest.raises(FrozenViolationError, match="mismatch"):
            load_frozen(root=eval_dir)

    def test_missing_manifest_entry_raises(self, eval_dir: Path) -> None:
        with pytest.raises(FrozenViolationError, match="not in"):
            load_frozen(root=eval_dir)

    def test_revalidates_columns_after_sha_passes(self, eval_dir: Path) -> None:
        eval_dir.mkdir(parents=True)
        pq.write_table(pa.table({"id": ["r0"]}), eval_dir / WATCHER_EVAL_NAME)  # schema-invalid frozen file
        sha = hashlib.sha256((eval_dir / WATCHER_EVAL_NAME).read_bytes()).hexdigest()
        (eval_dir / MANIFEST_NAME).write_text(json.dumps({WATCHER_EVAL_NAME: sha}) + "\n")  # sha matches, so it loads
        with pytest.raises(SchemaError, match="missing columns"):
            load_frozen(root=eval_dir)


class TestEvalFrame:
    def frame(self, dataset: Path, eval_dir: Path) -> EvalFrame:
        freeze_eval(dataset_dir=dataset, root=eval_dir)
        return EvalFrame.load(root=eval_dir)

    def test_masks_computed_exactly(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        assert frame.ids == ("r0", "r1", "r2", "r3")
        assert frame.labels.tolist() == [True, True, True, False]
        assert frame.corrective.tolist() == [True, False, True, False]
        assert frame.prose.tolist() == [True, True, False, True]

    def test_tails_are_render_v2_flattened(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        assert frame.tails[0] == "<user>\ncontext for r0"
        assert len(frame) == 4

    def test_duplicate_row_ids_fail_loud(self, tmp_path: Path, eval_dir: Path) -> None:
        dupes = write_dataset(
            tmp_path / "dataset",
            watcher_test_table([("dup", True, "wrong_approach", ""), ("dup", False, "", "")]),
        )
        freeze_eval(dataset_dir=dupes, root=eval_dir)
        with pytest.raises(SchemaError, match="duplicate row ids"):
            EvalFrame.load(root=eval_dir)


class TestProbsStore:
    def frame(self, dataset: Path, eval_dir: Path) -> EvalFrame:
        freeze_eval(dataset_dir=dataset, root=eval_dir)
        return EvalFrame.load(root=eval_dir)

    def probs(self, frame: EvalFrame) -> dict[str, float]:
        return {row_id: 0.1 * (index + 1) for index, row_id in enumerate(frame.ids)}

    def test_write_load_round_trip(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        path = write_probs(frame, "v001", self.probs(frame), auc=0.72, root=eval_dir)
        assert path == probs_path("v001", root=eval_dir)
        payload = json.loads(path.read_text())
        assert payload["meta"] == {"dataset_digest": frame.digest, "render": RENDER_VERSION, "auc": 0.72}
        assert load_probs(frame, "v001", expected_render=RENDER_VERSION, root=eval_dir).tolist() == [
            0.1,
            0.2,
            0.30000000000000004,
            0.4,
        ]

    def test_write_refuses_partial(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        partial = {row_id: 0.5 for row_id in frame.ids[:-1]}
        with pytest.raises(ProbsStoreError, match="partial"):
            write_probs(frame, "v001", partial, auc=0.5, root=eval_dir)

    def test_load_missing_file_fails_loud(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        with pytest.raises(ProbsStoreError, match="no stored probs"):
            load_probs(frame, "v999", expected_render=RENDER_VERSION, root=eval_dir)

    def test_load_digest_mismatch_fails_loud(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        write_probs(frame, "v001", self.probs(frame), auc=0.72, root=eval_dir)
        moved = replace(frame, digest="deadbeefdeadbeef")
        with pytest.raises(ProbsStoreError, match="frame digest"):
            load_probs(moved, "v001", expected_render=RENDER_VERSION, root=eval_dir)

    def test_load_missing_rows_fails_loud(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        write_probs(frame, "v001", self.probs(frame), auc=0.72, root=eval_dir)
        extended = replace(frame, ids=(*frame.ids, "extra"))
        with pytest.raises(ProbsStoreError, match="does not cover"):
            load_probs(extended, "v001", expected_render=RENDER_VERSION, root=eval_dir)

    def test_load_render_mismatch_fails_loud(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        write_probs(frame, "v001", self.probs(frame), auc=0.72, root=eval_dir)  # stamps render=RENDER_VERSION (2)
        with pytest.raises(ProbsStoreError, match="render"):
            load_probs(frame, "v001", expected_render=1, root=eval_dir)

    def test_load_accepts_matching_non_current_render(self, dataset: Path, eval_dir: Path) -> None:
        frame = self.frame(dataset, eval_dir)
        path = write_probs(frame, "v001", self.probs(frame), auc=0.72, root=eval_dir)
        payload = json.loads(path.read_text())
        payload["meta"]["render"] = 1  # a migrated render-1 incumbent file
        path.write_text(json.dumps(payload))
        assert load_probs(frame, "v001", expected_render=1, root=eval_dir).tolist() == [
            0.1,
            0.2,
            0.30000000000000004,
            0.4,
        ]

    @pytest.mark.parametrize("bad", [float("nan"), 1.2, -0.1], ids=["nan", "above-one", "below-zero"])
    def test_write_rejects_out_of_range_prob(self, dataset: Path, eval_dir: Path, bad: float) -> None:
        frame = self.frame(dataset, eval_dir)
        probs = self.probs(frame) | {frame.ids[0]: bad}
        with pytest.raises(ProbsStoreError, match=r"\[0, 1\]"):
            write_probs(frame, "v001", probs, auc=0.5, root=eval_dir)
