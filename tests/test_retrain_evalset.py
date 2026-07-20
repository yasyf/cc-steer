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

from cc_steer.decisions import DecisionRow, MineResult, write_decisions
from cc_steer.rendering import ask_block, gate_text
from cc_steer.retrain.evalset import (
    GATE_EVAL_NAME,
    MANIFEST_NAME,
    RENDER_VERSION,
    STEER_TYPE_CATEGORIES,
    STEER_TYPE_EVAL_NAME,
    WATCHER_EVAL_NAME,
    EmptyEvalContext,
    EvalFrame,
    FrozenViolationError,
    PickFrame,
    ProbsStoreError,
    SchemaError,
    SplitLeakError,
    SteerTypeFrame,
    build_steer_type_eval,
    freeze_eval,
    freeze_pick,
    freeze_steer_type,
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


def watcher_table_content(pairs: list[tuple[str, str]], *, split: str = "test") -> pa.Table:
    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    return pa.table(
        {
            "prompt": pa.array(
                [[{"role": "user", "content": content}] for _, content in pairs], type=pa.list_(message)
            ),
            "completion": pa.array(
                [[{"role": "assistant", "content": "steer"}] for _ in pairs], type=pa.list_(message)
            ),
            "verbatim": ["v" for _ in pairs],
            "label": [True for _ in pairs],
            "id": [rid for rid, _ in pairs],
            "category": ["wrong_approach" for _ in pairs],
            "source_kind": ["" for _ in pairs],
            "session_id": [f"s{i}" for i in range(len(pairs))],
            "split": [split] * len(pairs),
        }
    )


def write_watcher_split(root: Path, test_pairs: list[tuple[str, str]], train_pairs: list[tuple[str, str]]) -> Path:
    (root / "watcher").mkdir(parents=True, exist_ok=True)
    pq.write_table(watcher_table_content(test_pairs), root / "watcher" / "test.parquet")
    pq.write_table(watcher_table_content(train_pairs, split="train"), root / "watcher" / "train.parquet")
    return root


class TestSplitDisjointness:
    def test_freeze_refuses_a_train_eval_leak(self, tmp_path: Path, eval_dir: Path) -> None:
        dataset = write_watcher_split(
            tmp_path / "dataset",
            [("t0", "the exact same shared context window"), ("t1", "only in the test split")],
            [("x0", "the exact same shared context window"), ("x1", "only in the train split")],
        )
        with pytest.raises(SplitLeakError) as excinfo:
            freeze_eval("watcher", dataset_dir=dataset, root=eval_dir)
        assert excinfo.value.ids == ("t0",)  # the leaked test row is named
        assert not (eval_dir / WATCHER_EVAL_NAME).exists()  # nothing frozen over a leak

    def test_disjoint_split_freezes_and_records_the_check(self, tmp_path: Path, eval_dir: Path) -> None:
        dataset = write_watcher_split(
            tmp_path / "dataset",
            [("t0", "only in the test split a"), ("t1", "only in the test split b")],
            [("x0", "only in the train split a"), ("x1", "only in the train split b"), ("x2", "and a third")],
        )
        sha = freeze_eval("watcher", dataset_dir=dataset, root=eval_dir)
        manifest = json.loads((eval_dir / MANIFEST_NAME).read_text())
        assert manifest[WATCHER_EVAL_NAME] == sha
        assert manifest[f"{WATCHER_EVAL_NAME}.meta"]["disjoint_train_rows_checked"] == 3

    def test_no_train_sibling_skips_the_check(self, dataset: Path, eval_dir: Path) -> None:
        sha = freeze_eval(dataset_dir=dataset, root=eval_dir)  # only test.parquet exists
        assert json.loads((eval_dir / MANIFEST_NAME).read_text()) == {WATCHER_EVAL_NAME: sha}


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


def write_traces(root: Path, rows: list[dict[str, object]]) -> Path:
    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    table = pa.table(
        {
            "id": [row["id"] for row in rows],
            "context": pa.array([row["context"] for row in rows], type=pa.list_(message)),
            "agent_action": [row["agent_action"] for row in rows],
            "user_message": [row["user_message"] for row in rows],
            "category": [row["category"] for row in rows],
            "is_steering": [row["is_steering"] for row in rows],
            "source_kind": [row["source_kind"] for row in rows],
            "session_id": [row["session_id"] for row in rows],
            "split": [row["split"] for row in rows],
        }
    )
    (root / "traces").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, root / "traces" / "test.parquet")
    return root


def trace_row(
    rid: str, *, category: str, is_steering: bool, user: str, action: str = "did a thing"
) -> dict[str, object]:
    return {
        "id": rid,
        "context": [{"role": "user", "content": f"open the {rid} task and start working on the module"}],
        "agent_action": action,
        "user_message": user,
        "category": category,
        "is_steering": is_steering,
        "source_kind": "transcript_message",
        "session_id": f"session-{rid}",
        "split": "test",
    }


def decision_row(rid: str, *, question: str, options: tuple[str, ...], chosen: int, split: str = "test") -> DecisionRow:
    return DecisionRow(
        id=rid,
        session_id=f"session-{rid}",
        occurred_at="2026-01-01T00:00:00",
        turn_index=1,
        event_uuid=f"uuid-{rid}",
        tool_use_id=f"tool-{rid}",
        question=question,
        header="Pick",
        options=options,
        multi_select=False,
        answer=options[chosen],
        chosen_index=(chosen,),
        is_custom=False,
        split=split,
    )


class TestSteerTypeFrame:
    def source_rows(self) -> list[dict[str, object]]:
        return [
            trace_row("r0", category="wrong_approach", is_steering=True, user="no, do not vendor the dependency"),
            trace_row("r1", category="direction", is_steering=True, user="use python 3.14 for the pin"),
            trace_row("r2", category="operational_directive", is_steering=False, user="now run the tests"),
            # r3 is a byte-identical near-duplicate of r0's classifier input and collapses in dedup.
            trace_row("r3", category="wrong_approach", is_steering=True, user="no, do not vendor the dependency"),
        ]

    def test_build_and_freeze_round_trip_collapses_near_dups(self, tmp_path: Path) -> None:
        dataset = write_traces(tmp_path / "dataset", self.source_rows())
        eval_dir = tmp_path / "eval"
        freeze_steer_type(dataset_dir=dataset, root=eval_dir)
        frame = SteerTypeFrame.load(root=eval_dir)
        assert len(frame) == 3  # r0 and r3 render identical text and collapse to one representative
        assert set(frame.categories) == {"wrong_approach", "direction", "operational_directive"}
        assert set(frame.categories) <= set(STEER_TYPE_CATEGORIES)

    def test_freeze_is_deterministic_in_bytes_and_digest(self, tmp_path: Path) -> None:
        rows = self.source_rows()
        first = write_traces(tmp_path / "a", rows)
        second = write_traces(tmp_path / "b", rows)
        sha_a = freeze_steer_type(dataset_dir=first, root=tmp_path / "eval_a")
        sha_b = freeze_steer_type(dataset_dir=second, root=tmp_path / "eval_b")
        assert sha_a == sha_b
        assert (
            SteerTypeFrame.load(root=tmp_path / "eval_a").digest == SteerTypeFrame.load(root=tmp_path / "eval_b").digest
        )

    def test_build_rewrites_identical_bytes(self, tmp_path: Path) -> None:
        dataset = write_traces(tmp_path / "dataset", self.source_rows())
        one = build_steer_type_eval(dataset_dir=dataset).read_bytes()
        two = build_steer_type_eval(dataset_dir=dataset).read_bytes()
        assert one == two

    def test_freeze_records_dedup_counts_in_manifest(self, tmp_path: Path) -> None:
        dataset = write_traces(tmp_path / "dataset", self.source_rows())
        eval_dir = tmp_path / "eval"
        freeze_steer_type(dataset_dir=dataset, root=eval_dir)
        dedup = json.loads((eval_dir / MANIFEST_NAME).read_text())[f"{STEER_TYPE_EVAL_NAME}.meta"]["dedup"]
        assert dedup["dedup_n_in"] == 4.0
        assert dedup["dedup_n_removed"] == 1.0  # r0 and r3 render identical text and collapse to one
        assert dedup["dedup_n_semantic_removed"] == 0.0  # MinHash only, no embedder passed

    def test_role_marker_only_text_is_rejected(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset"
        (dataset / "steer_type").mkdir(parents=True)
        table = pa.table(
            {
                "id": ["good", "blank"],
                "text": ["<user>\nreal steer here", "<user>\n   "],
                "category": ["wrong_approach", "direction"],
                "is_steering": [True, True],
                "source_kind": ["transcript_message", "transcript_message"],
                "session_id": ["s0", "s1"],
                "split": ["test", "test"],
            }
        )
        pq.write_table(table, dataset / "steer_type" / "test.parquet")
        with pytest.raises(EmptyEvalContext) as excinfo:
            freeze_eval("steer_type", dataset_dir=dataset, root=tmp_path / "eval")
        assert excinfo.value.ids == ("blank",)
        assert not (tmp_path / "eval" / STEER_TYPE_EVAL_NAME).exists()

    def test_duplicate_ids_fail_loud(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset"
        (dataset / "steer_type").mkdir(parents=True)
        table = pa.table(
            {
                "id": ["dup", "dup"],
                "text": ["<user>\none", "<user>\ntwo"],
                "category": ["wrong_approach", "direction"],
                "is_steering": [True, True],
                "source_kind": ["transcript_message", "transcript_message"],
                "session_id": ["s0", "s1"],
                "split": ["test", "test"],
            }
        )
        pq.write_table(table, dataset / "steer_type" / "test.parquet")
        eval_dir = tmp_path / "eval"
        freeze_eval("steer_type", dataset_dir=dataset, root=eval_dir)
        with pytest.raises(SchemaError, match="duplicate row ids"):
            SteerTypeFrame.load(root=eval_dir)


class TestPickFrame:
    def source_rows(self) -> list[DecisionRow]:
        return [
            decision_row("p0", question="Which pin?", options=("3.13", "3.14", "3.15"), chosen=1),
            decision_row("p1", question="Vendor it?", options=("yes", "no"), chosen=0),
            # p2 renders the same ask as p0 and collapses in dedup.
            decision_row("p2", question="Which pin?", options=("3.13", "3.14", "3.15"), chosen=1),
            # excluded: wrong split.
            decision_row("p3", question="Ship it?", options=("now", "later"), chosen=0, split="train"),
        ]

    def excluded_rows(self) -> list[DecisionRow]:
        multi = replace(
            decision_row("m0", question="Pick many", options=("a", "b", "c"), chosen=0),
            multi_select=True,
            answer="a, b",
            chosen_index=(0, 1),
        )
        custom = replace(
            decision_row("c0", question="Off menu", options=("a", "b"), chosen=0),
            is_custom=True,
            chosen_index=(),
            answer="something else",
        )
        return [multi, custom]

    def decisions_path(self, tmp_path: Path, rows: list[DecisionRow]) -> Path:
        out = tmp_path / "decisions.parquet"
        write_decisions(MineResult(rows=tuple(rows), quarantined=()), out)
        return out

    def test_build_and_freeze_filters_and_dedups(self, tmp_path: Path) -> None:
        path = self.decisions_path(tmp_path, [*self.source_rows(), *self.excluded_rows()])
        eval_dir = tmp_path / "eval"
        freeze_pick(decisions_path=path, dataset_dir=tmp_path / "dataset", root=eval_dir)
        frame = PickFrame.load(root=eval_dir)
        assert len(frame) == 2  # p0/p2 collapse; train, multi-select, and off-menu rounds are dropped
        assert sorted(frame.chosen.tolist()) == [0, 1]
        assert sorted(frame.n_options.tolist()) == [2, 3]
        assert all("[assistant asked" in text for text in frame.texts)

    def test_freeze_is_deterministic_in_bytes_and_digest(self, tmp_path: Path) -> None:
        rows = self.source_rows()
        path_a = self.decisions_path(tmp_path / "a", rows)
        path_b = self.decisions_path(tmp_path / "b", rows)
        sha_a = freeze_pick(decisions_path=path_a, dataset_dir=tmp_path / "da", root=tmp_path / "ea")
        sha_b = freeze_pick(decisions_path=path_b, dataset_dir=tmp_path / "db", root=tmp_path / "eb")
        assert sha_a == sha_b
        assert PickFrame.load(root=tmp_path / "ea").digest == PickFrame.load(root=tmp_path / "eb").digest

    def test_text_is_the_rendered_ask(self, tmp_path: Path) -> None:
        path = self.decisions_path(
            tmp_path, [decision_row("only", question="Vendor it?", options=("yes", "no"), chosen=1)]
        )
        eval_dir = tmp_path / "eval"
        freeze_pick(decisions_path=path, dataset_dir=tmp_path / "dataset", root=eval_dir)
        frame = PickFrame.load(root=eval_dir)
        assert frame.texts[0] == ask_block("Vendor it?", header="Pick", options=("yes", "no"))
        assert int(frame.chosen[0]) == 1
