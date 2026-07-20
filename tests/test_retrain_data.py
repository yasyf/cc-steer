from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cc_steer.retrain.data import (
    WatcherRow,
    balance_no_steer,
    carve_val,
    dataset_digest,
    exact_text_overlap,
    load_train_rows,
    near_dup_indices,
    near_dup_representatives,
    oversample_corrective_to,
    train_digest,
    training_sample,
)

if TYPE_CHECKING:
    from pathlib import Path

LONG_A = "please refactor the authentication module to use the shared token cache in every request path"
LONG_B = "add a completely unrelated feature flag to the billing service configuration loader for tenants"
LONG_C = "rewrite the pagination cursor logic so that it never drops the final page of results anywhere"


def watcher_row(
    rid: str, *, label: bool = True, category: str = "wrong_approach", source_kind: str = "", content: str = LONG_A
) -> WatcherRow:
    return WatcherRow(
        id=rid,
        prompt=({"role": "user", "content": content},),
        reference="do the thing" if label else "NO_STEER",
        verbatim="do it" if label else "",
        label=label,
        category=category,
        source_kind=source_kind,
    )


def record(row: WatcherRow) -> dict[str, object]:
    return {
        "prompt": [dict(message) for message in row.prompt],
        "completion": [{"role": "assistant", "content": row.reference}],
        "verbatim": row.verbatim,
        "label": row.label,
        "id": row.id,
        "category": row.category,
        "source_kind": row.source_kind,
        "session_id": row.session_id,
        "split": "train",
    }


def write_train_parquet(root: Path, rows: list[WatcherRow]) -> Path:
    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    records = [record(row) for row in rows]
    table = pa.table(
        {
            "prompt": pa.array([r["prompt"] for r in records], type=pa.list_(message)),
            "completion": pa.array([r["completion"] for r in records], type=pa.list_(message)),
            "verbatim": [r["verbatim"] for r in records],
            "label": [r["label"] for r in records],
            "id": [r["id"] for r in records],
            "category": [r["category"] for r in records],
            "source_kind": [r["source_kind"] for r in records],
            "session_id": [r["session_id"] for r in records],
            "split": [r["split"] for r in records],
        }
    )
    (root / "watcher").mkdir(parents=True)
    path = root / "watcher" / "train.parquet"
    pq.write_table(table, path)
    return path


class TestNearDupRepresentatives:
    def test_identical_rows_collapse_to_one(self) -> None:
        rows = [watcher_row("a1"), watcher_row("a2"), watcher_row("a3"), watcher_row("b", content=LONG_B)]
        kept, stats = near_dup_representatives(rows)
        assert (stats.n_in, stats.n_kept, stats.n_removed) == (4, 2, 2)
        assert (stats.n_clusters, stats.n_multi_member_clusters) == (2, 1)
        assert 3 in kept
        assert len({0, 1, 2} & set(kept)) == 1

    def test_distinct_rows_all_survive(self) -> None:
        rows = [watcher_row("a", content=LONG_A), watcher_row("b", content=LONG_B), watcher_row("c", content=LONG_C)]
        kept, stats = near_dup_representatives(rows)
        assert kept == [0, 1, 2]
        assert (stats.n_kept, stats.n_removed) == (3, 0)

    def test_deterministic_in_seed(self) -> None:
        rows = [watcher_row(f"a{i}") for i in range(6)] + [watcher_row("b", content=LONG_B)]
        assert near_dup_representatives(rows, seed=1729)[0] == near_dup_representatives(rows, seed=1729)[0]

    def test_empty_input(self) -> None:
        kept, stats = near_dup_representatives([])
        assert kept == []
        assert stats.n_in == 0


class TestSemanticNearDup:
    # LONG_A/B/C share no char-5-gram, so MinHash keeps all three; the embedder maps A and B to the
    # same direction, so the semantic pass collapses them and reports the one marginal removal.
    def embedder(self, texts: list[str]) -> np.ndarray:
        direction = {LONG_A: [1.0, 0.0], LONG_B: [1.0, 0.001], LONG_C: [0.0, 1.0]}
        return np.array([direction[text] for text in texts])

    def test_minhash_only_keeps_all_and_reports_no_semantic(self) -> None:
        kept, stats = near_dup_indices([LONG_A, LONG_B, LONG_C])
        assert kept == [0, 1, 2]
        assert (stats.n_semantic_removed, stats.semantic_threshold) == (0, None)
        assert "dedup_semantic_threshold" not in stats.as_dict()

    def test_semantic_pass_collapses_paraphrases_beyond_minhash(self) -> None:
        kept, stats = near_dup_indices([LONG_A, LONG_B, LONG_C], embed=self.embedder, semantic_threshold=0.92)
        assert len(kept) == 2  # A and B collapse; C survives
        assert 2 in kept
        assert stats.n_semantic_removed == 1
        assert stats.as_dict()["dedup_n_semantic_removed"] == 1.0
        assert stats.as_dict()["dedup_semantic_threshold"] == 0.92

    def test_high_threshold_prunes_nothing_extra(self) -> None:
        kept, stats = near_dup_indices([LONG_A, LONG_B, LONG_C], embed=self.embedder, semantic_threshold=0.9999999)
        assert kept == [0, 1, 2]  # cosine(A, B) ~0.9999995 < threshold, so nothing extra is pruned
        assert stats.n_semantic_removed == 0


class TestExactTextOverlap:
    def test_strip_normalized_overlap(self) -> None:
        assert exact_text_overlap(["a", "b ", " c"], ["b", "d", "c"]) == ["b", "c"]

    def test_disjoint_is_empty(self) -> None:
        assert exact_text_overlap(["x", "y"], ["z"]) == []

    def test_blank_texts_never_leak(self) -> None:
        assert exact_text_overlap(["   ", ""], ["  ", "\t"]) == []


class TestBalanceNoSteer:
    def test_balances_and_reports_ratio(self) -> None:
        rows = [watcher_row(f"p{i}") for i in range(30)] + [watcher_row(f"n{i}", label=False) for i in range(10)]
        balanced, ratio = balance_no_steer(rows, seed=7)
        assert ratio == 3.0
        labels = [r.label for r in balanced]
        assert labels.count(True) == labels.count(False) == 30
        assert {r.id for r in rows} <= {r.id for r in balanced}

    def test_already_balanced_is_untouched(self) -> None:
        rows = [watcher_row("p"), watcher_row("n", label=False)]
        balanced, ratio = balance_no_steer(rows, seed=7)
        assert sorted(r.id for r in balanced) == ["n", "p"]
        assert ratio == 1.0


class TestCarveVal:
    def test_stratified_and_disjoint(self) -> None:
        rows = [watcher_row(f"p{i}") for i in range(60)] + [watcher_row(f"n{i}", label=False) for i in range(40)]
        val, rest = carve_val(rows, n=10, seed=3)
        assert len(val) == 10
        assert {r.id for r in val}.isdisjoint({r.id for r in rest})
        assert len(val) + len(rest) == len(rows)
        assert sum(r.label for r in val) == 6

    def test_deterministic(self) -> None:
        rows = [watcher_row(f"p{i}") for i in range(30)] + [watcher_row(f"n{i}", label=False) for i in range(30)]
        assert [r.id for r in carve_val(rows, n=8, seed=11)[0]] == [r.id for r in carve_val(rows, n=8, seed=11)[0]]


class TestOversampleCorrectiveTo:
    def test_lifts_direction_clamp_to_factor(self) -> None:
        corrective = [watcher_row(f"c{i}") for i in range(10)]
        rows = corrective + [watcher_row(f"d{i}", category="direction") for i in range(20)]
        out, before, after = oversample_corrective_to(rows, factor=6.0, seed=1729)
        assert (before, after) == (10, 60)
        assert sum(1 for r in out if r.label and r.category != "direction") == 60
        assert sum(1 for r in out if r.category == "direction") == 20

    def test_factor_three_matches_baseline_count(self) -> None:
        corrective = [watcher_row(f"c{i}") for i in range(10)]
        rows = corrective + [watcher_row(f"d{i}", category="direction") for i in range(20)]
        assert oversample_corrective_to(rows, factor=3.0, seed=1729)[2] == 30

    def test_no_corrective_is_noop(self) -> None:
        rows = [watcher_row(f"d{i}", category="direction") for i in range(4)]
        out, before, after = oversample_corrective_to(rows, factor=6.0, seed=1729)
        assert (before, after, len(out)) == (0, 0, 4)


class TestTrainingSample:
    def test_shape_and_content(self) -> None:
        row = WatcherRow(
            id="r",
            prompt=({"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}),
            reference="do the thing",
            verbatim="do it",
            label=True,
            category="direction",
        )
        sample = training_sample(row, system="SYS", cap=1000)
        assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
        assert sample["messages"][0]["content"] == "SYS"
        assert sample["messages"][1]["content"] == "<user>\nhello\n\n<assistant>\nworld"
        assert sample["messages"][2]["content"] == "do the thing"


class TestDatasetDigest:
    def test_same_rows_same_digest(self) -> None:
        rows = [record(watcher_row(f"r{i}", label=i % 2 == 0)) for i in range(5)]
        assert dataset_digest(rows) == dataset_digest(rows)

    def test_order_invariant(self) -> None:
        rows = [record(watcher_row(f"r{i}")) for i in range(4)]
        assert dataset_digest(rows) == dataset_digest(list(reversed(rows)))
        assert len(dataset_digest(rows)) == 16

    def test_one_changed_row_changes_digest(self) -> None:
        rows = [record(watcher_row(f"r{i}")) for i in range(4)]
        changed = [*rows[:-1], record(watcher_row("r3", content="a wholly different context window here now"))]
        assert dataset_digest(rows) != dataset_digest(changed)


class TestLoadTrainRows:
    def test_round_trip_from_parquet(self, tmp_path: Path) -> None:
        rows = [watcher_row("p0"), watcher_row("q1", source_kind="question_answer"), watcher_row("n2", label=False)]
        write_train_parquet(tmp_path, rows)
        loaded = load_train_rows(dataset_dir=tmp_path)
        assert [r.id for r in loaded] == ["p0", "q1", "n2"]
        assert loaded[1].source_kind == "question_answer"
        assert loaded[2].label is False

    def test_train_digest_matches_table_digest(self, tmp_path: Path) -> None:
        rows = [watcher_row("p0"), watcher_row("n1", label=False)]
        write_train_parquet(tmp_path, rows)
        table = pq.read_table(tmp_path / "watcher" / "train.parquet")
        assert train_digest(dataset_dir=tmp_path) == dataset_digest(table.to_pylist())

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_train_rows(dataset_dir=tmp_path)


class TestPoolDeterminism:
    def pool(self) -> list[WatcherRow]:
        # A distinct dominant char per row keeps pairwise Jaccard well under 0.8, so dedup keeps them all.
        def content(k: int) -> str:
            return chr(97 + k) * 120

        corrective = [watcher_row(f"c{i}", content=content(i)) for i in range(4)]
        direction = [watcher_row(f"d{i}", category="direction", content=content(4 + i)) for i in range(6)]
        negatives = [watcher_row(f"n{i}", label=False, content=content(10 + i)) for i in range(3)]
        return corrective + direction + negatives

    def build(self, seed: int) -> list[str]:
        rows = self.pool()
        kept, _ = near_dup_representatives(rows, seed=seed)
        deduped = [rows[i] for i in kept]
        balanced, _ = balance_no_steer(deduped, seed=seed)
        oversampled, _, _ = oversample_corrective_to(balanced, factor=6.0, seed=seed)
        return [r.id for r in oversampled]

    def test_pipeline_is_deterministic(self) -> None:
        assert self.build(1729) == self.build(1729)  # ordered, so shuffle nondeterminism would surface

    def test_exact_row_multiset(self) -> None:
        counts = Counter(self.build(1729))
        # 4 distinct corrective rows survive dedup; each is oversampled to 6x (24 copies).
        assert sum(counts[f"c{i}"] for i in range(4)) == 24
        assert all(counts[f"c{i}"] == 6 for i in range(4))
        # Every direction and negative row is kept exactly once (dedup keeps distinct, oversample skips them).
        assert all(counts[f"d{i}"] == 1 for i in range(6))
        # 10 positives (4 corrective + 6 direction) balance against 3 negatives -> 7 oversampled negative copies added.
        assert sum(counts[f"n{i}"] for i in range(3)) == 10
