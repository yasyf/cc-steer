from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from athome.research.golden import GoldenGateViolation

from cc_steer import registry
from cc_steer.retrain import evalset, golden, judged

if TYPE_CHECKING:
    from pathlib import Path

MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string())])
BUDGET = 0.5


def watcher_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.table(
        {
            "prompt": pa.array([[{"role": "user", "content": row["content"]}] for row in rows], type=pa.list_(MESSAGE)),
            "completion": pa.array(
                [[{"role": "assistant", "content": "steer" if row["label"] else "NO_STEER"}] for row in rows],
                type=pa.list_(MESSAGE),
            ),
            "verbatim": ["v" if row["label"] else "" for row in rows],
            "label": [row["label"] for row in rows],
            "id": [row["id"] for row in rows],
            "category": [row["category"] for row in rows],
            "source_kind": [row["source_kind"] for row in rows],
            "session_id": [f"s{i}" for i in range(len(rows))],
            "split": ["test"] * len(rows),
        }
    )


def warranted(index: int, *, content: str | None = None, prob: float = 0.4) -> dict[str, object]:
    return {"id": f"w{index}", "label": True, "category": "wrong_approach", "source_kind": "", "content": content or f"WARRANT=yes context w{index}", "prob": prob}


def negative(index: int, *, prob: float = 0.7) -> dict[str, object]:
    return {"id": f"n{index}", "label": False, "category": "", "source_kind": "", "content": f"WARRANT=no context n{index}", "prob": prob}


def other_positive(index: int, *, prob: float) -> dict[str, object]:
    return {"id": f"o{index}", "label": True, "category": "direction", "source_kind": "", "content": f"direction context o{index}", "prob": prob}


def rich_rows() -> list[dict[str, object]]:
    # 20 warranted, 20 prose-negatives, 3 other-positives that fire (p < budget), 3 that do not.
    return (
        [warranted(i) for i in range(20)]
        + [negative(i) for i in range(20)]
        + [other_positive(i, prob=0.1) for i in range(3)]
        + [other_positive(i, prob=0.9) for i in range(3, 6)]
    )


def setup_incumbent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> tuple[Path, str]:
    dataset = tmp_path / "dataset"
    (dataset / "watcher").mkdir(parents=True)
    pq.write_table(watcher_table(rows), dataset / "watcher" / "test.parquet")
    eval_dir = tmp_path / "eval"
    evalset.freeze_eval("watcher", dataset_dir=dataset, root=eval_dir)
    frame = evalset.EvalFrame.load(root=eval_dir)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    incumbent = registry.register(
        golden.WATCHER_COMPONENT,
        {"adapter.bin": b"stub"},
        {"base_model": "mlx", "render_version": 2, "thresholds": {"budget": BUDGET}, "dataset_digest": "d"},
    )
    registry.promote(golden.WATCHER_COMPONENT, incumbent.version)
    evalset.write_probs(frame, incumbent.version, {row["id"]: float(row["prob"]) for row in rows}, auc=0.5, root=eval_dir)
    return eval_dir, incumbent.version


def fill_labels(eval_dir: Path) -> None:
    directory = judged.golden_dir(root=eval_dir)
    stratum = {row["row_id"]: row["stratum"] for row in json.loads((directory / "manifest.json").read_text())["rows"]}
    (directory / golden.LABELS_NAME).write_text(
        json.dumps(
            [
                {"row": entry["row"], "row_id": entry["row_id"], "label": "yes" if stratum[entry["row_id"]] == golden.WARRANTED else "no"}
                for entry in json.loads((directory / "labels_template.json").read_text())
            ]
        )
    )


class TestAuditSampleAndPools:
    def test_pools_are_disjoint_and_priority_ordered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, _version = setup_incumbent(tmp_path, monkeypatch, rich_rows())
        frame = evalset.EvalFrame.load(root=eval_dir)
        _version, fire, threshold = golden.incumbent_fire(frame, root=eval_dir)
        pools = golden.audit_pools(frame, fire, threshold)
        assert {name: len(pools[name]) for name in golden.STRATUM_NAMES} == {"warranted": 20, "fired": 3, "negative": 20, "other-positive": 3}
        stacked = np.concatenate([pools[name] for name in golden.STRATUM_NAMES])
        assert len(set(stacked.tolist())) == len(stacked)  # every row lands in exactly one pool

    def test_sample_is_deterministic_and_take_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, _version = setup_incumbent(tmp_path, monkeypatch, rich_rows())
        frame = evalset.EvalFrame.load(root=eval_dir)
        _version, fire, threshold = golden.incumbent_fire(frame, root=eval_dir)
        first = golden.audit_sample(frame, fire, threshold)
        assert golden.stratum_counts(first) == {"warranted": 20, "fired": 3, "negative": 20, "other-positive": 3}
        assert [row.row_id for row in first] == [row.row_id for row in golden.audit_sample(frame, fire, threshold)]

    def test_negative_draw_never_starves_the_packet_on_a_warranted_heavy_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [warranted(index) for index in range(185)] + [negative(200 + index) for index in range(20)]
        eval_dir, _version = setup_incumbent(tmp_path, monkeypatch, rows)
        frame = evalset.EvalFrame.load(root=eval_dir)
        _version, fire, threshold = golden.incumbent_fire(frame, root=eval_dir)
        counts = golden.stratum_counts(golden.audit_sample(frame, fire, threshold))
        assert counts["warranted"] == 185
        assert counts["negative"] == 15  # floored at the packet's negative stratum, not round(15 * 0.64) = 10


class TestAuthorPacket:
    @pytest.mark.anyio
    async def test_author_stamps_gate_audit_and_round_trips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, version = setup_incumbent(tmp_path, monkeypatch, rich_rows())
        directory = await golden.author_packet(root=eval_dir)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["gate"] == {"n": 30, "floor": 24}
        assert manifest["audit"] == {
            "seed": 1729,
            "n": 200,
            "incumbent_version": version,
            "incumbent_fire_threshold": 0.5,
            "stratum_counts": {"warranted": 20, "fired": 3, "negative": 20, "other-positive": 3},
        }
        assert (directory / "packet.md").read_text().count("## Row ") == 30
        assert len((directory / golden.FIRES_NAME).read_text().splitlines()) == 30
        assert (directory / golden.README_NAME).exists()

    @pytest.mark.anyio
    async def test_verify_loads_through_the_real_judged_gate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, _version = setup_incumbent(tmp_path, monkeypatch, rich_rows())
        await golden.author_packet(root=eval_dir)
        fill_labels(eval_dir)
        loaded = await golden.verify_golden(root=eval_dir)
        assert len(loaded.human) == 30
        assert len(loaded.contexts) == 30
        # Every drawn row's bound window is byte-identical to the frame tail it was authored from.
        frame = evalset.EvalFrame.load(root=eval_dir)
        tail_of = dict(zip(frame.ids, frame.tails, strict=True))
        assert all(loaded.contexts[row_id] == tail_of[row_id] for row_id in loaded.contexts)

    @pytest.mark.anyio
    async def test_refuses_to_overwrite_a_labeled_packet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, _version = setup_incumbent(tmp_path, monkeypatch, rich_rows())
        await golden.author_packet(root=eval_dir)
        fill_labels(eval_dir)
        with pytest.raises(golden.GoldenAuthorError, match="refusing to overwrite a labeled packet"):
            await golden.author_packet(root=eval_dir)

    @pytest.mark.anyio
    async def test_round_trip_catches_a_fence_collision_in_a_window(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [warranted(i, prob=0.9) for i in range(14)] + [warranted(99, content="before\n~~~\nafter", prob=0.9)] + [negative(i, prob=0.9) for i in range(15)]
        eval_dir, _version = setup_incumbent(tmp_path, monkeypatch, rows)
        with pytest.raises(GoldenGateViolation, match="does not round-trip"):
            await golden.author_packet(root=eval_dir)
