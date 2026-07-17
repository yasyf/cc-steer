from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from athome.llm.spend import SpendExceeded
from athome.train import (
    Adapter,
    EvalRow,
    InsufficientData,
    RetrainOutcome,
    SavedCheckpoint,
    ScoredSequence,
    TrainReport,
)

from cc_steer import launchd, registry
from cc_steer.retrain import data, evalset, promotion, sentinel
from cc_steer.retrain import watcher as w
from cc_steer.retrain.evalset import ProbsStoreError
from cc_steer.retrain.watcher import FreshEpochError, WatcherRecipe, WatcherRetrainError
from cc_steer.watcher import drafter_mlx

if TYPE_CHECKING:
    from pathlib import Path

N = 20
CANDIDATE_WIN = np.full(N, 0.9)
CANDIDATE_WIN[:7] = np.linspace(0.01, 0.07, 7)
CANDIDATE_WIN[7:10] = [0.08, 0.09, 0.10]
CANDIDATE_WIN[10:] = np.linspace(0.5, 0.99, 10)
INCUMBENT = np.full(N, 0.9)
INCUMBENT[10:17] = 0.1


def default_kwargs() -> dict[str, Any]:
    return {
        "tinker_model": "Qwen/Qwen3-8B",
        "mlx_id": "mlx-community/Qwen3-8B-4bit",
        "rank": 32,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 2,
        "checkpoint_fracs": [0.25, 0.5, 0.75, 1.0],
        "max_tokens": 4096,
        "render_version": 2,
        "val_n": 200,
        "oversample_corrective": 3.0,
        "budget_fires_per_100": 2.0,
        "spend_cap_usd": 15.0,
        "diagnostic_rows": 20,
        "diagnostic_tolerance": 0.05,
        "seed": 1729,
    }


class TestWatcherRecipe:
    def test_packaged_default_exact_values(self) -> None:
        recipe = WatcherRecipe.default()
        assert recipe == WatcherRecipe(**default_kwargs())
        assert recipe.checkpoint_fracs == (0.25, 0.5, 0.75, 1.0)
        assert recipe.rank == 32
        assert recipe.batch_size == 16
        assert recipe.spend_cap_usd == 15.0
        assert recipe.seed == 1729

    def test_json_round_trip_coerces_fracs_to_tuple(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps(default_kwargs()))
        recipe = WatcherRecipe.from_json(path)
        assert recipe == WatcherRecipe(**default_kwargs())
        assert isinstance(recipe.checkpoint_fracs, tuple)

    def test_missing_key_crashes(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps({k: v for k, v in default_kwargs().items() if k != "seed"}))
        with pytest.raises(TypeError):
            WatcherRecipe.from_json(path)

    def test_extra_key_crashes(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps(default_kwargs() | {"bogus": 1}))
        with pytest.raises(TypeError):
            WatcherRecipe.from_json(path)

    @pytest.mark.parametrize(
        "override",
        [
            {"spend_cap_usd": float("nan")},
            {"spend_cap_usd": 0.0},
            {"learning_rate": 0.0},
            {"learning_rate": -1e-4},
            {"learning_rate": float("nan")},
            {"rank": 0},
            {"batch_size": 0},
            {"epochs": 0},
            {"max_tokens": 0},
            {"val_n": 0},
            {"diagnostic_rows": 0},
            {"oversample_corrective": 0.5},
            {"oversample_corrective": float("nan")},
            {"oversample_corrective": float("inf")},
            {"budget_fires_per_100": 0.0},
            {"budget_fires_per_100": -2.0},
            {"budget_fires_per_100": float("nan")},
            {"diagnostic_tolerance": 0.0},
            {"diagnostic_tolerance": 1.0},
            {"checkpoint_fracs": [0.0, 1.0]},
            {"checkpoint_fracs": [0.5, 1.5]},
            {"render_version": 1},
            {"render_version": 3},
            {"render_version": 2.0},
            {"rank": float("nan")},
            {"seed": 17.29},
        ],
        ids=[
            "nan-cap",
            "zero-cap",
            "zero-lr",
            "negative-lr",
            "nan-lr",
            "rank-0",
            "batch-size-0",
            "epochs-0",
            "max-tokens-0",
            "val-n-0",
            "diagnostic-rows-0",
            "oversample-lt-1",
            "oversample-nan",
            "oversample-inf",
            "budget-0",
            "budget-negative",
            "budget-nan",
            "diagnostic-tol-0",
            "diagnostic-tol-ge-1",
            "frac-0",
            "frac-gt-1",
            "render-1",
            "render-3",
            "render-float-2",
            "rank-nan",
            "seed-float",
        ],
    )
    def test_validation_rejects_degenerate(self, override: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            WatcherRecipe(**(default_kwargs() | override))


def test_register_watcher_adapter_requires_budget_threshold(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / drafter_mlx.ADAPTER_NAME).write_bytes(b"weights")
    (adapter / drafter_mlx.ADAPTER_CONFIG_NAME).write_bytes(b"{}")
    with pytest.raises(ValueError, match="'budget' operating point"):
        w.register_watcher_adapter(
            adapter,
            metadata={"base_model": "m", "render_version": 2, "thresholds": {"f1": 0.2}},
            root=tmp_path / "registry",
        )


def message_struct() -> pa.DataType:
    return pa.struct([("role", pa.string()), ("content", pa.string())])


def watcher_row(index: int, *, split: str, label: bool, category: str, source_kind: str) -> dict[str, Any]:
    reference = f"steer direction {index}" if label else "NO_STEER"
    return {
        "id": f"{split}-{index}",
        "prompt": [{"role": "user", "content": f"{split} window {index}: should I refactor module {index}?"}],
        "completion": [{"role": "assistant", "content": reference}],
        "verbatim": reference if label else "",
        "label": label,
        "category": category,
        "source_kind": source_kind,
        "session_id": f"s{index}",
        "split": split,
    }


def watcher_table(rows: list[dict[str, Any]]) -> pa.Table:
    struct = message_struct()
    return pa.table(
        {
            "prompt": pa.array([r["prompt"] for r in rows], type=pa.list_(struct)),
            "completion": pa.array([r["completion"] for r in rows], type=pa.list_(struct)),
            "verbatim": [r["verbatim"] for r in rows],
            "label": [r["label"] for r in rows],
            "id": [r["id"] for r in rows],
            "category": [r["category"] for r in rows],
            "source_kind": [r["source_kind"] for r in rows],
            "session_id": [r["session_id"] for r in rows],
            "split": [r["split"] for r in rows],
        }
    )


def eval_rows() -> list[dict[str, Any]]:
    positives = [
        watcher_row(i, split="test", label=True, category="wrong_approach", source_kind="transcript_message")
        for i in range(10)
    ]
    negatives = [
        watcher_row(i + 10, split="test", label=False, category="", source_kind="transcript_message")
        for i in range(10)
    ]
    return positives + negatives


def train_rows() -> list[dict[str, Any]]:
    positives = [
        watcher_row(i, split="train", label=True, category="wrong_approach", source_kind="transcript_message")
        for i in range(8)
    ]
    negatives = [
        watcher_row(i + 8, split="train", label=False, category="", source_kind="transcript_message")
        for i in range(8)
    ]
    return positives + negatives


@dataclass
class Lane:
    dataset_dir: Path
    eval_dir: Path
    registry_root: Path
    state_dir: Path
    frame: evalset.EvalFrame
    incumbent: registry.VersionInfo
    calls: dict[str, int] = field(
        default_factory=lambda: {"retrain": 0, "materialize": 0, "score": 0, "score_local": 0, "kickstart": 0}
    )
    candidate: np.ndarray = field(default_factory=lambda: CANDIDATE_WIN.copy())
    tinker: np.ndarray | None = None
    drafter_offset: float = 0.0
    train_error: Exception | None = None
    materialize_error: Exception | None = None
    diagnostic_error: Exception | None = None
    kickstart_result: bool = True

    def run(
        self, *, force: bool = True, fresh_epoch: bool = False, recipe: WatcherRecipe = WatcherRecipe.default()
    ) -> str:
        return w.retrain_watcher(
            force=force,
            fresh_epoch=fresh_epoch,
            recipe=recipe,
            dataset_dir=self.dataset_dir,
            eval_root=self.eval_dir,
            registry_root=self.registry_root,
            state_dir=self.state_dir,
            adapters_dir=self.state_dir / "adapters",
        )


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "watcher").mkdir(parents=True)
    pq.write_table(watcher_table(train_rows()), dataset_dir / "watcher" / "train.parquet")
    pq.write_table(watcher_table(eval_rows()), dataset_dir / "watcher" / "test.parquet")

    eval_dir = tmp_path / "eval"
    evalset.freeze_eval("watcher", dataset_dir=dataset_dir, root=eval_dir)
    frame = evalset.EvalFrame.load(root=eval_dir)

    registry_root = tmp_path / "models"
    incumbent = registry.register(
        w.WATCHER_COMPONENT,
        {drafter_mlx.ADAPTER_NAME: b"inc", drafter_mlx.ADAPTER_CONFIG_NAME: b"{}"},
        {"base_model": "mlx-community/Qwen3-8B-4bit", "render_version": 2, "thresholds": {"budget": 0.5},
         "dataset_digest": "old-digest"},
        root=registry_root,
    )
    registry.promote(w.WATCHER_COMPONENT, incumbent.version, root=registry_root)
    evalset.write_probs(
        frame, incumbent.version, {rid: float(INCUMBENT[i]) for i, rid in enumerate(frame.ids)}, auc=0.4, root=eval_dir
    )

    obj = Lane(dataset_dir, eval_dir, registry_root, tmp_path / "state", frame, incumbent)
    tail_index = {frame.tails[i]: i for i in range(len(frame))}

    def fake_eval_row(system: str, user: str, mlx_id: str) -> EvalRow:
        # Encode the frame index in tokens[0] so the fake backend.score can map a diagnostic row back
        # to its reference prob; val rows (not in the frame) fall back to 0 and are never scored.
        return EvalRow(tokens=(tail_index.get(user, 0),), weights=(1.0,))

    async def fake_retrain(
        backend: Any,
        spec: Any,
        *,
        checkpoints: Any,
        eval_rows: Any,
        select: Any,
        artifact_scorer: Any,
        gate: Any,
        work_dir: Path,
        sink: Any,
    ) -> RetrainOutcome:
        obj.calls["retrain"] += 1
        if obj.train_error is not None:
            raise obj.train_error
        steps = spec.hyperparams.steps
        saved = tuple(
            SavedCheckpoint(
                step=step,
                path=f"tinker://ckpt/{step}",
                final=(step == steps),
                scores=tuple(
                    ScoredSequence(logprob=-0.01 * (i + 1) - 0.001 * step, weight=1.0) for i in range(len(eval_rows))
                ),
            )
            for step in (*checkpoints.steps_for(steps), steps)
        )
        best = max(saved, key=select)
        if obj.materialize_error is not None:
            raise obj.materialize_error
        adapter_dir = work_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / drafter_mlx.ADAPTER_NAME).write_bytes(b"cand")
        (adapter_dir / drafter_mlx.ADAPTER_CONFIG_NAME).write_text("{}")
        obj.calls["materialize"] += 1
        adapter = Adapter(step=best.step, adapter_dir=adapter_dir, train_cost_usd=1.23)
        served = artifact_scorer(adapter)
        verdict = gate(served)
        report = TrainReport(method="sft", steps=(), checkpoints=saved, dropped=0, wall_s=0.0, train_cost_usd=1.23)
        return RetrainOutcome(report=report, best=best, adapter=adapter, served=served, verdict=verdict)

    class FakeBackend:
        @classmethod
        def from_settings(cls) -> FakeBackend:
            return cls()

        async def score(
            self, path: str, rows: Any, *, base: Any, max_usd: float | None = None
        ) -> tuple[ScoredSequence, ...]:
            obj.calls["score"] += 1
            if obj.diagnostic_error is not None:
                raise obj.diagnostic_error
            reference = obj.tinker if obj.tinker is not None else obj.candidate
            return tuple(ScoredSequence(logprob=math.log(float(reference[row.tokens[0]])), weight=1.0) for row in rows)

    class FakeDrafter:
        def __init__(
            self, version: Any = None, *, threshold: Any = None, root: Any = None, operating_point: str = "budget"
        ) -> None:
            self.version = version
            obj.calls["score_local"] += 1

        def nosteer_prob(self, tail: str) -> float:
            return float(obj.candidate[tail_index[tail]]) + obj.drafter_offset

        def clear_cache(self) -> None:
            pass

    def fake_kickstart() -> bool:
        obj.calls["kickstart"] += 1
        return obj.kickstart_result

    monkeypatch.setattr(w, "load_key", lambda **_: None)
    monkeypatch.setattr(w, "TinkerBackend", FakeBackend)
    monkeypatch.setattr(w, "athome_retrain", fake_retrain)
    monkeypatch.setattr(sentinel, "sentinel_eval_row", fake_eval_row)
    monkeypatch.setattr(drafter_mlx, "MlxDrafter", FakeDrafter)
    monkeypatch.setattr(launchd, "kickstart_watch", fake_kickstart)
    return obj


def sidecar_path(lane: Lane) -> Path:
    matches = sorted((lane.state_dir / "adapters").glob(f"watcher-*/{w.DIAGNOSTIC_NAME}"))
    assert len(matches) == 1, f"expected exactly one serving diagnostic, got {matches}"
    return matches[0]


class TestRetrainWatcher:
    def test_skip_when_digest_unchanged(self, lane: Lane) -> None:
        digest = data.train_digest(dataset_dir=lane.dataset_dir)
        matched = registry.register(  # a fresh incumbent whose digest matches the current train view
            w.WATCHER_COMPONENT,
            {drafter_mlx.ADAPTER_NAME: b"inc2", drafter_mlx.ADAPTER_CONFIG_NAME: b"{}"},
            lane.incumbent.metadata | {"dataset_digest": digest},
            root=lane.registry_root,
        )
        registry.promote(w.WATCHER_COMPONENT, matched.version, root=lane.registry_root)
        verdict = lane.run(force=False)
        assert verdict == f"watcher: skipped (no new data at digest {digest})"
        assert lane.calls["retrain"] == 0
        assert lane.calls["score_local"] == 0

    def test_promote_registers_prunes_writes_probs_and_kickstarts(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for stub in (b"stub-a", b"stub-b"):  # pad past KEEP_VERSIONS so the post-promote prune drops the oldest
            registry.register(
                w.WATCHER_COMPONENT,
                {drafter_mlx.ADAPTER_NAME: stub, drafter_mlx.ADAPTER_CONFIG_NAME: b"{}"},
                lane.incumbent.metadata,
                root=lane.registry_root,
            )
        order: list[str] = []

        def recorder(name: str, fn: Any) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                order.append(name)
                return fn(*args, **kwargs)

            return wrapped

        monkeypatch.setattr(registry, "register", recorder("register", registry.register))
        monkeypatch.setattr(registry, "promote", recorder("promote", registry.promote))
        monkeypatch.setattr(evalset, "write_probs", recorder("write_probs", evalset.write_probs))
        monkeypatch.setattr(launchd, "kickstart_watch", recorder("kickstart", launchd.kickstart_watch))

        verdict = lane.run()
        assert verdict.startswith("watcher: promoted")
        assert order == ["register", "promote", "write_probs", "kickstart"]
        assert lane.calls["kickstart"] == 1
        current = registry.current(w.WATCHER_COMPONENT, root=lane.registry_root)
        assert current is not None and current.version != lane.incumbent.version
        # FH1: thresholds["budget"] is the P(NO_STEER)-scale cut 1 - fire_score_cut (~0.01 here), not the 0.99 fire cut.
        expected_threshold = 1.0 - promotion.threshold_for_budget(
            1.0 - CANDIDATE_WIN,
            fires_per_100=w.WatcherRecipe.default().budget_fires_per_100,
            total_turns=len(lane.frame),
        )
        assert current.metadata["thresholds"]["budget"] == pytest.approx(expected_threshold)
        assert current.metadata["thresholds"]["budget"] < 0.5  # p-scale, not the fire-score scale
        assert current.metadata["render_version"] == 2
        assert "tinker_checkpoint" in current.metadata
        assert "diagnostic_max_abs_diff" in current.metadata
        remaining = [info.version for info in registry.versions(w.WATCHER_COMPONENT, root=lane.registry_root)]
        assert len(remaining) == w.KEEP_VERSIONS
        assert lane.incumbent.version not in remaining  # the oldest version was the one pruned
        assert evalset.probs_path(current.version, root=lane.eval_dir).exists()

    def test_reject_at_gate_after_materialize_never_registers(self, lane: Lane) -> None:
        # The gate scores through the materialized artifact, so a reject pays one materialize + local
        # scoring pass + diagnostic, but never registers a new version or kicks the daemon.
        lane.candidate = INCUMBENT.copy()  # served AUC ties the incumbent -> the watcher bar rejects
        verdict = lane.run()
        assert verdict.startswith("watcher: rejected")
        assert lane.calls["materialize"] == 1
        assert lane.calls["score_local"] == 1
        assert lane.calls["score"] == 1  # the diagnostic still runs, before the verdict acts
        assert lane.calls["kickstart"] == 0
        assert len(registry.versions(w.WATCHER_COMPONENT, root=lane.registry_root)) == 1
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version

    def test_gate_reads_the_served_channel_not_tinker(self, lane: Lane) -> None:
        # Served (local) probs tie the incumbent while Tinker would win outright: the gate follows
        # what we serve and rejects, ignoring the stronger Tinker frame.
        lane.candidate = INCUMBENT.copy()
        lane.tinker = CANDIDATE_WIN.copy()
        assert lane.run().startswith("watcher: rejected")
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version

    def test_gate_promotes_on_served_win_even_when_tinker_ties(self, lane: Lane) -> None:
        # The inverse: served wins while Tinker only ties the incumbent. The gate promotes on served.
        lane.candidate = CANDIDATE_WIN.copy()
        lane.tinker = INCUMBENT.copy()
        assert lane.run().startswith("watcher: promoted")
        current = registry.current(w.WATCHER_COMPONENT, root=lane.registry_root)
        assert current is not None and current.version != lane.incumbent.version

    def test_diagnostic_sidecar_records_drift_on_promote(self, lane: Lane) -> None:
        lane.tinker = CANDIDATE_WIN.copy()
        lane.tinker[0] += 0.2  # index 0 drifts 0.2 from served, index 1 drifts 0.1; both over the 0.05 tolerance
        lane.tinker[1] += 0.1
        verdict = lane.run()
        assert verdict.startswith("watcher: promoted")
        payload = json.loads(sidecar_path(lane).read_text())
        assert len(payload["rows"]) == N
        assert payload["rows"][0]["index"] == 0 and payload["rows"][0]["abs_diff"] == pytest.approx(0.2)
        assert payload["rows"][1]["index"] == 1 and payload["rows"][1]["abs_diff"] == pytest.approx(0.1)
        assert payload["summary"]["diagnostic_max_abs_diff"] == pytest.approx(0.2)
        assert payload["summary"]["diagnostic_over_tolerance"] == 2.0
        current = registry.current(w.WATCHER_COMPONENT, root=lane.registry_root)
        assert current.metadata["diagnostic_over_tolerance"] == 2.0

    def test_diagnostic_sidecar_persists_on_reject(self, lane: Lane) -> None:
        # Per-row serving evidence survives a reject: the sidecar is written before the gate decides,
        # closing the hole where write_probs (post-registration) discarded it on rejection.
        lane.candidate = INCUMBENT.copy()
        assert lane.run().startswith("watcher: rejected")
        payload = json.loads(sidecar_path(lane).read_text())
        assert len(payload["rows"]) == N
        assert payload["summary"]["diagnostic_over_tolerance"] == 0.0  # served == tinker here, no drift

    def test_diagnostic_failure_leaves_promotion_untouched(self, lane: Lane) -> None:
        # The diagnostic is observability, never a gate: a Tinker/API failure inside it must record
        # itself and leave a clean promote alone, not crash it into an unjournaled outcome.
        lane.diagnostic_error = RuntimeError("Tinker returned no log probability for the sentinel token")
        verdict = lane.run()
        assert verdict.startswith("watcher: promoted")
        assert "serving diagnostic failed" in verdict  # the reason is recorded on the verdict
        current = registry.current(w.WATCHER_COMPONENT, root=lane.registry_root)
        assert current is not None and current.version != lane.incumbent.version
        assert current.metadata["diagnostic_failed"] == 1.0
        entry = json.loads((lane.state_dir / "retrain" / "journal.jsonl").read_text().splitlines()[-1])
        assert entry["verdict"].startswith("promoted") and "serving diagnostic failed" in entry["verdict"]
        assert entry["metrics"]["diagnostic_failed"] == 1.0

    def test_score_frame_local_maps_each_row_to_its_served_prob(self, lane: Lane) -> None:
        # Heterogeneous per-row values: a within-stratum permutation would change this mapping, so
        # it guards the stored per-row probs that future paired incumbent comparisons read by row id.
        lane.candidate = np.linspace(0.02, 0.97, N)
        probs = w._score_frame_local(lane.frame, lane.state_dir, WatcherRecipe.default())
        assert probs == pytest.approx({rid: float(lane.candidate[i]) for i, rid in enumerate(lane.frame.ids)})

    def test_spend_cap_refusal_journaled_as_reject(self, lane: Lane) -> None:
        lane.train_error = SpendExceeded("projected $99.0000 exceeds cap $15.0000")
        verdict = lane.run()
        assert verdict.startswith("watcher: rejected (projected")
        assert lane.calls["score"] == 0
        assert lane.calls["score_local"] == 0
        assert lane.calls["materialize"] == 0
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version

    def test_insufficient_data_journaled_as_reject(self, lane: Lane) -> None:
        lane.train_error = InsufficientData(3, 16)
        verdict = lane.run()
        assert verdict.startswith("watcher: rejected")
        assert lane.calls["materialize"] == 0
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version

    def test_materialize_failure_aborts_before_register(self, lane: Lane) -> None:
        # An unfusable-tensor failure during materialize raises out of retrain and must abort before any
        # register, so a half-converted adapter is never promoted.
        lane.materialize_error = ValueError("mlx-lm cannot fuse 1 trained tensor(s): ['linear_attn.in_proj_qkv']")
        with pytest.raises(ValueError, match="cannot fuse"):
            lane.run()
        assert len(registry.versions(w.WATCHER_COMPONENT, root=lane.registry_root)) == 1

    def test_missing_incumbent_probs_fails_loud_before_training(self, lane: Lane) -> None:
        evalset.probs_path(lane.incumbent.version, root=lane.eval_dir).unlink()
        with pytest.raises(WatcherRetrainError, match="--seed-incumbent-probs"):
            lane.run()
        assert lane.calls["retrain"] == 0  # validate the incumbent gate before any Tinker spend


class TestFreshEpoch:
    def test_promotes_on_absolute_bar_without_incumbent_probs(self, lane: Lane) -> None:
        evalset.probs_path(lane.incumbent.version, root=lane.eval_dir).unlink()  # no incumbent probs for this frame
        verdict = lane.run(fresh_epoch=True)
        assert verdict.startswith("watcher: fresh-epoch promoted")
        assert "served AUC" in verdict and "no incumbent gate" in verdict
        current = registry.current(w.WATCHER_COMPONENT, root=lane.registry_root)
        assert current is not None and current.version != lane.incumbent.version
        assert evalset.probs_path(current.version, root=lane.eval_dir).exists()

    def test_refuses_below_chance_candidate(self, lane: Lane) -> None:
        evalset.probs_path(lane.incumbent.version, root=lane.eval_dir).unlink()
        below_chance = np.full(N, 0.9)
        below_chance[10:] = 0.1  # negatives fire hardest -> sentinel AUC 0.0, below chance
        lane.candidate = below_chance
        with pytest.raises(WatcherRetrainError, match="above chance"):
            lane.run(fresh_epoch=True)
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version

    def test_refuses_non_finite_auc(self, lane: Lane, monkeypatch: pytest.MonkeyPatch) -> None:
        # A single-class frame makes sklearn return NaN; `nan <= 0.5` is False, so require finite explicitly.
        evalset.probs_path(lane.incumbent.version, root=lane.eval_dir).unlink()
        monkeypatch.setattr(promotion, "sentinel_auc", lambda *_a, **_k: float("nan"))
        with pytest.raises(WatcherRetrainError, match="finite"):
            lane.run(fresh_epoch=True)
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version

    def test_refuses_on_orphan_probs_from_pruned_version(self, lane: Lane) -> None:
        evalset.probs_path(lane.incumbent.version, root=lane.eval_dir).unlink()  # the registered incumbent's probs
        # An orphan probs file from a pruned version still covers the current frame -> the cutover is over.
        orphan = evalset.write_probs(
            lane.frame, "pruned-v000", {rid: 0.5 for rid in lane.frame.ids}, auc=0.5, root=lane.eval_dir
        )
        with pytest.raises(FreshEpochError, match="one-shot") as excinfo:
            lane.run(fresh_epoch=True)
        assert str(orphan) in str(excinfo.value)
        assert lane.calls["retrain"] == 0

    def test_refuses_when_frame_already_scored(self, lane: Lane) -> None:
        # The lane seeds incumbent probs for the current frame; the one-shot guard fires before training.
        with pytest.raises(FreshEpochError, match="one-shot") as excinfo:
            lane.run(fresh_epoch=True)
        assert str(evalset.probs_path(lane.incumbent.version, root=lane.eval_dir)) in str(excinfo.value)
        assert lane.calls["retrain"] == 0
        assert registry.current(w.WATCHER_COMPONENT, root=lane.registry_root).version == lane.incumbent.version


class TestSeedIncumbentProbs:
    def frame_and_root(self, tmp_path: Path) -> tuple[evalset.EvalFrame, Path]:
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "watcher").mkdir(parents=True)
        pq.write_table(watcher_table(eval_rows()), dataset_dir / "watcher" / "test.parquet")
        eval_dir = tmp_path / "eval"
        evalset.freeze_eval("watcher", dataset_dir=dataset_dir, root=eval_dir)
        return evalset.EvalFrame.load(root=eval_dir), eval_dir

    def test_valid_cache_accepted_and_written(self, tmp_path: Path) -> None:
        frame, eval_dir = self.frame_and_root(tmp_path)
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({rid: str(INCUMBENT[i]) for i, rid in enumerate(frame.ids)}))
        version = "v001-20260101-abcdef123456"
        path = w.seed_incumbent_probs(cache, version=version, expected_render=2, eval_root=eval_dir)
        assert path.exists()
        loaded = evalset.load_probs(frame, version, expected_render=2, root=eval_dir)
        assert loaded.tolist() == [float(INCUMBENT[i]) for i in range(len(frame))]

    def test_migrated_cache_seeds_under_the_incumbents_own_render(self, tmp_path: Path) -> None:
        # v001 predates render 2: its cache is scored under its own contract (render 1, the
        # E12 precedent) and must seed as-is; load_probs verifies against that same render.
        frame, eval_dir = self.frame_and_root(tmp_path)
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({rid: str(INCUMBENT[i]) for i, rid in enumerate(frame.ids)}))
        version = "v001-20260101-abcdef123456"
        path = w.seed_incumbent_probs(cache, version=version, expected_render=1, eval_root=eval_dir)
        assert json.loads(path.read_text())["meta"]["render"] == 1
        loaded = evalset.load_probs(frame, version, expected_render=1, root=eval_dir)
        assert loaded.tolist() == [float(INCUMBENT[i]) for i in range(len(frame))]
        with pytest.raises(ProbsStoreError, match="render"):
            evalset.load_probs(frame, version, expected_render=2, root=eval_dir)

    def test_incomplete_cache_fails_loud(self, tmp_path: Path) -> None:
        frame, eval_dir = self.frame_and_root(tmp_path)
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({rid: str(INCUMBENT[i]) for i, rid in enumerate(frame.ids) if i < len(frame) - 1}))
        with pytest.raises(ProbsStoreError, match="missing"):
            w.seed_incumbent_probs(cache, version="v001", expected_render=2, eval_root=eval_dir)

    def test_drifted_cache_fails_loud(self, tmp_path: Path) -> None:
        frame, eval_dir = self.frame_and_root(tmp_path)
        drifted = {rid: str(INCUMBENT[i]) for i, rid in enumerate(frame.ids)} | {"foreign-row": "0.5"}
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps(drifted))
        with pytest.raises(ProbsStoreError, match="foreign"):
            w.seed_incumbent_probs(cache, version="v001", expected_render=2, eval_root=eval_dir)
