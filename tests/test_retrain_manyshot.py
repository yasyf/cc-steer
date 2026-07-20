from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cc_steer.claude import ClaudeResult, ClaudeUsage
from cc_steer.retrain import manyshot
from cc_steer.retrain.data import WatcherRow
from cc_steer.retrain.evalset import EvalFrame

if TYPE_CHECKING:
    from pathlib import Path


def exemplar(index: int) -> WatcherRow:
    return WatcherRow(
        id=f"e{index}",
        prompt=({"role": "user", "content": f"exemplar {index} context with enough text to shingle nicely"},),
        reference="NO_STEER" if index % 2 else f"steer direction {index}",
        verbatim="",
        label=bool(index % 2),
        category="direction",
    )


def frame(n: int) -> EvalFrame:
    labels = np.array([bool(i % 2) for i in range(n)], dtype=bool)
    return EvalFrame(
        ids=tuple(f"r{i}" for i in range(n)),
        labels=labels,
        corrective=labels,
        prose=np.ones(n, dtype=bool),
        tails=tuple(f"<user>\ncontext {i}" for i in range(n)),
        digest="deadbeefdeadbeef",
    )


class FakeClaude:
    def __init__(self, *, cache_read: int) -> None:
        self.cache_read = cache_read
        self.systems: list[str] = []
        self.calls = 0

    async def __call__(self, prompt: str, *, system: str, model: str) -> ClaudeResult:
        self.systems.append(system)
        count = prompt.count("CONTEXT ")
        text = "\n".join(f"{position}: 0.{position}" for position in range(1, count + 1))
        usage = ClaudeUsage(
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=0 if self.calls == 0 else self.cache_read,
            cache_creation_input_tokens=0,
            cost_usd=0.0,
        )
        self.calls += 1
        return ClaudeResult(text=text, usage=usage)


class TestExemplarSystem:
    def test_byte_constant_across_calls(self) -> None:
        rows = [exemplar(i) for i in range(20)]
        assert manyshot.build_exemplar_system(rows, budget_chars=3000, seed=5) == manyshot.build_exemplar_system(
            rows, budget_chars=3000, seed=5
        )

    def test_seed_changes_the_selection(self) -> None:
        rows = [exemplar(i) for i in range(20)]
        assert manyshot.build_exemplar_system(rows, budget_chars=1500, seed=1) != manyshot.build_exemplar_system(
            rows, budget_chars=1500, seed=2
        )

    def test_respects_the_budget(self) -> None:
        rows = [exemplar(i) for i in range(200)]
        system = manyshot.build_exemplar_system(rows, budget_chars=2000, seed=3)
        assert len(system) <= 2000 + len(manyshot.SYSTEM_PREFIX) + len(manyshot.SYSTEM_SUFFIX)
        assert system.startswith(manyshot.SYSTEM_PREFIX) and system.endswith(manyshot.SYSTEM_SUFFIX)


class TestParseBatch:
    def test_clamps_to_unit_interval(self) -> None:
        assert manyshot.parse_batch_probs("1: 0.9\n2: 1.4\n3: -0.3", 3) == [0.9, 1.0, 0.0]

    def test_missing_line_fails_loud(self) -> None:
        with pytest.raises(manyshot.ManyShotError, match="missing 2 of 3"):
            manyshot.parse_batch_probs("2: 0.5", 3)


class TestScoreFrame:
    @pytest.mark.anyio
    async def test_aggregates_batches_and_writes_paired_probs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeClaude(cache_read=500)
        monkeypatch.setattr(manyshot, "run_claude", fake)
        target = frame(5)
        path = await manyshot.score_frame(
            target, [exemplar(i) for i in range(10)], version="ms1", model="claude", batch_size=2, eval_root=tmp_path
        )
        assert fake.calls == 3  # ceil(5 / 2)
        assert len(set(fake.systems)) == 1  # one byte-constant prefix reused across every batch
        stored = json.loads(path.read_text())["probs"]
        assert set(stored) == set(target.ids)
        # steer 0.1/0.2 by within-batch position → P(NO_STEER) = 0.9 / 0.8
        assert stored["r0"] == pytest.approx(0.9) and stored["r1"] == pytest.approx(0.8)
        assert stored["r2"] == pytest.approx(0.9) and stored["r4"] == pytest.approx(0.9)

    @pytest.mark.anyio
    async def test_aborts_on_a_cold_cache_streak(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(manyshot, "run_claude", FakeClaude(cache_read=0))
        with pytest.raises(manyshot.ManyShotError, match="cache cold"):
            await manyshot.score_frame(
                frame(6),
                [exemplar(i) for i in range(4)],
                version="ms2",
                model="claude",
                batch_size=2,
                eval_root=tmp_path,
                max_cache_miss_streak=2,
            )
