from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import numpy as np
import pytest
import spawnllm
from athome.research.golden import (
    MANIFEST_NAME,
    PACKET_NAME,
    GoldenGateViolation,
    Stratum,
    build_packet,
)
from athome.research.judge import Pairwise, Vote

from cc_steer.retrain import judged
from cc_steer.retrain.promotion import corrected_gate, watcher_promotable

if TYPE_CHECKING:
    from pathlib import Path

N = 16
GOLDEN_ROWS = [("g0", True), ("g1", False), ("g2", True), ("g3", False)]


@dataclass(frozen=True, slots=True)
class FrameStub:
    ids: tuple[str, ...]
    tails: tuple[str, ...]
    digest: str


def context_for(row_id: str, warrant: bool) -> str:
    return f"session for {row_id}: WARRANT={'yes' if warrant else 'no'}"


def frame_and_scores(*, warrant: bool) -> tuple[FrameStub, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    # 8 warranted rows where only the candidate fires, 8 unwarranted rows where only the incumbent fires:
    # every warranted row is a candidate-fired disagreement, so the judge's warrant marker drives harm.
    ids = tuple(f"r{i}" for i in range(N))
    frame = FrameStub(ids=ids, tails=tuple(context_for(rid, warrant) for rid in ids), digest="frame-digest")
    candidate = np.concatenate([np.linspace(0.99, 0.90, 8), np.full(8, 0.1)])
    incumbent = np.concatenate([np.full(8, 0.1), np.full(8, 0.9)])
    mask = np.array([True] * 8 + [False] * 8)
    return frame, candidate, incumbent, 0.5, mask, mask.copy()


def build_golden(root: Path, *, rows: list[tuple[str, bool]], labeled: bool) -> None:
    source = [{"row_id": row_id, "context": context_for(row_id, warrant)} for row_id, warrant in rows]
    packet = build_packet(
        source,
        strata=[Stratum(name="fire", size=len(source))],
        stratum_of=lambda _row: "fire",
        window_of=lambda row: row["context"],
        row_id=lambda row: row["row_id"],
        seed=7,
        dataset_digest="golden-digest",
        question="Should the watcher steer here?",
        header="# Watcher fires golden packet",
    )
    directory = judged.golden_dir(root=root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PACKET_NAME).write_text(packet.packet_md)
    (directory / MANIFEST_NAME).write_text(json.dumps(packet.manifest))
    (directory / judged.FIRES_NAME).write_text(
        "".join(json.dumps({"row_id": r, "context": context_for(r, w)}) + "\n" for r, w in rows)
    )
    if labeled:
        warrant_by_id = dict(rows)
        entries = [
            {"row": entry["row"], "row_id": entry["row_id"], "label": "yes" if warrant_by_id[entry["row_id"]] else "no"}
            for entry in json.loads(packet.labels_template)
        ]
        (directory / judged.LABELS_NAME).write_text(json.dumps(entries))


def _slots(prompt: str) -> tuple[str, str]:
    _, rest = prompt.split("--- A ---\n", 1)
    a, b = rest.split("\n\n--- B ---\n", 1)
    return a, b.rstrip("\n")


def decide(prompt: str) -> str:
    # A stand-in for the opus judge: reference wins over garbage, identical slots tie, and the STEER
    # action wins exactly when the context is marked WARRANT=yes.
    a, b = _slots(prompt)
    match (judged.GARBAGE_TEXT in a, judged.GARBAGE_TEXT in b):
        case (True, False):
            return "B"
        case (False, True):
            return "A"
    if a == b:
        return "tie"
    steer = "A" if a.startswith(f"[{judged.FIRE_ACTION}]") else "B"
    return steer if "WARRANT=yes" in prompt else ("B" if steer == "A" else "A")


class FakeExtract:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, response_model: type[Pairwise], **_kw: Any) -> Pairwise:
        self.prompts.append(prompt)
        return Pairwise(winner=decide(prompt))


def run_judged(**kwargs: Any) -> bool:
    async def _inner() -> bool:
        return await judged.judged_harmful_favors_incumbent(**kwargs)

    return anyio.run(_inner)


class TestDisagreementFires:
    def test_selects_warranted_xor_fires(self) -> None:
        candidate = np.array([0.9, 0.85, 0.1, 0.1, 0.1, 0.1])
        incumbent = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.1])
        warranted = np.array([True, True, True, False, False, False])
        fires = judged.disagreement_fires(
            candidate,
            incumbent,
            incumbent_fire_threshold=0.5,
            warranted=warranted,
            ids=tuple(f"r{i}" for i in range(6)),
            contexts=tuple(f"c{i}" for i in range(6)),
        )
        # incumbent fires {3,4} (budget 2); candidate matches to its top 2 {0,1}; only {0,1} are warranted.
        assert [(f.row_id, f.candidate_fired) for f in fires] == [("r0", True), ("r1", True)]
        assert [f.context for f in fires] == ["c0", "c1"]

    def test_incumbent_only_fire_is_a_disagreement(self) -> None:
        # A warranted row the incumbent fires but the candidate misses is a disagreement, candidate_fired False.
        candidate = np.array([0.1, 0.1, 0.9, 0.9])
        incumbent = np.array([0.9, 0.1, 0.1, 0.1])
        warranted = np.array([True, True, False, False])
        fires = judged.disagreement_fires(
            candidate,
            incumbent,
            incumbent_fire_threshold=0.5,
            warranted=warranted,
            ids=("r0", "r1", "r2", "r3"),
            contexts=("c0", "c1", "c2", "c3"),
        )
        assert [(f.row_id, f.candidate_fired) for f in fires] == [("r0", False)]


class TestFavorsIncumbent:
    @pytest.mark.parametrize(
        ("votes", "expected"),
        [
            pytest.param([Vote.LOSS, Vote.LOSS, Vote.WIN], True, id="losses-exceed-wins"),
            pytest.param([Vote.WIN, Vote.WIN, Vote.LOSS], False, id="wins-exceed-losses"),
            pytest.param([Vote.WIN, Vote.LOSS, Vote.TIE], False, id="tie-does-not-tip"),
            pytest.param([], False, id="no-votes"),
        ],
    )
    def test_aggregation(self, votes: list[Vote], expected: bool) -> None:
        assert judged._favors_incumbent(votes) is expected


class TestJudgedGate:
    def test_no_disagreement_returns_false_without_spend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", spy)
        frame, _candidate, incumbent, threshold, warranted, _labels = frame_and_scores(warrant=True)
        result = run_judged(
            candidate_fire_scores=incumbent.copy(),  # candidate == incumbent -> no disagreement
            incumbent_fire_scores=incumbent,
            incumbent_fire_threshold=threshold,
            frame=frame,
            warranted=warranted,
            root=tmp_path,
        )
        assert result is False
        assert spy.prompts == []  # no golden load, no judge spend

    def test_flag_day_raises_before_any_spend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", spy)
        build_golden(tmp_path, rows=GOLDEN_ROWS, labeled=False)  # packet + fires, but no human labels.json
        frame, candidate, incumbent, threshold, warranted, _labels = frame_and_scores(warrant=True)
        with pytest.raises(GoldenGateViolation, match="refusing to fabricate labels"):
            run_judged(
                candidate_fire_scores=candidate,
                incumbent_fire_scores=incumbent,
                incumbent_fire_threshold=threshold,
                frame=frame,
                warranted=warranted,
                root=tmp_path,
            )
        assert spy.prompts == []  # raised before Judge.bind and any vote

    @pytest.mark.parametrize(
        "sidecar",
        [
            pytest.param(
                [(row_id, f"SUBSTITUTED {row_id}") for row_id, _ in GOLDEN_ROWS],
                id="substituted-contexts",
            ),
            pytest.param(
                [
                    ("g0", context_for("g1", False)),  # real windows, reassigned to the wrong row ids
                    ("g1", context_for("g0", True)),
                    ("g2", context_for("g2", True)),
                    ("g3", context_for("g3", False)),
                ],
                id="reassigned-contexts",
            ),
        ],
    )
    def test_unbound_sidecar_raises_before_any_spend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar: list[tuple[str, str]]
    ) -> None:
        spy = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", spy)
        build_golden(tmp_path, rows=GOLDEN_ROWS, labeled=True)
        (judged.golden_dir(root=tmp_path) / judged.FIRES_NAME).write_text(
            "".join(json.dumps({"row_id": row_id, "context": ctx}) + "\n" for row_id, ctx in sidecar)
        )
        frame, candidate, incumbent, threshold, warranted, _labels = frame_and_scores(warrant=True)
        with pytest.raises(GoldenGateViolation, match="not bound to the verified packet"):
            run_judged(
                candidate_fire_scores=candidate,
                incumbent_fire_scores=incumbent,
                incumbent_fire_threshold=threshold,
                frame=frame,
                warranted=warranted,
                root=tmp_path,
            )
        assert spy.prompts == []  # bound check raises before Judge.bind and any vote

    @pytest.mark.parametrize(
        ("warrant", "expected_harmful", "expected_promote"),
        [
            pytest.param(True, False, True, id="warranted-fires-promote"),
            pytest.param(False, True, False, id="unwarranted-fires-refuse"),
        ],
    )
    def test_end_to_end_through_enforced_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        warrant: bool,
        expected_harmful: bool,
        expected_promote: bool,
    ) -> None:
        fake = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", fake)
        build_golden(tmp_path, rows=GOLDEN_ROWS, labeled=True)
        frame, candidate, incumbent, threshold, warranted, labels = frame_and_scores(warrant=warrant)
        harmful = run_judged(
            candidate_fire_scores=candidate,
            incumbent_fire_scores=incumbent,
            incumbent_fire_threshold=threshold,
            frame=frame,
            warranted=warranted,
            root=tmp_path,
        )
        assert harmful is expected_harmful
        # The panel, its controls, and every disagreement vote all bought through the mocked backend.
        assert fake.prompts
        result = corrected_gate(
            candidate,
            incumbent,
            candidate="cand",
            incumbent="inc",
            incumbent_fire_threshold=threshold,
            labels=labels,
            warranted=warranted,
            harmful_favors_incumbent=harmful,
        )
        assert result.promote is expected_promote
        assert watcher_promotable(result).promote is expected_promote
