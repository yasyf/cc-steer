from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId

from cc_steer.exemplars import build_index
from cc_steer.rendering import NO_STEER, gate_text, watcher_prompt
from cc_steer.watcher.cascade import Cascade, HeuristicGate, concrete_model, flattened
from cc_steer.watcher.types import CascadeConfig, Draft
from tests.test_exemplars import TRAIN_SESSION, CountingEncoder, seed_steering

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_steer.exemplars import Encoder, Exemplar
    from cc_steer.rendering import Message
    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio

SESSION = "sess-live"


class FakeGate:
    def __init__(self, score: float) -> None:
        self.value = score
        self.calls: list[str] = []

    def score(self, text: str) -> float:
        self.calls.append(text)
        return self.value


class FakeDrafter:
    def __init__(self, output: str, sentinel_prob: float | None = None) -> None:
        self.output = output
        self.sentinel_prob = sentinel_prob
        self.prompts: list[list[Message]] = []

    async def draft(self, prompt: list[Message]) -> Draft:
        self.prompts.append(prompt)
        return Draft(self.output, self.sentinel_prob)


class FakeRefiner:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[list[Message], str, tuple[str, ...]]] = []

    async def refine(self, prompt: list[Message], draft: str, exemplars: Sequence[Exemplar]) -> str:
        self.calls.append((prompt, draft, tuple(exemplar.dedup_key for exemplar in exemplars)))
        return self.output


def live_window(turn_count: int = 4) -> ContextWindow:
    return ContextWindow(
        anchor=EventRef(SessionId(SESSION), EventUuid("anchor-0")),
        before=tuple(
            TurnRef(
                role="user" if index % 2 == 0 else "assistant",
                refs=(),
                preview=f"turn {index}",
                tool_digests=(),
            )
            for index in range(turn_count)
        ),
        trigger=None,
        after=(),
        fidelity="full",
        preview_chars=200,
    )


def cascade_for(
    store: FeedbackStore,
    *,
    gate: FakeGate | HeuristicGate | None = None,
    drafter: FakeDrafter | None = None,
    refiner: FakeRefiner | None = None,
    no_refiner: bool = False,
    encoder: Encoder | None = None,
    **overrides: float | int | str,
) -> Cascade:
    return Cascade(
        gate=gate or FakeGate(1.0),
        drafter=drafter or FakeDrafter("draft steer"),
        refiner=None if no_refiner else (refiner or FakeRefiner("final steer")),
        store=store,
        config=CascadeConfig(gate_threshold=0.5, **overrides),
        encoder=encoder,
    )


async def test_gate_suppression_returns_none_without_llm_calls(store: FeedbackStore) -> None:
    gate, drafter = FakeGate(0.4), FakeDrafter("draft steer")
    cascade = cascade_for(store, gate=gate, drafter=drafter)
    result = await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window())
    assert result is None
    assert gate.calls == [gate_text(live_window())]
    assert drafter.prompts == []


async def test_stage2_abstention_records_a_proposal_with_no_steer(store: FeedbackStore) -> None:
    refiner = FakeRefiner("never")
    cascade = cascade_for(store, drafter=FakeDrafter(NO_STEER), refiner=refiner)
    proposal = await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window())
    assert proposal is not None
    assert (proposal.draft, proposal.steer, proposal.exemplar_keys) == (None, None, ())
    assert proposal.gate_score == 1.0
    assert refiner.calls == []


async def test_full_path_conditions_the_refiner_on_exemplars(store: FeedbackStore) -> None:
    await seed_steering(store, "k-train", TRAIN_SESSION, "u1")
    encoder = CountingEncoder()
    await build_index(store, encoder=encoder)
    drafter, refiner = FakeDrafter("draft steer"), FakeRefiner("final steer")
    cascade = cascade_for(store, drafter=drafter, refiner=refiner, encoder=encoder)
    window = live_window()
    proposal = await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=window)
    assert proposal is not None
    assert (proposal.session_id, proposal.anchor_uuid, proposal.turn_index) == (SESSION, "a1", 3)
    assert (proposal.draft, proposal.steer) == ("draft steer", "final steer")
    assert proposal.exemplar_keys == ("k-train",)
    assert drafter.prompts == [watcher_prompt(window)]
    assert refiner.calls == [(watcher_prompt(window), "draft steer", ("k-train",))]
    versions = json.loads(proposal.stage_versions)
    assert versions["stage2_model"] == "medium"
    assert versions["stage3_tier"] == "large"
    assert versions["retrieval"] is True


async def test_proposal_carries_the_rendered_window(store: FeedbackStore) -> None:
    window = live_window()
    proposal = await cascade_for(store).evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=window)
    assert proposal is not None
    assert proposal.window_render == gate_text(window)


async def test_refiner_no_steer_keeps_the_draft(store: FeedbackStore) -> None:
    cascade = cascade_for(store, refiner=FakeRefiner(NO_STEER))
    proposal = await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window())
    assert proposal is not None
    assert (proposal.draft, proposal.steer, proposal.exemplar_keys) == ("draft steer", None, ())


async def test_no_refiner_ships_the_fired_draft_as_the_steer(store: FeedbackStore) -> None:
    cascade = cascade_for(store, drafter=FakeDrafter("draft steer", sentinel_prob=0.12), no_refiner=True)
    proposal = await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window())
    assert proposal is not None
    assert (proposal.draft, proposal.steer, proposal.exemplar_keys) == ("draft steer", "draft steer", ())
    assert proposal.sentinel_prob == 0.12


async def test_no_refiner_abstention_still_records_the_score(store: FeedbackStore) -> None:
    cascade = cascade_for(store, drafter=FakeDrafter(NO_STEER, sentinel_prob=0.93), no_refiner=True)
    proposal = await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window())
    assert proposal is not None
    assert (proposal.draft, proposal.steer) == (None, None)
    assert proposal.sentinel_prob == 0.93


async def test_spawn_style_drafts_carry_no_sentinel_prob(store: FeedbackStore) -> None:
    proposal = await cascade_for(store).evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window())
    assert proposal is not None
    assert proposal.sentinel_prob is None


async def test_cooldown_suppresses_turns_after_a_proposal(store: FeedbackStore) -> None:
    drafter = FakeDrafter("draft steer")
    cascade = cascade_for(store, drafter=drafter, cooldown_turns=5)
    assert await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window()) is not None
    assert await cascade.evaluate(SESSION, turn_index=7, anchor_uuid="a2", window=live_window()) is None
    assert len(drafter.prompts) == 1
    assert await cascade.evaluate(SESSION, turn_index=8, anchor_uuid="a3", window=live_window()) is not None
    assert await cascade.evaluate("sess-other", turn_index=4, anchor_uuid="b1", window=live_window()) is not None


async def test_max_per_session_caps_proposals(store: FeedbackStore) -> None:
    drafter = FakeDrafter("draft steer")
    cascade = cascade_for(store, drafter=drafter, cooldown_turns=0, max_per_session=1)
    assert await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window()) is not None
    assert await cascade.evaluate(SESSION, turn_index=4, anchor_uuid="a2", window=live_window()) is None
    assert len(drafter.prompts) == 1


async def test_same_turn_never_reevaluates(store: FeedbackStore) -> None:
    gate, drafter = FakeGate(1.0), FakeDrafter("draft steer")
    cascade = cascade_for(store, gate=gate, drafter=drafter, cooldown_turns=0)
    assert await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window()) is not None
    assert await cascade.evaluate(SESSION, turn_index=3, anchor_uuid="a1", window=live_window()) is None
    assert len(gate.calls) == 1
    assert len(drafter.prompts) == 1


def test_heuristic_gate_scores_on_the_turn_floor() -> None:
    gate = HeuristicGate(min_turns=3)
    assert gate.score(gate_text(live_window(turn_count=3))) == 1.0
    assert gate.score(gate_text(live_window(turn_count=2))) == 0.0


def test_flattened_prompt_matches_gate_text() -> None:
    window = live_window()
    assert flattened(watcher_prompt(window)) == gate_text(window)


def test_concrete_model_resolves_tiers_and_passes_ids_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.cascade.resolved_model", lambda tier: f"resolved-{tier}")
    assert concrete_model("medium") == "resolved-medium"
    assert concrete_model("large") == "resolved-large"
    assert concrete_model("mlx-community/some-local-model") == "mlx-community/some-local-model"
