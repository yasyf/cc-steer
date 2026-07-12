"""The watcher's shared shapes: a stage-2 draft, one shadow proposal row, and the cascade's knobs."""

from __future__ import annotations

from dataclasses import dataclass

from cc_steer.exemplars import EMBED_MODEL


@dataclass(frozen=True, slots=True)
class Draft:
    """One drafter answer: the raw text plus the abstain score, when the drafter has one.

    Attributes:
        text: The draft steering message, or exactly ``NO_STEER`` on abstain.
        sentinel_prob: The drafter's first-token P(NO_STEER) — the score its
            abstain decision was made at — or None for drafters that decide by
            text alone (the spawn path).
    """

    text: str
    sentinel_prob: float | None = None


@dataclass(frozen=True, slots=True)
class SteerProposal:
    """One cascade verdict on a live moment, ready for delivery.

    A proposal is recorded for every stage-2 invocation — abstentions included,
    so shadow analysis sees what the drafter declined — but never for turns the
    gate suppressed.

    Attributes:
        session_id: The session the moment came from.
        anchor_uuid: The uuid of the last event in the evaluated window — the
            proposal's identity within the session.
        turn_index: The completed turn the window ends at.
        ts: When the cascade fired, ISO-8601.
        gate_score: The stage-1 score the moment passed at.
        sentinel_prob: The drafter's abstain score for the moment, when it has one.
        draft: The stage-2 draft, or None when stage 2 abstained.
        steer: The stage-3 final steering message, or None on ``NO_STEER``.
        exemplar_keys: The dedup keys of the exemplars shown to stage 3.
        stage_versions: A JSON blob of the model ids and config the cascade ran with.
        window_render: The exact flattened window text the cascade scored, so
            replay reads the moment off the row instead of reconstructing it
            from ``(session_id, anchor_uuid, turn_index)``.
    """

    session_id: str
    anchor_uuid: str
    turn_index: int
    ts: str
    gate_score: float | None
    draft: str | None
    steer: str | None
    exemplar_keys: tuple[str, ...]
    stage_versions: str
    window_render: str
    sentinel_prob: float | None = None


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """The live cascade's tuning knobs.

    Attributes:
        gate_threshold: Stage-1 score below which a turn is suppressed without
            a proposal row.
        cooldown_turns: How many turns after a proposal a session is left
            alone before the cascade fires again.
        min_turns: The turn floor the heuristic gate requires in a window.
        max_per_session: Proposals a session may accumulate before the cascade
            stops evaluating it.
        exemplar_k: How many exemplars stage-3 retrieval selects.
        stage2_model: The drafter's model — an abstract spawnllm tier
            (``small``/``medium``/``large``), a concrete model id, or the
            local watcher's base model id.
        stage2_threshold: The local drafter's abstain threshold on
            P(NO_STEER); None for drafters without a score-based decision.
        stage3_tier: The refiner's model, same forms as ``stage2_model``.
        drafter_kind: Which stage-2 implementation runs (``spawn`` or ``mlx``).
        render_version: The prompt-rendering contract version the drafter was
            trained on; carried from the promoted watcher's metadata.
        embed_model: The embedding model the exemplar index was built with.
    """

    gate_threshold: float
    cooldown_turns: int = 5
    min_turns: int = 3
    max_per_session: int = 5
    exemplar_k: int = 8
    stage2_model: str = "medium"
    stage2_threshold: float | None = None
    stage3_tier: str = "large"
    drafter_kind: str = "spawn"
    render_version: int = 1
    embed_model: str = EMBED_MODEL
