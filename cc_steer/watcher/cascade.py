"""The three-stage steering cascade: cheap gate, drafting watcher, exemplar-conditioned refiner.

Every stage sees model input built through :mod:`cc_steer.rendering` — the
drafter's chat prompt is :func:`~cc_steer.rendering.watcher_prompt` and every
flattened surface (the gate's input, the refiner's context block) reproduces
:func:`~cc_steer.rendering.gate_text` byte-for-byte — so what the live models
see is exactly what their training and evaluation data rendered. The stages
hide behind Protocols so tests inject fakes and a lab-trained gate or a local
drafter drops in without touching the orchestrator.

The :class:`Cascade` also owns the per-session ledger: a turn is evaluated at
most once, a session cools down for ``cooldown_turns`` after each proposal and
stops being evaluated after ``max_per_session`` proposals. Suppressed turns
(cooldown, cap, or a below-threshold gate score) produce no proposal row;
every stage-2 invocation does, so shadow analysis sees the drafter's
abstentions.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from cc_transcript.judge import resolved_model

from cc_steer.claude import run_claude
from cc_steer.exemplars import exemplars_for, load_index, mmr_select
from cc_steer.rendering import NO_STEER, gate_text, watcher_prompt
from cc_steer.watcher.types import Draft, SteerProposal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_transcript.context import ContextWindow

    from cc_steer.exemplars import Encoder, Exemplar
    from cc_steer.rendering import Message
    from cc_steer.store import FeedbackStore
    from cc_steer.watcher.types import CascadeConfig

TURN_HEADER = re.compile(r"^<(?:user|assistant)>$", re.MULTILINE)

DRAFT_SYSTEM = """\
You are watching a live Claude Code session on the user's behalf. The recent turns of
the session follow, each tagged with its role. Decide whether the user, seeing this
moment, would step in and steer the assistant — correct its course, reject an unwanted
action, redirect its approach, or resolve a choice it raised.

If they would, output the single steering message the user would send right now —
their words, nothing else. If the session needs no steering, output exactly NO_STEER."""

REFINE_SYSTEM = """\
You are the final voice of a steering watcher over a live Claude Code session. A
cheaper watcher drafted a steering message for the current moment; you decide whether
steering is genuinely warranted and, if so, say it the way this user would.

Past moments where this user actually steered are provided as exemplars — each shows
the context they reacted to and the direction they gave. Match their voice: their
bluntness, their brevity, their vocabulary. Output the final steering message alone,
or exactly NO_STEER when the moment is not worth interrupting."""

REFINE_PROMPT = """\
=== RECENT SESSION CONTEXT ===
{context}

=== DRAFT STEERING MESSAGE ===
{draft}

=== HOW THIS USER HAS STEERED IN SIMILAR MOMENTS ===
{exemplars}

Output the final steering message in this user's voice, or exactly NO_STEER if
steering isn't warranted."""


class Gate(Protocol):
    """Stage 1: a cheap scorer over the flattened window text."""

    def score(self, text: str) -> float: ...


class Drafter(Protocol):
    """Stage 2: drafts the steering message (``NO_STEER`` text on abstain), scored when it can."""

    async def draft(self, prompt: list[Message]) -> Draft: ...


class Refiner(Protocol):
    """Stage 3: voices the final steer from the draft and exemplars, or ``NO_STEER``."""

    async def refine(self, prompt: list[Message], draft: str, exemplars: Sequence[Exemplar]) -> str: ...


@dataclass(slots=True)
class SessionLedger:
    """One session's cascade bookkeeping.

    Attributes:
        evaluated: Turn indices already run through the cascade.
        proposals: How many proposals the session has produced.
        last_proposal_turn: The turn the latest proposal fired at, or None.
    """

    evaluated: set[int] = field(default_factory=set)
    proposals: int = 0
    last_proposal_turn: int | None = None


@dataclass(frozen=True, slots=True)
class HeuristicGate:
    """The v0 gate: pass any window with at least ``min_turns`` turns.

    A placeholder until the lab's trained gate arrives — it scores the same
    flattened text a loaded-model gate would, so one drops in for the other.
    Cooldown and per-session caps are the orchestrator's job and run before
    any gate; the heuristic reduces to the turn floor.
    """

    min_turns: int = 3

    def score(self, text: str) -> float:
        return 1.0 if len(TURN_HEADER.findall(text)) >= self.min_turns else 0.0


@dataclass(frozen=True, slots=True)
class SpawnDrafter:
    """Stage 2 over the ``claude`` CLI: one headless turn on a cheap tier.

    Attributes:
        model: An abstract spawnllm tier or a concrete model id.
    """

    model: str = "medium"

    async def draft(self, prompt: list[Message]) -> Draft:
        return Draft(await run_claude(flattened(prompt), system=DRAFT_SYSTEM, model=concrete_model(self.model)))


@dataclass(frozen=True, slots=True)
class SpawnRefiner:
    """Stage 3 over the ``claude`` CLI: the frontier voice, conditioned on exemplars.

    Attributes:
        tier: An abstract spawnllm tier or a concrete model id.
    """

    tier: str = "large"

    async def refine(self, prompt: list[Message], draft: str, exemplars: Sequence[Exemplar]) -> str:
        body = REFINE_PROMPT.format(context=flattened(prompt), draft=draft, exemplars=exemplar_block(exemplars))
        return await run_claude(body, system=REFINE_SYSTEM, model=concrete_model(self.tier))


@dataclass(frozen=True, slots=True)
class Cascade:
    """The orchestrator: gate, draft, retrieve, refine — one proposal per live moment.

    Retrieval is optional: without an encoder (the ``embed`` extra missing or
    the exemplar index unbuilt) the refiner runs on the draft alone. The refiner
    itself is optional: with ``refiner=None`` a fired draft IS the steer —
    the two-stage configuration E2 validated, with stage 3 disabled pending its
    rewrite-only redesign (E9).

    Attributes:
        gate: The stage-1 scorer.
        drafter: The stage-2 draft model.
        refiner: The stage-3 voice model, or None to ship fired drafts as-is.
        store: The feedback store carrying the exemplar index.
        config: The cascade's tuning knobs.
        encoder: The query encoder for exemplar retrieval, or None to disable it.
    """

    gate: Gate
    drafter: Drafter
    refiner: Refiner | None
    store: FeedbackStore
    config: CascadeConfig
    encoder: Encoder | None = None
    ledgers: dict[str, SessionLedger] = field(default_factory=dict)

    async def evaluate(
        self, session_id: str, *, turn_index: int, anchor_uuid: str, window: ContextWindow
    ) -> SteerProposal | None:
        """Runs one live moment through the cascade, at most once per (session, turn).

        Args:
            session_id: The session the window came from.
            turn_index: The completed turn the window ends at.
            anchor_uuid: The uuid of the last event in the window.
            window: The live, triggerless context window.

        Returns:
            The proposal — recorded for every stage-2 invocation, abstentions
            included — or None when the turn was already evaluated, the session
            is cooling down or capped, or the gate suppressed the moment.
        """
        ledger = self.ledgers.setdefault(session_id, SessionLedger())
        if turn_index in ledger.evaluated:
            return None
        ledger.evaluated.add(turn_index)
        if ledger.proposals >= self.config.max_per_session:
            return None
        if (last := ledger.last_proposal_turn) is not None and turn_index - last < self.config.cooldown_turns:
            return None
        text = gate_text(window)
        if (score := self.gate.score(text)) < self.config.gate_threshold:
            return None
        prompt = watcher_prompt(window, render_version=self.config.render_version)
        drafted = await self.drafter.draft(prompt)
        draft = steer_or_none(drafted.text)
        exemplars: list[Exemplar] = []
        steer = None
        if draft is not None:
            if self.refiner is None:
                steer = draft
            else:
                exemplars = await self.retrieve(text)
                steer = steer_or_none(await self.refiner.refine(prompt, draft, exemplars))
        ledger.proposals += 1
        ledger.last_proposal_turn = turn_index
        return SteerProposal(
            session_id=session_id,
            anchor_uuid=anchor_uuid,
            turn_index=turn_index,
            ts=datetime.now(UTC).isoformat(),
            gate_score=score,
            sentinel_prob=drafted.sentinel_prob,
            draft=draft,
            steer=steer,
            exemplar_keys=tuple(exemplar.dedup_key for exemplar in exemplars),
            stage_versions=self.stage_versions(),
            window_render=text,
        )

    async def retrieve(self, query_text: str) -> list[Exemplar]:
        """MMR-retrieves the exemplars for one query; empty without an encoder or index."""
        if self.encoder is None:
            return []
        keys, matrix = await load_index(self.store, model=self.config.embed_model)
        if not keys:
            return []
        hits = mmr_select(self.encoder.encode([query_text])[0], matrix, k=self.config.exemplar_k)
        return await exemplars_for(self.store, [(keys[index], score) for index, score in hits])

    def stage_versions(self) -> str:
        return json.dumps(
            dataclasses.asdict(self.config) | {"gate": type(self.gate).__name__, "retrieval": self.encoder is not None},
            sort_keys=True,
        )


def concrete_model(model: str) -> str:
    """The model id to run: an abstract tier resolves through the active backend."""
    match model:
        case "small" | "medium" | "large" as tier:
            return resolved_model(tier)
        case _:
            return model


def flattened(prompt: Sequence[Message]) -> str:
    return "\n\n".join(f"<{message['role']}>\n{message['content']}" for message in prompt)


def steer_or_none(text: str) -> str | None:
    return None if not (clean := text.strip()) or clean == NO_STEER else clean


def exemplar_block(exemplars: Sequence[Exemplar]) -> str:
    if not exemplars:
        return "(no exemplars available)"
    return "\n\n".join(
        f"--- exemplar {index} ---\n{exemplar.context_text}\n"
        f">>> the user steered: {exemplar.direction or exemplar.verbatim}"
        for index, exemplar in enumerate(exemplars, start=1)
    )
