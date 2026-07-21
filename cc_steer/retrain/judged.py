"""The judged promotion gate: an opus panel grades the disagreement fires behind a golden gate.

The free-metric gate (:func:`~cc_steer.retrain.promotion.corrected_gate`) proves a candidate
covers more warranted fires than the incumbent without exceeding its budget or regressing AUC.
It cannot see whether the candidate's *extra* fires are helpful or noise. This module closes that
gap: on the warranted rows where exactly one of candidate/incumbent fires — the corrected gate's
own discordant coverage pairs, reconstructed via :func:`~athome.train.gate.matched_fire_mask` — an
opus judge grades each disagreement pairwise ([STEER] vs [STAY_SILENT] over the same session
context). The candidate is the qwen-family watcher, so an anthropic judge grades cross-family and
:func:`~athome.research.judge.ensure_cross_family` passes honestly.

Every judge call runs through athome's enforced spend path: a :class:`~athome.research.judge.PanelGrant`
votes the human-labeled golden packet, :func:`~athome.research.golden.agreement` scores the panel
against those labels, and :func:`~athome.research.golden.prove_gate` mints the
:class:`~athome.research.golden.GoldenProof` that :func:`~athome.research.judge.judge_candidates`
requires before it will spend on the candidate rows. There is no bare LLM call to forget the gate.

The golden packet lives under ``~/.cc-steer/eval/golden/watcher-fires/``. This is a flag-day
bootstrap: until a human fills ``labels.json``, :func:`load_golden` raises
:class:`~athome.research.golden.GoldenGateViolation` before any judge spend — the gate never
fabricates a label to unblock itself. The verdict aggregates to ``harmful_favors_incumbent``
(judged losses outnumber judged wins), which feeds back into the corrected gate whose
terms :func:`~cc_steer.retrain.promotion.watcher_promotable` recomposes under the
instrument card's paired DeLong rule to decide the promotion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anyio
import numpy as np
from athome.concurrency import gather_bounded
from athome.research.golden import (
    MANIFEST_NAME,
    PACKET_NAME,
    GoldenGateViolation,
    GoldenProof,
    VerifiedManifest,
    agreement,
    prove_gate,
    read_labels,
    verify_packet,
)
from athome.research.judge import (
    ControlPair,
    Judge,
    JudgeRow,
    Pairwise,
    PanelGrant,
    Vote,
    VoteContext,
    judge_candidates,
    pairwise_vote,
)
from athome.train.gate import matched_fire_mask

from cc_steer.retrain.evalset import eval_root

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cc_steer.retrain.evalset import EvalFrame

GOLDEN_LEAF: tuple[str, ...] = ("golden", "watcher-fires")
LABELS_NAME = "labels.json"
FIRES_NAME = "fires.jsonl"
PROMPT_VERSION = "watcher-fire-harm-v1"
GENERATOR_FAMILY = "qwen"
JUDGE_BACKEND = "claude"
JUDGE_SEED = 1729
JUDGE_CONCURRENCY = 8
CONTROL_ROWS = 4
FIRE_ACTION = "STEER"
NOFIRE_ACTION = "STAY_SILENT"
GARBAGE_TEXT = "The mitochondrion is the powerhouse of the cell. Bananas ripen fastest at 20C."


@dataclass(frozen=True, slots=True)
class FireDisagreement:
    """One warranted row where the candidate and incumbent disagree on firing.

    Attributes:
        row_id: The frame row id the vote is keyed on.
        context: The render-v2 session context both behaviours respond to.
        candidate_fired: Whether the *candidate* fired here; the incumbent took the other action.
    """

    row_id: str
    context: str
    candidate_fired: bool


@dataclass(frozen=True, slots=True)
class Golden:
    """A loaded, verified watcher-fires golden packet with its human labels.

    Attributes:
        manifest: The manifest whose ``packet_sha256`` matched the rendered packet.
        human: The human golden labels, ``row_id -> fire-warranted``.
        contexts: The judgeable session context per golden row, ``row_id -> window``.
    """

    manifest: VerifiedManifest
    human: Mapping[str, bool]
    contexts: Mapping[str, str]


def golden_dir(*, root: Path | None = None) -> Path:
    """The watcher-fires golden packet directory under the frozen-eval root."""
    return eval_root(root).joinpath(*GOLDEN_LEAF)


def render_action(*, fired: bool, context: str) -> str:
    return f"[{FIRE_ACTION if fired else NOFIRE_ACTION}]\n{context}"


def build_prompt(a: str, b: str) -> str:
    """Render the pairwise harm-judging prompt over two watcher behaviours on one session context."""
    return (
        "You are auditing an AI pair-programming watcher that decides, at one moment in a coding "
        f"session, whether to interrupt the engineer with a steering note ([{FIRE_ACTION}]) or "
        f"stay silent ([{NOFIRE_ACTION}]). Two behaviours respond to the same context below. Pick "
        "the one whose choice is the more appropriate, helpful, and least disruptive call for that "
        "moment. Answer 'tie' only when they are genuinely indistinguishable.\n\n"
        f"--- A ---\n{a}\n\n--- B ---\n{b}\n"
    )


def judge_row(fire: FireDisagreement) -> JudgeRow:
    return JudgeRow(
        row_id=fire.row_id,
        candidate=render_action(fired=fire.candidate_fired, context=fire.context),
        reference=render_action(fired=not fire.candidate_fired, context=fire.context),
    )


def control_pairs(contexts: Mapping[str, str]) -> tuple[ControlPair, ...]:
    """Synthesize the embedded health controls a candidate batch must clear before it spends.

    A paraphrase restates the reference behaviour verbatim (a healthy judge answers tie); a garbage
    control offers unrelated text against a real behaviour (a healthy judge rejects it). Both are
    drawn from the golden contexts so the controls travel with the packet they gate.
    """
    items = tuple(contexts.items())[:CONTROL_ROWS]
    return (
        *(
            ControlPair(
                row_id=row_id,
                kind="paraphrase",
                candidate=(action := render_action(fired=True, context=window)),
                reference=action,
            )
            for row_id, window in items
        ),
        *(
            ControlPair(
                row_id=row_id,
                kind="garbage",
                candidate=GARBAGE_TEXT,
                reference=render_action(fired=True, context=window),
            )
            for row_id, window in items
        ),
    )


def disagreement_fires(
    candidate_fire_scores: np.ndarray,
    incumbent_fire_scores: np.ndarray,
    *,
    incumbent_fire_threshold: float,
    warranted: np.ndarray,
    ids: Sequence[str],
    contexts: Sequence[str],
) -> tuple[FireDisagreement, ...]:
    """Reconstruct the corrected gate's warranted discordant coverage pairs as judge inputs.

    Mirrors :func:`~athome.train.gate.corrected_gate`: the incumbent fires strictly above its
    threshold, the candidate is matched conservatively to that fire count, and a disagreement is a
    warranted row where exactly one side fires. Every score is higher-is-fire.
    """
    incumbent_fires = incumbent_fire_scores > incumbent_fire_threshold
    candidate_fires = matched_fire_mask(candidate_fire_scores, budget_fires=int(incumbent_fires.sum()))
    return tuple(
        FireDisagreement(row_id=ids[i], context=contexts[i], candidate_fired=bool(candidate_fires[i]))
        for i in np.flatnonzero((candidate_fires ^ incumbent_fires) & warranted)
    )


async def load_golden(directory: Path) -> Golden:
    """Load and verify the watcher-fires golden packet, refusing to spend on an unlabeled one.

    The fires sidecar is not a trust anchor: its ``row_id -> context`` map is bound to the verified
    packet by reconstructing each row's window from the packet the human labeled, so a substituted or
    reassigned context — same row ids, different prompts — is rejected before any panel spend.

    Raises:
        GoldenGateViolation: ``labels.json`` is absent (the flag-day bootstrap — never fabricate a
            label to unblock the gate), the packet content drifted from its manifest, or the fires
            sidecar's contexts are not the windows the verified packet pins.
    """
    root = anyio.Path(directory)
    labels = root / LABELS_NAME
    if not await labels.exists():
        raise GoldenGateViolation(
            f"no human golden labels at {labels}; a human must label the watcher-fires packet before "
            "the judged gate may spend — refusing to fabricate labels"
        )
    packet_md = await (root / PACKET_NAME).read_text()
    verified = verify_packet(packet_md=packet_md, manifest=json.loads(await (root / MANIFEST_NAME).read_text()))
    human = await read_labels(labels, manifest=verified)
    contexts = _read_fires(await (root / FIRES_NAME).read_text())
    if contexts != (bound := _packet_contexts(packet_md, verified)):
        raise GoldenGateViolation(
            f"golden fires sidecar is not bound to the verified packet: {sorted(contexts)} maps to contexts "
            f"that differ from the packet windows the human labeled ({sorted(bound)})"
        )
    return Golden(manifest=verified, human=human, contexts=contexts)


async def panel_labels(judge: Judge[Pairwise], golden: Golden, *, seed: int) -> dict[str, bool]:
    """Vote the golden packet under a :class:`PanelGrant`, mapping each pairwise verdict to fire-warranted."""
    grant = PanelGrant(golden.manifest)
    context = VoteContext(prompt_version=PROMPT_VERSION, digest=golden.manifest.rows_sha256)
    items = tuple(golden.contexts.items())
    votes = await gather_bounded(
        [
            lambda row_id=row_id, window=window: pairwise_vote(
                judge,
                generator_family=GENERATOR_FAMILY,
                grant=grant,
                context=context,
                row_id=row_id,
                candidate=render_action(fired=True, context=window),
                reference=render_action(fired=False, context=window),
                build_prompt=build_prompt,
                seed=seed,
            )
            for row_id, window in items
        ],
        concurrency=JUDGE_CONCURRENCY,
    )
    return {row_id: vote is Vote.WIN for (row_id, _), vote in zip(items, votes, strict=True)}


async def judged_harmful_favors_incumbent(
    *,
    candidate_fire_scores: np.ndarray,
    incumbent_fire_scores: np.ndarray,
    incumbent_fire_threshold: float,
    frame: EvalFrame,
    warranted: np.ndarray,
    root: Path | None = None,
    seed: int = JUDGE_SEED,
) -> bool:
    """Judge the disagreement fires and report whether harmful fires favor the incumbent.

    Selects the warranted disagreement fires, and — only if there are any — loads the golden packet
    (raising before any spend when it is unlabeled), proves the golden gate with a panel vote, and
    grades the disagreements through :func:`~athome.research.judge.judge_candidates`. No disagreement
    means nothing to judge, so no golden packet is loaded and no judge is spent.

    Args:
        candidate_fire_scores: Candidate higher-is-fire scores over the frame.
        incumbent_fire_scores: Incumbent higher-is-fire scores over the frame.
        incumbent_fire_threshold: The strict lower bound above which the incumbent fires.
        frame: The frozen eval frame supplying row ids and rendered contexts.
        warranted: The boolean mask selecting rows where coverage disagreements count.
        root: The frozen-eval root override the golden packet resolves under.
        seed: The position-debias seed the panel and candidate votes share.

    Returns:
        Whether judged losses outnumber judged wins over the disagreement fires.

    Raises:
        GoldenGateViolation: the golden packet is unlabeled, drifted, or the panel disagrees with
            the human labels.
    """
    fires = disagreement_fires(
        candidate_fire_scores,
        incumbent_fire_scores,
        incumbent_fire_threshold=incumbent_fire_threshold,
        warranted=warranted,
        ids=frame.ids,
        contexts=frame.tails,
    )
    if not fires:
        return False
    golden = await load_golden(golden_dir(root=root))
    judge = Judge.bind(Pairwise, backend=JUDGE_BACKEND)
    proof = prove_gate(report=agreement(golden.human, await panel_labels(judge, golden, seed=seed)), manifest=golden.manifest)
    votes = await judge_candidates(
        judge,
        tuple(judge_row(fire) for fire in fires),
        generator_family=GENERATOR_FAMILY,
        controls=control_pairs(golden.contexts),
        golden=proof,
        context=VoteContext(prompt_version=PROMPT_VERSION, digest=frame.digest),
        build_prompt=build_prompt,
        seed=seed,
        concurrency=JUDGE_CONCURRENCY,
    )
    return _favors_incumbent(votes)


def _favors_incumbent(votes: Sequence[Vote]) -> bool:
    return sum(vote is Vote.LOSS for vote in votes) > sum(vote is Vote.WIN for vote in votes)


def _read_fires(text: str) -> dict[str, str]:
    records = (json.loads(line) for line in text.splitlines() if line.strip())
    return {record["row_id"]: record["context"] for record in records}


def _packet_contexts(packet_md: str, verified: VerifiedManifest) -> dict[str, str]:
    return {
        cast("str", row["row_id"]): _window_at(packet_md, number=cast("int", row["row"]))
        for row in cast("Sequence[Mapping[str, object]]", verified.manifest["rows"])
    }


def _window_at(packet_md: str, *, number: int) -> str:
    if match := re.search(rf"## Row {number}\n\n~~~text\n(.*?)\n~~~\n", packet_md, re.DOTALL):
        return match.group(1)
    raise GoldenGateViolation(f"verified packet has no window for row {number}")
