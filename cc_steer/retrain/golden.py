"""Author the blind watcher-fires golden packet the judged gate and the warrant audit both consume.

The judged promotion gate (:mod:`cc_steer.retrain.judged`) spends opus votes only behind a
human-labeled golden packet under ``~/.cc-steer/eval/golden/watcher-fires/``. This module mints
that packet: it draws a deterministic diagnostic sample of the frozen eval frame, renders a
sub-sample of it as the blind labeling document through :func:`athome.research.golden.build_packet`,
and writes it beside the fires sidecar :func:`~cc_steer.retrain.judged.load_golden` binds against.
Zero LLM calls — it only reads the frame, the registry-current incumbent's stored probs, and renders.

The sample is two-staged and reproducible from ``(frame, incumbent probs, seed)``. Stage one
(:func:`audit_sample`) selects a fire-diagnostic population: every warranted row and every incumbent
fire, plus a rank-stratified draw of prose negatives (weighted toward the scores the incumbent is
most likely to wrongly fire on) and a random draw of the remaining true-steer positives. Stage two
draws the blind packet — 15 warranted and 15 negative rows — from within that population. The audit
provenance (seed, sample size, incumbent version and fire threshold, realized stratum counts) is
stamped into the manifest outside its hashed regions, so :func:`~cc_steer.retrain.audit.run_warrant_audit`
can re-derive the identical sample and prove the labeled rows came from it. Donor: cc-steer-lab
``e10_golden.py`` (the E10.A protocol prose and blindness rules).
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import numpy as np
from athome.research.golden import (
    MANIFEST_NAME,
    PACKET_NAME,
    GoldenGateViolation,
    Stratum,
    build_packet,
    verify_packet,
    write_packet,
)

from cc_steer import registry
from cc_steer.retrain import evalset, judged
from cc_steer.watcher.drafter_mlx import COMPONENT as WATCHER_COMPONENT

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from athome.research.golden import GoldenPacket

    from cc_steer.retrain.evalset import EvalFrame

WARRANTED = "warranted"
FIRED = "fired"
NEGATIVE = "negative"
OTHER_POSITIVE = "other-positive"
STRATUM_NAMES: tuple[str, ...] = (WARRANTED, FIRED, NEGATIVE, OTHER_POSITIVE)

SEED = 1729
SAMPLE_N = 200
NEGATIVE_SHARE = 0.64
RANK_BANDS = 4
BAND_WEIGHTS: tuple[int, ...] = (4, 3, 2, 1)

PACKET_STRATA: tuple[Stratum, ...] = (Stratum(name=WARRANTED, size=15), Stratum(name=NEGATIVE, size=15))
GATE_N = sum(stratum.size for stratum in PACKET_STRATA)
GATE_FLOOR = 24
FIRES_NAME = judged.FIRES_NAME
LABELS_NAME = judged.LABELS_NAME
README_NAME = "README.md"

QUESTION = "Was steering the assistant warranted at this point?"
FRAME_DESCRIPTION = "judgeable frozen watcher-eval rows (render-v2 tail, non-empty context)"

PACKET_HEADER = f"""# Watcher-Fires Golden Labeling Packet

Blind human warrant labels — read `{README_NAME}` first. {GATE_N} rows, one section each. For every
row, read the context window (exactly what the watcher saw at one decision point) and answer the same
question. You will NOT see whether a steer fired, any draft, or the correction the user actually made
— only the situation leading up to it.

**The question for every row:** *{QUESTION}*

- `yes` = a supervisor watching this session SHOULD have intervened at this point.
- `no` = no intervention was warranted; the assistant was on track (or there was nothing to act on yet).
- When the window ends with the assistant asking the user something, judge whether answering or giving
  direction was warranted.

Record answers by row number in `{LABELS_NAME}` (copy `labels_template.json` to `{LABELS_NAME}` and
fill each `label`). Do not open `{MANIFEST_NAME}` or read the `row_id` values until every label is in.

---
"""


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One row of the diagnostic audit sample.

    Attributes:
        index: The row's position in the frozen eval frame (indexes ``frame.tails``).
        row_id: The frame row id, the opaque provenance key linking back to the source row.
        stratum: The pool the row was drawn from — one of :data:`STRATUM_NAMES`.
    """

    index: int
    row_id: str
    stratum: str


class GoldenAuthorError(RuntimeError):
    """Authoring the golden packet cannot proceed: no incumbent, or a labeled packet already exists."""


def incumbent_fire(frame: EvalFrame, *, root: Path | None = None) -> tuple[str, np.ndarray, float]:
    """The registry-current watcher's higher-is-fire scores over the frame and its strict fire threshold.

    Reads the promoted incumbent's stored ``P(NO_STEER)`` probs (never rescoring), orienting them to
    fire scores (``1 - p``) and its budget cut to the strict fire lower bound (``1 - budget``), so the
    incumbent fires exactly where ``fire_score > threshold``.

    Raises:
        GoldenAuthorError: no promoted watcher incumbent exists to score the sample against.
    """
    incumbent = registry.current(WATCHER_COMPONENT)
    if incumbent is None:
        raise GoldenAuthorError("no promoted watcher incumbent to sample against; register and promote one first")
    render = int(str(incumbent.metadata["render_version"]))
    probs = evalset.load_probs(frame, incumbent.version, expected_render=render, root=root)
    return incumbent.version, 1.0 - probs, 1.0 - float(incumbent.metadata["thresholds"]["budget"])


def audit_pools(frame: EvalFrame, incumbent_fire_scores: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Partition the frame into the four disjoint diagnostic pools, in priority order.

    A row belongs to exactly one pool: warranted (a true prose-corrective steer) first, then an
    incumbent fire, then a prose negative, then a remaining true-steer positive. The incumbent fires
    strictly above ``threshold`` on higher-is-fire scores.
    """
    warranted = frame.corrective & frame.prose
    fired = incumbent_fire_scores > threshold
    return {
        WARRANTED: np.flatnonzero(warranted),
        FIRED: np.flatnonzero(fired & ~warranted),
        NEGATIVE: np.flatnonzero(frame.prose & ~frame.labels & ~fired),
        OTHER_POSITIVE: np.flatnonzero(frame.labels & ~warranted & ~fired),
    }


def audit_sample(
    frame: EvalFrame, incumbent_fire_scores: np.ndarray, threshold: float, *, n: int = SAMPLE_N, seed: int = SEED
) -> tuple[AuditRow, ...]:
    """One deterministic diagnostic draw of the frame: take-all warranted and fired, filled with negatives and positives.

    Warranted and fired rows are taken whole. The remaining budget (``n`` minus the take-all counts)
    fills with prose negatives — rank-stratified by incumbent fire score and weighted toward the top
    band the incumbent is most likely to wrongly fire on — and the rest with random true-steer
    positives. Every draw clamps to its pool, so a thinner frame yields a smaller sample rather than
    raising; the negative draw never drops below the blind packet's negative stratum (pool
    permitting), so a warranted-heavy frame may overshoot ``n`` slightly instead of starving the
    packet. Fully reproducible from ``(frame, incumbent_fire_scores, threshold, n, seed)``.
    """
    pools = audit_pools(frame, incumbent_fire_scores, threshold)
    rng = np.random.default_rng(seed)
    residual = max(0, n - len(pools[WARRANTED]) - len(pools[FIRED]))
    packet_negatives = next(stratum.size for stratum in PACKET_STRATA if stratum.name == NEGATIVE)
    negatives = _rank_stratified(
        pools[NEGATIVE],
        incumbent_fire_scores,
        min(len(pools[NEGATIVE]), max(round(residual * NEGATIVE_SHARE), packet_negatives)),
        rng=rng,
    )
    others = _random_draw(pools[OTHER_POSITIVE], min(len(pools[OTHER_POSITIVE]), max(0, residual - len(negatives))), rng=rng)
    return tuple(
        AuditRow(index=int(index), row_id=frame.ids[int(index)], stratum=stratum)
        for stratum, indices in ((WARRANTED, pools[WARRANTED]), (FIRED, pools[FIRED]), (NEGATIVE, negatives), (OTHER_POSITIVE, others))
        for index in indices
    )


def stratum_counts(sample: tuple[AuditRow, ...]) -> dict[str, int]:
    """The realized row count per stratum in a draw, every :data:`STRATUM_NAMES` key present."""
    counts = Counter(row.stratum for row in sample)
    return {name: counts[name] for name in STRATUM_NAMES}


async def author_packet(*, root: Path | None = None) -> Path:
    """Author the blind watcher-fires golden packet under the frozen-eval root and return its directory.

    Draws the audit sample against the registry-current incumbent, renders 15 warranted and 15
    negative rows from within it as the blind packet, tightens the agreement gate to 24/30, stamps
    the audit provenance into the manifest outside its hashed regions, and writes the packet, its
    fires sidecar, and the labeling ``README.md`` beside them. Before returning it re-reads the
    written packet and proves the fires sidecar binds to it exactly as
    :func:`~cc_steer.retrain.judged.load_golden` will — a fence collision in any window fails here,
    loud, rather than at gate time.

    Raises:
        GoldenAuthorError: a human ``labels.json`` already exists (refusing to clobber human work),
            or no promoted incumbent exists.
        GoldenGateViolation: the written packet does not round-trip — an extracted window differs
            from the sidecar context it should equal.
    """
    frame = evalset.EvalFrame.load(root=root)
    directory = judged.golden_dir(root=root)
    if (directory / LABELS_NAME).exists():
        raise GoldenAuthorError(f"{directory / LABELS_NAME} exists; refusing to overwrite a labeled packet")
    version, fire_scores, threshold = incumbent_fire(frame, root=root)
    sample = audit_sample(frame, fire_scores, threshold, n=SAMPLE_N, seed=SEED)
    packet = _stamp_audit(
        build_packet(
            [{"row_id": row.row_id, "stratum": row.stratum, "window": frame.tails[row.index]} for row in sample],
            strata=PACKET_STRATA,
            stratum_of=lambda row: row["stratum"],
            window_of=lambda row: row["window"],
            row_id=lambda row: row["row_id"],
            seed=SEED,
            dataset_digest=frame.digest,
            question=QUESTION,
            header=PACKET_HEADER,
        ),
        version=version,
        threshold=threshold,
        counts=stratum_counts(sample),
    )
    await write_packet(packet, anyio.Path(directory))
    (directory / FIRES_NAME).write_text("".join(json.dumps({"row_id": row.row_id, "context": row.window}) + "\n" for row in packet.rows))
    (directory / README_NAME).write_text(_render_readme())
    _round_trip(directory)
    return directory


async def verify_golden(*, root: Path | None = None) -> judged.Golden:
    """Load and verify the authored golden packet through the real judged loader — zero spend.

    Thin wrapper over :func:`~cc_steer.retrain.judged.load_golden`: it requires the human ``labels.json``,
    verifies the packet against its manifest, and re-binds the fires sidecar to the packet windows.

    Raises:
        GoldenGateViolation: the packet is unlabeled, drifted from its manifest, or the sidecar is
            not bound to the verified packet windows.
    """
    return await judged.load_golden(judged.golden_dir(root=root))


def _stamp_audit(packet: GoldenPacket, *, version: str, threshold: float, counts: Mapping[str, int]) -> GoldenPacket:
    audit = {
        "seed": SEED,
        "n": SAMPLE_N,
        "incumbent_version": version,
        "incumbent_fire_threshold": threshold,
        "stratum_counts": dict(counts),
    }
    return dataclasses.replace(packet, manifest=packet.manifest | {"gate": {"n": GATE_N, "floor": GATE_FLOOR}, "audit": audit})


def _round_trip(directory: Path) -> None:
    packet_md = (directory / PACKET_NAME).read_text()
    verified = verify_packet(packet_md=packet_md, manifest=json.loads((directory / MANIFEST_NAME).read_text()))
    if (sidecar := judged._read_fires((directory / FIRES_NAME).read_text())) != (bound := judged._packet_contexts(packet_md, verified)):
        raise GoldenGateViolation(
            f"authored packet does not round-trip: rows {sorted(rid for rid in sidecar if sidecar.get(rid) != bound.get(rid))} "
            "extract a window differing from their fires sidecar context — a window collides with the '~~~' fence or a '## Row N' heading"
        )


def _rank_stratified(pool: np.ndarray, scores: np.ndarray, take: int, *, rng: np.random.Generator) -> np.ndarray:
    if take >= len(pool):
        return pool
    ranked = pool[np.argsort(scores[pool])[::-1]]
    bands = np.array_split(ranked, min(RANK_BANDS, len(ranked)))
    chosen: list[int] = []
    leftover: list[int] = []
    for band, budget in zip(bands, _apportion(take, BAND_WEIGHTS[: len(bands)]), strict=True):
        picked = {int(band[i]) for i in rng.choice(len(band), size=min(budget, len(band)), replace=False)} if len(band) else set()
        chosen.extend(sorted(picked))
        leftover.extend(int(index) for index in band if int(index) not in picked)
    return np.array(chosen + leftover[: take - len(chosen)], dtype=int)


def _random_draw(pool: np.ndarray, take: int, *, rng: np.random.Generator) -> np.ndarray:
    return pool if take >= len(pool) else pool[np.sort(rng.choice(len(pool), size=take, replace=False))]


def _apportion(total: int, weights: tuple[int, ...]) -> list[int]:
    exact = [total * weight / sum(weights) for weight in weights]
    base = [int(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def _render_readme() -> str:
    return f"""# Watcher-Fires Golden Gate — Labeling Protocol

This is the blind human labeling that validates the watcher judge panel **before** it spends on any
model output. You are the ground truth the panel is measured against, so label from the situation
alone — never from what the model or the user did next.

## What you do

1. Open `{PACKET_NAME}`. It has {GATE_N} numbered rows. Each row shows one context window: the exact
   text the watcher saw at a decision point in a real Claude Code session.
2. For each row, answer one question:

   > *{QUESTION}*

3. Copy `labels_template.json` to `{LABELS_NAME}` and record answers keyed by row number: set `label`
   to `"yes"` or `"no"`.

## What "warranted" means

Answer **`yes`** when a supervisor watching this session in real time *should* have intervened at this
point — independent of whether anyone did, and of what the assistant went on to do. You are judging
the merit of intervening here, not grading an outcome. Answer **`no`** when no intervention was
warranted: the assistant was on a reasonable track, or there was nothing yet to act on.

## Blindness rules (do not break)

- Label strictly from the window shown in `{PACKET_NAME}`.
- Do **not** open `{MANIFEST_NAME}`, and do not read or reason about the `row_id` strings — they are
  opaque provenance keys and carry no signal you should use.
- Do not look for how the session continued; that information is deliberately withheld.

## What happens next

After you submit all {GATE_N} labels, a cross-family judge panel labels the same rows. The panel is
accepted only if it agrees with you on **>= {GATE_FLOOR}/{GATE_N}** rows and is not a constant decider
(a panel answering all-`yes` or all-`no` fails). If the panel fails this gate, no downstream judge
spend proceeds until the panel is fixed.

Rows: {PACKET_STRATA[0].size} warranted, {PACKET_STRATA[1].size} negative — interleaved in random
order. Frame: {FRAME_DESCRIPTION}. Seed {SEED}.
"""
