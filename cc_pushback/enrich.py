"""Stage 4 of the pipeline: ground each refined pair in the code it complains about.

The refine stage distills accepted pushback into atomic complaint pairs; this stage
links each pair to concrete code evidence. It harvests candidate incorrect edits —
and the corrections that later overwrote them, from the same session or from git
history — around the pushback anchor via :mod:`cc_transcript.evidence`, then asks an
LLM to pick the one edit the complaint faults, copied verbatim. Two outcomes are
deterministic and free: an expired transcript and an editless lookback window both
persist ``no_code`` sentinels without an LLM call. Evidence lands in the
``pair_evidence`` table, keyed to the refine generation it annotates, and surfaces
through the ``refined_pairs`` view.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio
import anyio.to_thread
from cc_transcript.activity import SessionActivity, meta_of
from cc_transcript.discovery import TranscriptExpiredError
from cc_transcript.evidence import EXTRACTOR_VERSION, GitFix, harvest_pairs
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.judge import resolved_model, structured_judge
from cc_transcript.mining import DedupKey
from cc_transcript.render import Budget, clip, hunk_lines
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from cc_transcript.activity import Turn
    from cc_transcript.evidence import CandidatePair
    from cc_transcript.tools import Hunk
    from spawnllm import TModel

    from cc_pushback.store import FeedbackStore

ENRICH_VERSION = 1
SENTINEL_INDEX = -1
HUNK_CHARS = 600
HUNK_BUDGET = Budget(tool_chars=HUNK_CHARS)
EXPIRED_NOTE = "transcript expired before enrichment"
NO_ANCHOR_NOTE = "candidate carries no anchor event"
COMPACTED_NOTE = "anchor compacted away within the transcript"
NO_EDITS_NOTE = "no edits in the lookback window"

type Source = Literal["session", "git"]

ENRICH_PROMPT = """\
You are grounding one piece of accepted developer PUSHBACK — an atomic complaint a
developer made about an AI coding assistant's work — in the concrete code change it
complains about.

Below are candidate edits the assistant made shortly before the pushback, newest
first. Each shows the edit's before (-) and after (+) content and, when one was
found, the correction that later overwrote it — from the same session, or from git
history (marked "git <sha>"). The correction ranked most likely by content overlap
is tagged [likely fix].

Decide which single candidate edit THIS complaint faults, if any:
- kind: "code" when exactly one candidate's after-content is what the complaint is
  about; "no_code" when the complaint is not about any of these edits (it may fault
  the approach, a command, or work outside this window).
- file_path: the chosen candidate's file path, verbatim.
- incorrect_edit: the chosen candidate's before/after content as old/new, copied
  character-for-character from its block (without the "- "/"+ " prefixes). Never
  invent, merge, or edit content.
- correct_edit: the chosen candidate's correction old/new, copied the same way, or
  null when it shows no correction.
- note: one short clause explaining the call.

[what the assistant did: {action}]
[the complaint: {complaint}]
=== CANDIDATE EDITS ===
{candidates}
=== USER PUSHBACK (verbatim) ===
{text}"""


class EditSide(BaseModel):
    """One edit's before/after content, copied verbatim from a candidate block.

    Attributes:
        old: The content the edit replaced.
        new: The content the edit wrote.
    """

    old: str
    new: str


class CodeEvidence(BaseModel):
    """The code evidence linking one refined pair to the edit it complains about.

    Attributes:
        kind: ``'code'`` when one candidate edit is what the complaint faults,
            ``'no_code'`` when the complaint is not about any harvested edit.
        file_path: The chosen candidate's file path; None for ``'no_code'``.
        incorrect_edit: The chosen candidate's verbatim old/new content.
        correct_edit: The verbatim correction that overwrote it, when one exists.
        note: One short clause explaining the call.
    """

    kind: Literal["code", "no_code"]
    file_path: str | None = None
    incorrect_edit: EditSide | None = None
    correct_edit: EditSide | None = None
    note: str

    @model_validator(mode="after")
    def code_carries_an_edit(self) -> CodeEvidence:
        if self.kind == "code" and (self.file_path is None or self.incorrect_edit is None):
            raise ValueError("kind='code' requires file_path and incorrect_edit")
        return self


@dataclass(frozen=True, slots=True)
class EnrichReport:
    """The outcome of one enrich pass.

    Attributes:
        enriched: How many pair rows resolved to evidence this pass.
        code: How many of those grounded the complaint in a concrete edit.
        no_code: How many resolved as no-code (free sentinel or judged).
        git: How many code rows carry a git-history correction.
        failed: How many rows failed (timeout, parse error) and stay pending.
        pending: How many pair rows remain unenriched after this pass.
    """

    enriched: int
    code: int
    no_code: int
    git: int
    failed: int
    pending: int


def hunk_block(hunks: Sequence[Hunk]) -> str:
    return "\n".join(line for hunk in hunks for line in hunk_lines(hunk.old, hunk.new, budget=HUNK_BUDGET))


def likely_fix(pairs: Sequence[CandidatePair]) -> CandidatePair | None:
    best = max(pairs, key=lambda pair: pair.overlap)
    return best if best.correction is not None and best.overlap > 0 else None


def correction_header(pair: CandidatePair, *, likely: bool) -> str:
    match pair.correction:
        case None:
            return "no correction found"
        case GitFix(commit=commit):
            head = f"correction (git {commit}, overlap {pair.overlap:.2f})"
        case correction:
            turns = correction.turn_index - pair.incorrect.turn_index
            head = f"correction (same session, {turns} turn(s) later, overlap {pair.overlap:.2f})"
    return f"{head} [likely fix]:" if likely else f"{head}:"


def candidate_block(index: int, pair: CandidatePair, *, anchor_turn: int, likely: bool) -> str:
    return "\n".join(
        (
            f"--- candidate {index}: {pair.incorrect.file_path} "
            f"({pair.incorrect.tool}, {anchor_turn - pair.incorrect.turn_index} turn(s) before the pushback) ---",
            hunk_block(pair.incorrect.hunks),
            correction_header(pair, likely=likely),
            *(() if pair.correction is None else (hunk_block(pair.correction.hunks),)),
        )
    )


def build_enrich_prompt(row: Mapping[str, object], pairs: Sequence[CandidatePair], *, anchor_turn: int) -> str:
    likely = likely_fix(pairs)
    return ENRICH_PROMPT.format(
        action=row["action"],
        complaint=row["complaint"],
        candidates="\n\n".join(
            candidate_block(index, pair, anchor_turn=anchor_turn, likely=pair is likely)
            for index, pair in enumerate(pairs, 1)
        ),
        text=row["complaint_verbatim"],
    )


def correction_source(pairs: Sequence[CandidatePair], evidence: CodeEvidence) -> Source | None:
    if evidence.correct_edit is None:
        return None
    forms = {
        form
        for pair in pairs
        if isinstance(pair.correction, GitFix)
        for hunk in pair.correction.hunks
        for form in ((hunk.old, hunk.new), (clip(hunk.old, HUNK_CHARS), clip(hunk.new, HUNK_CHARS)))
    }
    return "git" if (evidence.correct_edit.old, evidence.correct_edit.new) in forms else "session"


def repo_of(turn: Turn, anchor: EventRef) -> Path | None:
    return next(
        (
            Path(meta.cwd)
            for event in turn.events
            if (meta := meta_of(event)) is not None and meta.uuid == anchor.event_uuid and meta.cwd
        ),
        None,
    )


async def resolve_evidence(
    row: Mapping[str, object], judge: Callable[[str], Awaitable[CodeEvidence]]
) -> tuple[CodeEvidence, Source | None, int]:
    match row["session_id"], row["event_uuid"]:
        case (None, _) | (_, None):
            return CodeEvidence(kind="no_code", note=NO_ANCHOR_NOTE), None, SENTINEL_INDEX
        case (session_id, event_uuid):
            anchor = EventRef(SessionId(str(session_id)), EventUuid(str(event_uuid)))
    try:
        activity = await SessionActivity.from_session(anchor.session_id)
    except TranscriptExpiredError:
        return CodeEvidence(kind="no_code", note=EXPIRED_NOTE), None, SENTINEL_INDEX
    if (turn := activity.turn_of(anchor)) is None:
        return CodeEvidence(kind="no_code", note=COMPACTED_NOTE), None, SENTINEL_INDEX
    pairs = await anyio.to_thread.run_sync(partial(harvest_pairs, activity, anchor, repo=repo_of(turn, anchor)))
    if not pairs:
        return CodeEvidence(kind="no_code", note=NO_EDITS_NOTE), None, SENTINEL_INDEX
    evidence = await judge(build_enrich_prompt(row, pairs, anchor_turn=turn.index))
    return evidence, correction_source(pairs, evidence), int(str(row["pair_index"]))


async def run_enrichments(
    store: FeedbackStore,
    rows: Sequence[Mapping[str, object]],
    *,
    enrich_version: int,
    tier: TModel,
    concurrency: int,
) -> tuple[int, int, int, int]:
    judge = structured_judge(CodeEvidence, tier=tier)
    model = resolved_model(tier)
    counts = {"code": 0, "no_code": 0, "git": 0, "failed": 0}
    limiter = anyio.CapacityLimiter(concurrency)

    async def worker(row: Mapping[str, object]) -> None:
        async with limiter:
            try:
                evidence, source, pair_index = await resolve_evidence(row, judge)
            except Exception:
                counts["failed"] += 1
                return
        await store.record_evidence(
            DedupKey(str(row["dedup_key"])),
            evidence,
            refine_version=int(str(row["refine_version"])),
            refine_model=str(row["refine_model"]),
            pair_index=pair_index,
            enrich_version=enrich_version,
            enrich_model=model,
            extractor_version=EXTRACTOR_VERSION,
            source=source,
        )
        counts[evidence.kind] += 1
        counts["git"] += source == "git"

    async with anyio.create_task_group() as tg:
        for row in rows:
            tg.start_soon(worker, row)
    return counts["code"], counts["no_code"], counts["git"], counts["failed"]


async def enrich(
    store: FeedbackStore, *, tier: TModel = "medium", limit: int | None = None, concurrency: int = 8
) -> EnrichReport:
    """Enriches every refined pair lacking code evidence at the current versions.

    Incremental and idempotent: each pair's evidence persists as soon as its row
    resolves, a failed pair stays unenriched and is retried on the next run, and
    re-running over a fully enriched corpus is a no-op. Two outcomes cost no LLM
    call: an expired transcript and an editless lookback window both persist a
    ``no_code`` sentinel (``pair_index=-1``) covering the whole refine generation.
    Evidence keys to the refine generation it annotates, so a refine re-run
    resurfaces its new pairs here automatically; so does bumping the platform's
    :data:`~cc_transcript.evidence.EXTRACTOR_VERSION`.

    Args:
        store: The open feedback store.
        tier: The linking judge's abstract model tier.
        limit: When set, enrich at most this many pairs this pass.
        concurrency: The maximum number of concurrent ``claude`` subshells.

    Returns:
        The pass's enriched/code/no_code/git/failed/pending counts.
    """
    model = resolved_model(tier)
    rows = await store.unenriched(
        enrich_version=ENRICH_VERSION, enrich_model=model, extractor_version=EXTRACTOR_VERSION, limit=limit
    )
    code, no_code, git, failed = await run_enrichments(
        store, rows, enrich_version=ENRICH_VERSION, tier=tier, concurrency=concurrency
    )
    pending = len(
        await store.unenriched(enrich_version=ENRICH_VERSION, enrich_model=model, extractor_version=EXTRACTOR_VERSION)
    )
    return EnrichReport(enriched=code + no_code, code=code, no_code=no_code, git=git, failed=failed, pending=pending)
