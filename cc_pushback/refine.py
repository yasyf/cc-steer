"""Stage 3 of the pipeline: the LLM refiner that distills accepted pushback into pairs.

The judge decides which messages are genuine pushback; this stage turns each accepted
message into one or more atomic training pairs. For every distinct complaint in the
message it re-synthesizes what the assistant did, keeps the verbatim span that voices
the complaint, and distills the objection into one sentence. Pairs land in the
``refinement`` table and surface through the ``refined_pairs`` view — the deliverable.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
from cc_transcript.domains.mining import ContextSnapshot, DedupKey
from pydantic import BaseModel, Field, ValidationError

from cc_pushback.claude import resolved_model, run_claude_structured
from cc_pushback.triage import TRIGGER_TEXT_LIMIT, render_turn, render_turns

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spawnllm import TModel

    from cc_pushback.store import FeedbackStore

PROMPT_VERSION = 1

REFINE_PROMPT = """\
You are refining one accepted piece of developer PUSHBACK — corrective feedback a
developer sent to an AI coding assistant (Claude) about something it just did — into
clean, atomic training pairs.

A prior judge already confirmed this message is genuine pushback and wrote a one-line
hint about the action; treat the hint as a clue only, NOT ground truth to copy.

A single message often bundles SEVERAL distinct complaints ("no, use a generator — and
stop hardcoding the path" is two). Emit one pair per distinct complaint: at least one,
more only when the message genuinely faults more than one separable thing. Do not invent
complaints, and do not fragment a single complaint.

For EACH complaint produce:
- action: one to three sentences faithfully re-synthesizing what the assistant actually
  did that THIS complaint is about — written from the action under review (its text, tool
  calls, tool inputs), naming the concrete thing (the command, the file, the approach). It
  must stand alone. Do not merely restate the hint; do not describe the user's reaction.
- complaint_verbatim: the exact, unedited span of the USER MESSAGE voicing this complaint,
  copied character-for-character.
- complaint: one clean self-contained sentence distilling the objection.

[source: {source_kind}]
[judge's hint about the action: {what_claude_did}]
=== conversation before ===
{before}
=== assistant action under review ===
{trigger}
=== USER PUSHBACK TO REFINE ===
{text}
=== conversation after ===
{after}"""


class RefinedPair(BaseModel):
    """One atomic training pair distilled from a single complaint in a pushback message.

    Attributes:
        action: A faithful, self-contained re-synthesis of what the assistant did that
            this complaint is about.
        complaint_verbatim: The exact span of the user's message voicing this complaint.
        complaint: One clean sentence distilling the objection.
    """

    action: str
    complaint_verbatim: str
    complaint: str


class Refinement(BaseModel):
    """The atomic split of one accepted pushback message into one or more pairs.

    Attributes:
        pairs: The distinct complaints, one pair each; always at least one.
    """

    pairs: list[RefinedPair] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class RefineReport:
    """The outcome of one refine pass.

    Attributes:
        refined: How many events were split into pairs this pass.
        pairs: How many atomic pairs were written this pass.
        failed: How many events failed (timeout, parse error) and stay pending.
        pending: How many accepted events remain unrefined after this pass.
    """

    refined: int
    pairs: int
    failed: int
    pending: int


def build_refine_prompt(row: Mapping[str, object]) -> str:
    ctx = ContextSnapshot.from_json(str(row["context_json"]))
    return REFINE_PROMPT.format(
        source_kind=row["source_kind"],
        what_claude_did=row["what_claude_did"],
        before=render_turns(ctx.before),
        trigger=render_turn(ctx.trigger, TRIGGER_TEXT_LIMIT) if ctx.trigger else "(unknown)",
        text=row["text"],
        after=render_turns(ctx.after),
    )


async def run_refinements(
    store: FeedbackStore,
    rows: Sequence[Mapping[str, object]],
    *,
    prompt_version: int,
    tier: TModel,
    concurrency: int,
) -> tuple[int, int, int]:
    counts = {"refined": 0, "pairs": 0, "failed": 0}
    limiter = anyio.CapacityLimiter(concurrency)

    async def worker(row: Mapping[str, object]) -> None:
        async with limiter:
            try:
                refinement = await run_claude_structured(build_refine_prompt(row), response_model=Refinement, tier=tier)
            except (subprocess.SubprocessError, ValidationError, json.JSONDecodeError):
                counts["failed"] += 1
                return
        await store.record_refinement(
            DedupKey(str(row["dedup_key"])), refinement, prompt_version=prompt_version, model=resolved_model(tier)
        )
        counts["refined"] += 1
        counts["pairs"] += len(refinement.pairs)

    async with anyio.create_task_group() as tg:
        for row in rows:
            tg.start_soon(worker, row)
    return counts["refined"], counts["pairs"], counts["failed"]


async def refine(
    store: FeedbackStore, *, tier: TModel = "medium", limit: int | None = None, concurrency: int = 8
) -> RefineReport:
    """Refines every accepted pushback event lacking pairs at the current prompt version.

    Incremental and idempotent: each event's pairs commit atomically as soon as its
    call completes, a failed event stays unrefined and is retried on the next run, and
    re-running over a fully refined corpus is a no-op.

    Args:
        store: The open feedback store.
        tier: The refiner's abstract model tier.
        limit: When set, refine at most this many events this pass.
        concurrency: The maximum number of concurrent ``claude`` subshells.

    Returns:
        The pass's refined/pairs/failed/pending counts.
    """
    model = resolved_model(tier)
    rows = await store.unrefined(prompt_version=PROMPT_VERSION, model=model, limit=limit)
    refined, pairs, failed = await run_refinements(
        store, rows, prompt_version=PROMPT_VERSION, tier=tier, concurrency=concurrency
    )
    pending = len(await store.unrefined(prompt_version=PROMPT_VERSION, model=model))
    return RefineReport(refined=refined, pairs=pairs, failed=failed, pending=pending)
