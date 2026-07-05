"""Stage 3 of the pipeline: the LLM refiner that distills accepted steering into pairs.

The judge decides which messages are genuine steering; this stage turns each accepted
message into one or more atomic training pairs. For a corrective message it splits each
separable fault into its own pair; for a directional message (an answer to a question
the assistant asked, an option pick, a resolving directive) it emits one pair whose
direction is the choice the developer made. Each pair re-synthesizes what the assistant
did, keeps the verbatim span that voices the steering, and distills it into one
sentence. Pairs land in the ``refinement`` table and surface through the
``refined_pairs`` view — the deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cc_transcript.context import ContextWindow
from cc_transcript.judge import resolved_model, run_verdicts, structured_judge
from cc_transcript.mining import DedupKey
from pydantic import BaseModel, Field

from cc_steer.triage import question_answer_block, render_context

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spawnllm import TModel

    from cc_steer.store import FeedbackStore

PROMPT_VERSION = 3

REFINE_PROMPT = """\
You are refining one piece of developer STEERING — a message a developer sent to an AI
coding assistant (Claude) to shape what it does — into clean, atomic training pairs.

Steering is broader than correction. It is either:
  (a) corrective — the developer faults something the assistant just did (its plan, a
      change it made, an action it took), or
  (b) directional — the developer resolves a choice the assistant faced or raised:
      answering a question it asked, picking an option it offered, or a directive that
      settles an open decision.

A prior judge already confirmed this message is genuine steering and wrote a one-line
hint about the action; treat the hint as a clue only, NOT ground truth to copy.

Emit one pair per distinct piece of steering, at least one:
- A corrective message often bundles SEVERAL separable faults ("no, use a generator —
  and stop hardcoding the path" is two): emit one pair each. Do not invent faults, and
  do not fragment a single one.
- A directional message — an answer to the assistant's question, an option pick, a
  single resolving directive — is atomic: emit exactly ONE pair. When a
  `=== QUESTION THE ASSISTANT ASKED ===` block appears below, the action is the decision
  the assistant put to the developer and the direction is the answer it received; carry
  the resolved option into the direction so the pair stands alone without the question.

For EACH pair produce:
- action: one to three sentences faithfully re-synthesizing what the assistant actually
  did that THIS pair is about — the change, command, or approach under review, or the
  decision it raised — written from the action itself (its text, tool calls, tool
  inputs), naming the concrete thing. It must stand alone. Do not merely restate the
  hint; do not describe the developer's reaction.
- direction_verbatim: the exact, unedited span of the USER MESSAGE voicing this steering,
  copied character-for-character.
- direction: one clean self-contained sentence distilling what the developer wants — the
  correction to make or the choice to follow.

[source: {source_kind}]
[judge's hint about the action: {what_claude_did}]
{context}
{question_answer}=== USER STEERING TO REFINE ===
{text}"""


class RefinedPair(BaseModel):
    """One atomic training pair distilled from a single piece of steering.

    Attributes:
        action: A faithful, self-contained re-synthesis of what the assistant did — or
            the decision it raised — that this pair is about.
        direction_verbatim: The exact span of the user's message voicing this steering.
        direction: One clean sentence distilling the correction to make or choice to follow.
    """

    action: str
    direction_verbatim: str
    direction: str


class Refinement(BaseModel):
    """The atomic split of one accepted steering message into one or more pairs.

    Attributes:
        pairs: The distinct pieces of steering, one pair each; always at least one.
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


async def build_refine_prompt(row: Mapping[str, object]) -> str:
    context, _ = await render_context(ContextWindow.from_json(str(row["context_json"])))
    return REFINE_PROMPT.format(
        source_kind=row["source_kind"],
        what_claude_did=row["what_claude_did"],
        context=context,
        question_answer=question_answer_block(row),
        text=row["text"],
    )


async def run_refinements(
    store: FeedbackStore,
    rows: Sequence[Mapping[str, object]],
    *,
    prompt_version: int,
    tier: TModel,
    concurrency: int,
) -> tuple[int, int, int]:
    pairs = 0

    async def persist(row: Mapping[str, object], refinement: Refinement) -> None:
        nonlocal pairs
        await store.record_refinement(
            DedupKey(str(row["dedup_key"])), refinement, prompt_version=prompt_version, model=resolved_model(tier)
        )
        pairs += len(refinement.pairs)

    refined, failed = await run_verdicts(
        rows, build_refine_prompt, structured_judge(Refinement, tier=tier), persist, concurrency=concurrency
    )
    return refined, pairs, failed


async def refine(
    store: FeedbackStore, *, tier: TModel = "medium", limit: int | None = None, concurrency: int = 8
) -> RefineReport:
    """Refines every accepted steering event lacking pairs at the current prompt version.

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
