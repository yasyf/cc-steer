"""Stage 2 of the pipeline: the LLM judge that turns mined candidates into training pairs.

The deterministic detectors are tuned for recall; this module supplies the precision.
A judge classifies every stored candidate into a pushback or noise category, and an
independently-prompted auditor (a stronger model, blind to the judge's verdicts)
samples the results so the judge's error rate is measurable. Verdicts land in the
``triage`` table; accepted rows surface through the ``training_pairs`` view.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from itertools import zip_longest
from random import Random
from typing import TYPE_CHECKING, Literal

import anyio
from cc_transcript.domains.mining import ContextSnapshot, DedupKey
from pydantic import BaseModel, Field, ValidationError

from cc_pushback.claude import resolved_model, run_claude_structured

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cc_transcript.domains.mining import ContextTurn
    from spawnllm import TModel

    from cc_pushback.store import FeedbackStore

PROMPT_VERSION = 2
AUDIT_VERSION = 2
JUDGE = "judge"
AUDITOR = "auditor"
TURN_TEXT_LIMIT = 700
TRIGGER_TEXT_LIMIT = 2000
OVERSAMPLE_SHARE = 0.3
KIND_QUOTAS: dict[str, int | None] = {"interrupt_rejection": None, "review_comment": 10, "plan_review": 10}
REMAINDER_KIND = "transcript_message"

PUSHBACK_CATEGORIES = frozenset(
    {"wrong_approach", "incorrect_change", "unwanted_action", "style_violation", "premature"}
)

Category = Literal[
    "wrong_approach",
    "incorrect_change",
    "unwanted_action",
    "style_violation",
    "premature",
    "operational_directive",
    "status_update",
    "new_task",
    "question",
    "other",
]

JUDGE_PROMPT = """\
You are auditing one message a developer sent to an AI coding assistant (Claude),
deciding whether it is genuine PUSHBACK — corrective feedback on something the
assistant just did — or non-pushback noise.

Pick exactly one category:
- wrong_approach: rejects the assistant's plan, strategy, or design.
- incorrect_change: the code or content the assistant produced is wrong or broken.
- unwanted_action: the assistant did something the user did not ask for or want.
- style_violation: the work violates the user's conventions or stated preferences.
- premature: the assistant stopped early, skipped work, or claimed completion when work remains.
- operational_directive: a forward instruction ("commit and push", "set a higher timeout")
  that does not criticize prior work. Approving then adding scope ("yes — also do Y",
  "upgrade our alias as well") is operational; forward words like "fix" or "review"
  aimed at future work do not by themselves fault what was already done.
- status_update: the user reporting state ("done, its running", "I killed it already").
- new_task: a fresh request or spec, not a reaction to the preceding action. A report
  that an external tool or pre-existing system is broken is new_task or status_update,
  not incorrect_change, unless the assistant built it.
- question: a genuine request for information. A skeptical or rhetorical question that
  presses on a choice the assistant made ("why are we hardcoding this?", "should this
  ever be optional?", "there's really no better way?") is pushback, not a question —
  categorize it by what it challenges.
- other: none of the above.

The first five categories are pushback; the rest are noise.
A mixed message that contains ANY genuine corrective content is pushback — pick the
category of the corrective part. Corrective content is often implicit in a directive:
"figure out the right proper fix" faults the current fix, "look more closely" faults a
shallow look, "not just X — give me Y" faults an insufficient answer, and exasperation
("stop wasting time") faults the behavior, even when phrased as the next step.
The assistant action being corrected may predate the trigger shown: when the message
critiques files or output the assistant produced earlier in the session, it is pushback
on that work even if the immediately preceding action is unrelated.
When the source is review_comment, the message is an inline code-review comment on code
the assistant wrote: terse imperatives there ("inline", "remove this one", "maybe make
this _safe?") are corrections — usually style_violation, incorrect_change, or
wrong_approach — not operational directives.

what_claude_did: ONE neutral sentence naming the assistant action the message responds
to (e.g. "Force-pushed to the shared branch with git push --force"). Write it even when
the message is noise.
confidence: your probability (0 to 1) that your pushback-vs-noise call is correct.
rationale: one short clause.

[source: {source_kind}]
=== conversation before ===
{before}
=== assistant action under review ===
{trigger}
=== USER MESSAGE TO CLASSIFY ===
{text}
=== conversation after ===
{after}"""

AUDIT_PROMPT = """\
A dataset is being built of developer corrections: moments where a human, reading what
an AI coding assistant just did, told it that something about that work was wrong,
unwanted, or off-course. You are the quality gate: given one human message and its
surrounding conversation, decide independently whether the message belongs in that
dataset.

It belongs (it is a correction) when the message faults the assistant's preceding work
or behavior in any way — its direction, its output, its side effects, its style, or its
stopping point — even partially, even alongside unrelated content.

It does not belong when the message only moves work forward or reports facts: telling
the assistant what to do next, giving a new assignment, asking or answering a question,
relaying status, or approving. A message that raises a doubt but ends by accepting the
assistant's choice ("nevermind, if there's a reason for it, then do it") moves work
forward and does not belong.

Two sharp edges. An inline code-review comment (source review_comment) annotates a
specific line of code the assistant authored: short imperatives and suggestions there
("inline", "it is required", "use dataclasses always") fault that line and belong in
the dataset. And a question can go either way: one that presses on a choice ("couldn't
this inherit from userlist?") faults the work, while one that merely seeks information
does not.

Choose the single best-fitting label:
- wrong_approach / incorrect_change / unwanted_action / style_violation / premature (corrections)
- operational_directive / status_update / new_task / question / other (not corrections)

Also provide: what_claude_did — one neutral sentence describing the assistant's
preceding action; confidence — your probability (0 to 1) that your correction-vs-not
call is right; rationale — one short clause.

[source: {source_kind}]
=== conversation before ===
{before}
=== assistant's preceding action ===
{trigger}
=== HUMAN MESSAGE TO ASSESS ===
{text}
=== conversation after ===
{after}"""


class Verdict(BaseModel):
    """One triage verdict on a stored feedback candidate.

    Attributes:
        category: The single best-fitting pushback or noise category.
        what_claude_did: One neutral sentence naming the assistant action the
            message responds to.
        confidence: The model's probability that its pushback-vs-noise call is right.
        rationale: One short clause explaining the call.
    """

    category: Category
    what_claude_did: str
    confidence: float = Field(ge=0, le=1)
    rationale: str

    @property
    def is_pushback(self) -> bool:
        """Whether the category marks genuine pushback."""
        return self.category in PUSHBACK_CATEGORIES


@dataclass(frozen=True, slots=True)
class TriageReport:
    """The outcome of one triage or audit pass.

    Attributes:
        judged: How many rows received a verdict this pass.
        failed: How many rows failed (timeout, parse error) and stay pending.
        pending: How many rows remain unjudged after this pass.
    """

    judged: int
    failed: int
    pending: int


@dataclass(frozen=True, slots=True)
class AuditSample:
    """The seeded audit draw over one prompt version's judged rows.

    Attributes:
        core: Uniform draws — the only rows entering headline precision metrics.
        oversample: Lowest-judge-confidence draws — diagnosis fuel only.
    """

    core: tuple[Mapping[str, object], ...]
    oversample: tuple[Mapping[str, object], ...]


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_turn(turn: ContextTurn, limit: int = TURN_TEXT_LIMIT) -> str:
    tools = "".join(
        f"\n  {name}({clip(input, limit)})" if input else f"\n  {name}()"
        for name, input in zip_longest(turn.tool_calls, turn.tool_inputs, fillvalue="")
    )
    return f"{turn.role}: {clip(turn.text, limit)}{tools}"


def render_turns(turns: Sequence[ContextTurn]) -> str:
    return "\n".join(render_turn(turn) for turn in turns) or "(none)"


def build_prompt(template: str, row: Mapping[str, object]) -> str:
    ctx = ContextSnapshot.from_json(str(row["context_json"]))
    return template.format(
        source_kind=row["source_kind"],
        before=render_turns(ctx.before),
        trigger=render_turn(ctx.trigger, TRIGGER_TEXT_LIMIT) if ctx.trigger else "(unknown)",
        text=row["text"],
        after=render_turns(ctx.after),
    )


def stratified(
    rows: Sequence[Mapping[str, object]], n: int, rng: Random
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    by_kind: dict[str, list[Mapping[str, object]]] = {}
    for row in sorted(rows, key=lambda r: str(r["dedup_key"])):
        by_kind.setdefault(str(row["source_kind"]), []).append(row)
    core: list[Mapping[str, object]] = []
    oversample: list[Mapping[str, object]] = []
    spent = 0
    for kind, quota in KIND_QUOTAS.items():
        group = by_kind.get(kind, [])
        take = len(group) if quota is None else min(quota, len(group))
        kind_core, kind_over = draw(group, take, rng)
        core.extend(kind_core)
        oversample.extend(kind_over)
        spent += take
    remainder = by_kind.get(REMAINDER_KIND, [])
    rest_core, rest_over = draw(remainder, min(max(n - spent, 0), len(remainder)), rng)
    return core + rest_core, oversample + rest_over


def draw(
    group: Sequence[Mapping[str, object]], k: int, rng: Random
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    if k >= len(group):
        return list(group), []
    n_over = round(k * OVERSAMPLE_SHARE)
    over = sorted(group, key=lambda r: (float(str(r["confidence"])), str(r["dedup_key"])))[:n_over]
    over_keys = {str(r["dedup_key"]) for r in over}
    pool = [r for r in group if str(r["dedup_key"]) not in over_keys]
    return rng.sample(pool, k - n_over), over


def sample_audit(judged_rows: Sequence[Mapping[str, object]], *, accepts: int, rejects: int, seed: int) -> AuditSample:
    """Draws the deterministic stratified audit sample over one version's judged rows.

    The draw is seeded and pure, so the evaluator can reproduce the exact core set
    by calling it with the same inputs. Per side (accepted/rejected): every kind in
    :data:`KIND_QUOTAS` gets its quota (``None`` means exhaustive), the remainder
    budget goes to transcript messages, and within each subsampled kind 30% of the
    draw oversamples the judge's lowest-confidence verdicts.

    Args:
        judged_rows: Events joined with their judge verdicts for one prompt version.
        accepts: The audit budget for judge-accepted rows.
        rejects: The audit budget for judge-rejected rows.
        seed: The iteration's deterministic sampling seed.

    Returns:
        The sampled rows, split into the uniform core and the oversample.
    """
    rng = Random(seed)
    accepted = [row for row in judged_rows if row["is_pushback"]]
    rejected = [row for row in judged_rows if not row["is_pushback"]]
    accept_core, accept_over = stratified(accepted, accepts, rng)
    reject_core, reject_over = stratified(rejected, rejects, rng)
    return AuditSample(core=(*accept_core, *reject_core), oversample=(*accept_over, *reject_over))


async def run_verdicts(
    store: FeedbackStore,
    rows: Sequence[Mapping[str, object]],
    prompt_for: Callable[[Mapping[str, object]], str],
    *,
    role: str,
    prompt_version: int,
    tier: TModel,
    concurrency: int,
) -> tuple[int, int]:
    counts = {"judged": 0, "failed": 0}
    limiter = anyio.CapacityLimiter(concurrency)

    async def worker(row: Mapping[str, object]) -> None:
        async with limiter:
            try:
                verdict = await run_claude_structured(prompt_for(row), response_model=Verdict, tier=tier)
            except (subprocess.SubprocessError, ValidationError, json.JSONDecodeError):
                counts["failed"] += 1
                return
        await store.record_verdict(
            DedupKey(str(row["dedup_key"])),
            verdict,
            role=role,
            prompt_version=prompt_version,
            model=resolved_model(tier),
        )
        counts["judged"] += 1

    async with anyio.create_task_group() as tg:
        for row in rows:
            tg.start_soon(worker, row)
    return counts["judged"], counts["failed"]


async def triage(
    store: FeedbackStore, *, tier: TModel = "medium", limit: int | None = None, concurrency: int = 8
) -> TriageReport:
    """Judges every stored candidate lacking a verdict at the current prompt version.

    Incremental and idempotent: each row's verdict persists as soon as its call
    completes, a failed row stays unjudged and is retried on the next run, and
    re-running over a fully judged corpus is a no-op.

    Args:
        store: The open feedback store.
        tier: The judge's abstract model tier.
        limit: When set, judge at most this many rows this pass.
        concurrency: The maximum number of concurrent ``claude`` subshells.

    Returns:
        The pass's judged/failed/pending counts.
    """
    model = resolved_model(tier)
    rows = await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION, model=model, limit=limit)
    judged, failed = await run_verdicts(
        store,
        rows,
        lambda row: build_prompt(JUDGE_PROMPT, row),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        tier=tier,
        concurrency=concurrency,
    )
    pending = len(await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION, model=model))
    return TriageReport(judged=judged, failed=failed, pending=pending)


async def audit(
    store: FeedbackStore,
    *,
    accepts: int = 60,
    rejects: int = 60,
    seed: int = 1,
    tier: TModel = "large",
    concurrency: int = 8,
) -> TriageReport:
    """Audits a seeded stratified sample of the current version's judge verdicts.

    The auditor is blind by construction: its prompt is built from the event row
    alone and never sees the judge's verdict. Already-audited rows cost nothing —
    auditor verdicts key on :data:`AUDIT_VERSION`, independent of the judge's
    prompt version, so they accumulate across iterations.

    Args:
        store: The open feedback store.
        accepts: The audit budget for judge-accepted rows.
        rejects: The audit budget for judge-rejected rows.
        seed: The iteration's deterministic sampling seed.
        tier: The auditor's abstract model tier.
        concurrency: The maximum number of concurrent ``claude`` subshells.

    Returns:
        The pass's judged/failed/pending counts over the sampled rows.
    """
    sample = sample_audit(
        await store.judged(role=JUDGE, prompt_version=PROMPT_VERSION), accepts=accepts, rejects=rejects, seed=seed
    )
    audited = {str(row["dedup_key"]) for row in await store.judged(role=AUDITOR, prompt_version=AUDIT_VERSION)}
    fresh = [row for row in (*sample.core, *sample.oversample) if str(row["dedup_key"]) not in audited]
    judged, failed = await run_verdicts(
        store,
        fresh,
        lambda row: build_prompt(AUDIT_PROMPT, row),
        role=AUDITOR,
        prompt_version=AUDIT_VERSION,
        tier=tier,
        concurrency=concurrency,
    )
    return TriageReport(judged=judged, failed=failed, pending=len(fresh) - judged)
