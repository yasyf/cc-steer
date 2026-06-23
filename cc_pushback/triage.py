"""Stage 2 of the pipeline: the LLM judge that turns mined candidates into training pairs.

The deterministic detectors are tuned for recall; this module supplies the precision.
A judge classifies every stored candidate into a pushback or noise category, and an
independently-prompted auditor (a stronger model, blind to the judge's verdicts)
samples the results so the judge's error rate is measurable. Prompts render each
candidate's :class:`~cc_transcript.context.ContextWindow` at full fidelity while the
transcript lives — a generous budget on the trigger turn, a moderate budget on the
surrounding turns — and fall back to the labeled summary previews once it expires;
each verdict records the fidelity it was judged at. Verdicts land in the ``triage``
table; accepted rows surface through the ``accepted_pushback`` view.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from cc_transcript.context import ContextWindow, HydratedWindow
from cc_transcript.judge import resolved_model, run_verdicts, sample_audit, structured_judge
from cc_transcript.mining import DedupKey
from cc_transcript.render import Budget
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from cc_transcript.activity import Turn
    from cc_transcript.context import Fidelity
    from spawnllm import TModel

    from cc_pushback.store import FeedbackStore

PROMPT_VERSION = 5
AUDIT_VERSION = 3
JUDGE = "judge"
AUDITOR = "auditor"
TRIGGER_BUDGET = Budget(turn_chars=2000, tool_chars=6000)
CONTEXT_BUDGET = Budget()
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
  categorize it by what it challenges. But a question proposing a NEW addition
  ("should we add a pyright config?", "can we also bundle the plugin?") presses on
  nothing the assistant did — it stays a question or new_task.
- other: none of the above.

The first five categories are pushback; the rest are noise.
A mixed message that contains ANY genuine corrective content is pushback — pick the
category of the corrective part. Corrective content is often implicit in a directive:
"figure out the right proper fix" faults the current fix, "look more closely" faults a
shallow look, "not just X — give me Y" faults an insufficient answer, praise followed
by a redirect ("its good that you did X, but the more important thing is Y") faults the
scope, and exasperation ("stop wasting time", "fucking hell just commit") faults the
behavior, even when phrased as the next step. Ordering the removal or reversal of
something the assistant just produced ("get rid of the MAX_SEQ_LEN and the trt shapes",
"remove that", "take it back out", "we dont want the shim") faults that work as
unwanted, even with no other criticism. And a clause noting the work still contains or
uses something it should not ("we arent supposed to be using litellm anymore", "why is
X still here", "thats not how we do it") is corrective even when the lead clause is a
forward directive.
An implicit fault only counts when the faulted thing exists in the assistant's prior
work: "set the right headers" while assigning fresh config work is a plain directive;
"use the right template" after the assistant used the wrong one is a correction;
"get rid of X" where X is pre-existing state the assistant did not create (clearing
local state, deleting an old branch) is a plain directive, not pushback. And when the
user corrects their own earlier instruction ("ah sorry, my bad — we treat it as
native"), that is a spec clarification, not pushback on the assistant.
A leading "no" or "nope" is not automatically rejection. When it answers a yes/no
question the assistant asked or declines an option the assistant offered, and the rest
is a plain forward directive with no fault of completed work ("nope just push the
resolved branch", "nope this is good, commit and push"), it is an answer — noise. It is
pushback only when the "no" countermands or rejects work the assistant actually did or
proposed ("no we dont want to vendor it", "no, that's the wrong file").
The assistant action being corrected may predate the context shown: when the message
critiques files or output the assistant produced earlier in the session, it is pushback
on that work even if the immediately preceding action is unrelated.
When the source is review_comment, the message is an inline code-review comment on code
the assistant wrote: terse imperatives there ("inline", "remove this one", "maybe make
this _safe?") are corrections — usually style_violation, incorrect_change, or
wrong_approach — not operational directives. A bare prohibition or convention naming a
rule for that line ("no comments", "no globals", "it is required", "always use X") is a
style_violation correction, not other or noise — even with no verb.

what_claude_did: ONE neutral sentence naming the assistant action the message responds
to (e.g. "Force-pushed to the shared branch with git push --force"). Write it even when
the message is noise.
confidence: your probability (0 to 1) that your pushback-vs-noise call is correct.
rationale: one short clause.

[source: {source_kind}]
{context}
=== USER MESSAGE TO CLASSIFY ===
{text}"""

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
{context}
=== HUMAN MESSAGE TO ASSESS ===
{text}"""


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

    @property
    def accepted(self) -> bool:
        """Alias satisfying the judge package's ``VerdictLike`` protocol."""
        return self.is_pushback

    @property
    def summary(self) -> str:
        """Alias satisfying the judge package's ``VerdictLike`` protocol."""
        return self.what_claude_did


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


def section(window: ContextWindow, label: str, turns: tuple[Turn, ...], budget: Budget) -> str:
    return f"=== {label} ===\n" + (HydratedWindow(window=window, turns=turns).render(budget=budget) or "(none)")


async def render_context(window: ContextWindow) -> tuple[str, Fidelity]:
    """Renders a candidate's window for a prompt, at the best fidelity available.

    While the transcript lives, the window hydrates and renders at full fidelity —
    the trigger turn under the generous :data:`TRIGGER_BUDGET`, the surrounding
    turns under the moderate :data:`CONTEXT_BUDGET`. Once it expires (or any ref
    was compacted away), the persisted previews render instead, led by the
    built-in summary-fidelity label.

    Returns:
        The rendered context and the fidelity it was rendered at.
    """
    if (hydrated := await window.hydrate()) is None:
        return replace(window, fidelity="summary").render_preview(budget=CONTEXT_BUDGET), "summary"
    split = len(window.before)
    end = split + (window.trigger is not None)
    return (
        "\n".join(
            (
                section(window, "conversation before", hydrated.turns[:split], CONTEXT_BUDGET),
                section(window, "the turn the message arrived in", hydrated.turns[split:end], TRIGGER_BUDGET),
                section(window, "conversation after", hydrated.turns[end:], CONTEXT_BUDGET),
            )
        ),
        "full",
    )


async def build_prompt(template: str, row: Mapping[str, object]) -> tuple[str, Fidelity]:
    context, fidelity = await render_context(ContextWindow.from_json(str(row["context_json"])))
    return template.format(source_kind=row["source_kind"], context=context, text=row["text"]), fidelity


def prompt_builder(template: str, fidelities: dict[str, Fidelity]) -> Callable[[Mapping[str, object]], Awaitable[str]]:
    async def build(row: Mapping[str, object]) -> str:
        prompt, fidelity = await build_prompt(template, row)
        fidelities[str(row["dedup_key"])] = fidelity
        return prompt

    return build


def persist_verdict(
    store: FeedbackStore, *, role: str, prompt_version: int, model: str, fidelities: Mapping[str, Fidelity]
) -> Callable[[Mapping[str, object], Verdict], Awaitable[None]]:
    async def persist(row: Mapping[str, object], verdict: Verdict) -> None:
        await store.record_verdict(
            DedupKey(str(row["dedup_key"])),
            verdict,
            role=role,
            prompt_version=prompt_version,
            model=model,
            fidelity=fidelities[str(row["dedup_key"])],
        )

    return persist


async def triage(
    store: FeedbackStore,
    *,
    tier: TModel = "medium",
    limit: int | None = None,
    concurrency: int = 8,
    refresh_summary: bool = False,
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
        refresh_summary: When True, also re-judge rows whose verdict was recorded
            at summary fidelity; a full-fidelity verdict replaces the summary one
            once the row's window hydrates again.

    Returns:
        The pass's judged/failed/pending counts.
    """
    model = resolved_model(tier)
    rows = await store.unjudged(
        role=JUDGE, prompt_version=PROMPT_VERSION, model=model, limit=limit, refresh_summary=refresh_summary
    )
    fidelities: dict[str, Fidelity] = {}
    judged, failed = await run_verdicts(
        rows,
        prompt_builder(JUDGE_PROMPT, fidelities),
        structured_judge(Verdict, tier=tier),
        persist_verdict(store, role=JUDGE, prompt_version=PROMPT_VERSION, model=model, fidelities=fidelities),
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
        await store.judged(role=JUDGE, prompt_version=PROMPT_VERSION),
        accepts=accepts,
        rejects=rejects,
        seed=seed,
        quotas=KIND_QUOTAS,
        remainder_kind=REMAINDER_KIND,
    )
    audited = {str(row["dedup_key"]) for row in await store.judged(role=AUDITOR, prompt_version=AUDIT_VERSION)}
    fresh = [row for row in (*sample.core, *sample.oversample) if str(row["dedup_key"]) not in audited]
    fidelities: dict[str, Fidelity] = {}
    judged, failed = await run_verdicts(
        fresh,
        prompt_builder(AUDIT_PROMPT, fidelities),
        structured_judge(Verdict, tier=tier),
        persist_verdict(
            store, role=AUDITOR, prompt_version=AUDIT_VERSION, model=resolved_model(tier), fidelities=fidelities
        ),
        concurrency=concurrency,
    )
    return TriageReport(judged=judged, failed=failed, pending=len(fresh) - judged)
