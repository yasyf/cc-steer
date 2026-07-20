"""Stage 2 of the pipeline: the LLM judge that turns mined candidates into training pairs.

The deterministic detectors are tuned for recall; this module supplies the precision.
A judge classifies every stored candidate into a steering or noise category, and an
independently-prompted auditor (a stronger model, blind to the judge's verdicts)
samples the results so the judge's error rate is measurable. Prompts render each
candidate's :class:`~cc_transcript.context.ContextWindow` at full fidelity while the
transcript lives — a generous budget on the trigger turn, a moderate budget on the
surrounding turns — and fall back to the labeled summary previews once it expires;
each verdict records the fidelity it was judged at. Verdicts land in the ``triage``
table; accepted rows surface through the ``accepted_steering`` view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from cc_transcript.context import ContextWindow, HydratedWindow
from cc_transcript.judge import resolved_model, run_verdicts, sample_audit
from cc_transcript.mining import DedupKey
from cc_transcript.render import Budget
from pydantic import BaseModel, Field

from cc_steer.claude import cached_judge

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from cc_transcript.activity import Turn
    from cc_transcript.context import Fidelity
    from spawnllm import TModel

    from cc_steer.store import FeedbackStore

PROMPT_VERSION = 7
AUDIT_VERSION = 6
JUDGE = "judge"
AUDITOR = "auditor"
TRIGGER_BUDGET = Budget(turn_chars=2000, tool_chars=6000)
CONTEXT_BUDGET = Budget()
KIND_QUOTAS: dict[str, int | None] = {
    "interrupt_rejection": None,
    "review_comment": 10,
    "plan_review": 10,
    "question_answer": 10,
}
REMAINDER_KIND = "transcript_message"

STEERING_CATEGORIES = frozenset(
    {"wrong_approach", "incorrect_change", "unwanted_action", "style_violation", "premature", "direction"}
)

Category = Literal[
    "wrong_approach",
    "incorrect_change",
    "unwanted_action",
    "style_violation",
    "premature",
    "direction",
    "operational_directive",
    "status_update",
    "new_task",
    "question",
    "other",
]

JUDGE_SYSTEM = """\
You are classifying one message a developer sent to an AI coding assistant (Claude),
deciding whether it is genuine STEERING — the human shaping a decision the assistant
faced or raised — or non-steering noise.

Steering has two faces. The corrective face faults or redirects work the assistant
already did or proposed. The forward face resolves an open choice the assistant put on
the table — picking among options it offered, answering a question it asked, or
settling a decision it surfaced. Noise shapes no decision: routine next-step logistics
any operator would issue, status reports, bare approvals, pure information-seeking, and
fresh unrelated tasks.

Pick exactly one category:
- wrong_approach: rejects the assistant's plan, strategy, or design.
- incorrect_change: the code or content the assistant produced is wrong or broken.
- unwanted_action: the assistant did something the user did not ask for or want.
- style_violation: the work violates the user's conventions or stated preferences.
- premature: the assistant stopped early, skipped work, or claimed completion when work remains.
- direction: a forward instruction or answer that RESOLVES a decision the assistant
  faced or raised — choosing among options it offered, answering a question it asked
  (including declining an option: "nope just push the resolved branch"), or a directive
  that settles a specific open choice it surfaced ("lets do 3.14 for the python pin").
  The choice must trace back to the assistant: a directive that resolves nothing the
  assistant raised is operational_directive, not direction.
- operational_directive: routine logistics ("commit and push", "set a higher timeout")
  that faults nothing and resolves no choice the assistant raised. Approving then
  adding scope ("yes — also do Y", "upgrade our alias as well") is operational; forward
  words like "fix" or "review" aimed at future work do not by themselves fault what was
  already done.
- status_update: the user reporting state ("done, its running", "I killed it already").
- new_task: a fresh request or spec, not a reaction to the preceding action and not the
  resolution of anything the assistant raised. A report that an external tool or
  pre-existing system is broken is new_task or status_update, not incorrect_change,
  unless the assistant built it.
- question: a pure request for information, settling nothing by itself. A skeptical or
  rhetorical question that presses on a choice the assistant made ("why are we
  hardcoding this?", "should this ever be optional?", "there's really no better way?")
  is steering, not a question — categorize it by what it challenges. But a question
  proposing a NEW addition ("should we add a pyright config?", "can we also bundle the
  plugin?") presses on nothing the assistant did and resolves nothing it raised — it
  stays a question or new_task.
- other: none of the above.

The first six categories are steering; the last five are noise.
A mixed message that contains ANY steering content is steering — pick the category of
the steering part. Corrective steering is often implicit in a directive:
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
forward directive. An incredulous aside riding on a directive ("unify the code paths —
why on earth are they separate", "why is this hardcoded") faults the current
arrangement the assistant produced or left standing; the directive it rides on is
corrective steering, not a fresh new_task.
An implicit fault only counts when the faulted thing exists in the assistant's prior
work: "set the right headers" while assigning fresh config work is a plain directive;
"use the right template" after the assistant used the wrong one is a correction;
"get rid of X" where X is pre-existing state the assistant did not create (clearing
local state, deleting an old branch) is a plain directive — noise unless it settles a
choice the assistant raised. And when the user corrects their own earlier instruction
("ah sorry, my bad — we treat it as native"), that is a spec revision originating with
the user — the decision it changes was never the assistant's — so it is a new
instruction, not steering.
Forward direction only counts when the choice was genuinely open. Picking an option,
declining one, or supplying the missing parameter of a decision the assistant surfaced
is direction; a bare approval or acknowledgment that merely green-lights the course the
assistant already proposed ("lgtm", "sounds good", "yes go ahead") chooses nothing —
noise. A leading "no" or "nope" is therefore rarely noise: when it answers a yes/no
question the assistant asked or declines an option it offered, and the rest is a plain
forward directive with no fault of completed work ("nope just push the resolved
branch", "nope this is good, commit and push"), it resolves the assistant's open
question — direction. When the "no" countermands or rejects work the assistant actually
did or proposed ("no we dont want to vendor it", "no, that's the wrong file"), it is
corrective steering — categorize it by what it rejects.
The assistant action being steered may predate the context shown: when the message
critiques files or output the assistant produced earlier in the session, it is steering
on that work even if the immediately preceding action is unrelated.
When the source is review_comment, the message is an inline code-review comment on code
the assistant wrote: terse imperatives there ("inline", "remove this one", "maybe make
this _safe?") are corrections — usually style_violation, incorrect_change, or
wrong_approach — not operational directives. A bare prohibition or convention naming a
rule for that line ("no comments", "no globals", "it is required", "always use X") is a
style_violation correction, not other or noise — even with no verb.
When the source is question_answer, the message is the user's answer to a question the
assistant posed through its AskUserQuestion tool; the question and the option the
answer resolves to are rendered below. The assistant explicitly handed this decision to
the user, so the answer is steering by definition — it is never noise. A plain pick or
answer is direction (an ordinal or shorthand like "3, ..." resolves to that option's
label); reach for wrong_approach, unwanted_action, or style_violation when the answer
rejects the offered options outright, redirects the assistant, or specifies a different
approach than anything presented.

what_claude_did: ONE neutral sentence naming the assistant action or question the
message responds to (e.g. "Force-pushed to the shared branch with git push --force").
Write it even when the message is noise.
confidence: your probability (0 to 1) that your steering-vs-noise call is correct.
rationale: one short clause."""

JUDGE_USER = """\
[source: {source_kind}]
{context}
{question_answer}
=== USER MESSAGE TO CLASSIFY ===
{text}"""

AUDIT_SYSTEM = """\
A dataset is being built of developer steering: moments where a human, watching an AI
coding assistant work, shaped its course — told it that something about its work was
wrong, unwanted, or off-track, or settled a decision the assistant had put in the
human's hands. You are the quality gate: given one human message and its surrounding
conversation, decide independently whether the message belongs in that dataset.

It belongs (it is steering) in either of two cases. First, when the message faults the
assistant's preceding work or behavior in any way — its direction, its output, its side
effects, its style, or its stopping point — even partially, even alongside unrelated
content. Second, when the message resolves a decision the assistant faced or raised:
picks among options it offered, answers a question it asked, or settles a specific open
choice it surfaced.

It does not belong when the message shapes no decision: routine logistics any operator
would issue that settle nothing the assistant raised, a fresh assignment, a status
report, a pure request for information, or a bare approval that merely green-lights the
course the assistant already proposed.

The boundary runs through what the message targets, not how it is phrased. Weigh each
of these before deciding:

- Removal orders. "get rid of the MAX_SEQ_LEN and the trt shapes" right after the
  assistant added them faults that work — steering. "get rid of the old build
  artifacts", targeting pre-existing state the assistant never created, assigns
  forward work.
- Buried faults. "switch the client over — we arent supposed to be using litellm
  anymore" leads with a directive but faults the work for still using something it
  should not — steering. An incredulous aside riding on a directive ("why on earth
  are these separate", "why is X still here") faults the current state the same way.
  "switch the client over to the new SDK" alone assigns work and settles nothing.
- Leading "no". "no we dont want to vendor it" countermands the assistant's proposal —
  corrective steering. "nope just push the resolved branch", answering a question the
  assistant asked and rolling into a directive, settles that open question — forward
  steering. Both belong; contrast "commit and push" issued unprompted, which answers
  nothing and settles nothing the assistant raised.
- Handed-over decisions. When the source is question_answer, the human is answering a
  question the assistant posed through its AskUserQuestion tool (the question and the
  option the answer resolves to are rendered below): the assistant deferred the
  decision, so the answer always belongs — as forward steering when it plainly picks
  or answers, as a correction when it rejects or redirects what was offered.
- Questions. "couldn't this inherit from userlist?" presses on a choice the assistant
  made — steering. "what are the tradeoffs of each option here?" seeks information to
  settle something still open, and a question proposing a new addition presses on
  nothing the assistant did — neither steers.
- Inline review comments (source review_comment) annotate a specific line the
  assistant authored: short imperatives, suggestions, and bare prohibitions or
  conventions there ("inline", "it is required", "no comments", "use dataclasses
  always") fault that line — steering, even with no verb.
- Accepted doubts and bare approvals. A message that raises a doubt but ends by
  deferring to the assistant's choice ("nevermind, if there's a reason for it, then do
  it"), or that simply approves the proposed course ("lgtm", "yes go ahead"), chooses
  nothing — not steering.

Choose the single best-fitting label:
- wrong_approach / incorrect_change / unwanted_action / style_violation / premature /
  direction (steering; direction marks the forward face — the message steers by
  resolving a choice the assistant raised rather than by faulting work)
- operational_directive / status_update / new_task / question / other (not steering)

Also provide: what_claude_did — one neutral sentence describing the assistant's
preceding action or question; confidence — your probability (0 to 1) that your
steering-vs-not call is right; rationale — one short clause."""

AUDIT_USER = """\
[source: {source_kind}]
{context}
{question_answer}
=== HUMAN MESSAGE TO ASSESS ===
{text}"""


class Verdict(BaseModel):
    """One triage verdict on a stored feedback candidate.

    Attributes:
        category: The single best-fitting steering or noise category.
        what_claude_did: One neutral sentence naming the assistant action the
            message responds to.
        confidence: The model's probability that its steering-vs-noise call is right.
        rationale: One short clause explaining the call.
    """

    category: Category
    what_claude_did: str
    confidence: float = Field(ge=0, le=1)
    rationale: str

    @property
    def is_steering(self) -> bool:
        """Whether the category marks genuine steering."""
        return self.category in STEERING_CATEGORIES

    @property
    def accepted(self) -> bool:
        """Alias satisfying the judge package's ``VerdictLike`` protocol."""
        return self.is_steering

    @property
    def summary(self) -> str:
        """Alias satisfying the judge package's ``VerdictLike`` protocol."""
        return self.what_claude_did

    @property
    def canonical_key(self) -> str | None:
        """Satisfies the judge package's ``VerdictLike`` protocol; steering triage names no durable rule."""
        return None


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
    if (hydrated := window.hydrate()) is None:
        return replace(window, fidelity="summary").render_preview(budget=CONTEXT_BUDGET), "summary"
    split = len(window.before)
    end = split + (window.trigger is not None)
    turns = hydrated.turns
    if window.trigger is not None and split and turns[split - 1] is turns[split]:
        # A tool-result anchor splits one real turn across before's tail and the
        # trigger; both hydrate to it, so drop the duplicate and render it once.
        turns = turns[: split - 1] + turns[split:]
        split -= 1
        end -= 1
    return (
        "\n".join(
            (
                section(window, "conversation before", turns[:split], CONTEXT_BUDGET),
                section(window, "the turn the message arrived in", turns[split:end], TRIGGER_BUDGET),
                section(window, "conversation after", turns[end:], CONTEXT_BUDGET),
            )
        ),
        "full",
    )


def question_answer_block(row: Mapping[str, object]) -> str:
    if row["source_kind"] != "question_answer":
        return ""
    payload = json.loads(str(row["payload_json"]))
    if not (picked := payload.get("picked_labels")):
        resolved = "The user selected none of the offered options and wrote their own answer."
    else:
        marked = (
            "the option the assistant marked (Recommended)"
            if payload.get("recommended_pick")
            else "an option (not the recommended one)"
        )
        resolved = f"The answer resolves to {marked}: " + "; ".join(picked)
    return f"=== QUESTION THE ASSISTANT ASKED ===\n{payload['question']}\n{resolved}\n"


async def build_prompt(template: str, row: Mapping[str, object]) -> tuple[str, Fidelity]:
    context, fidelity = await render_context(ContextWindow.from_json(str(row["context_json"])))
    prompt = template.format(
        source_kind=row["source_kind"], context=context, text=row["text"], question_answer=question_answer_block(row)
    )
    return prompt, fidelity


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
    rows = await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION, limit=limit, refresh_summary=refresh_summary)
    fidelities: dict[str, Fidelity] = {}
    judged, failed = await run_verdicts(
        rows,
        prompt_builder(JUDGE_USER, fidelities),
        cached_judge(Verdict, tier=tier, system=JUDGE_SYSTEM),
        persist_verdict(store, role=JUDGE, prompt_version=PROMPT_VERSION, model=model, fidelities=fidelities),
        concurrency=concurrency,
    )
    pending = len(await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION))
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
        prompt_builder(AUDIT_USER, fidelities),
        cached_judge(Verdict, tier=tier, system=AUDIT_SYSTEM),
        persist_verdict(
            store, role=AUDITOR, prompt_version=AUDIT_VERSION, model=resolved_model(tier), fidelities=fidelities
        ),
        concurrency=concurrency,
    )
    return TriageReport(judged=judged, failed=failed, pending=len(fresh) - judged)
