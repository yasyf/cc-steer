"""Decode the feedback corpus and render it for the dashboard.

Holds the data model the dashboard serves — :class:`Sample`, :class:`VerdictRow`,
:class:`RefinedPairRow`, and the :class:`Lineage` that stitches a candidate's whole
pipeline trail together — plus the corpus :class:`Summary` (written by the ``claude``
CLI when available, falling back to heuristics) and the HTML renderers for a
candidate's five-stage lineage. The FastAPI surface lives in :mod:`cc_pushback.dashboard`.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript.domains.mining import NOISE_FLOOR, effective_confidence
from cc_transcript.domains.mining.confidence import from_payload

from cc_pushback.claude import claude_available, run_claude
from cc_pushback.context import ContextSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal

    from cc_transcript.domains.mining import CandidateSignal

    from cc_pushback.context import ContextTurn
    from cc_pushback.evaluate import GoldenRow

CONTEXT_TURN_LIMIT = 700
SAMPLE_TEXT_LIMIT = 400
HIGHLIGHT_POOL_PER_KIND = 8
HEURISTIC_HIGHLIGHTS = 12

SUMMARY_SYSTEM = """\
You analyze a developer's "pushback" — the corrective feedback they give an AI coding assistant.
You receive corpus statistics and a numbered pool of real feedback samples.
Return ONLY a JSON object, with no prose around it, of exactly this shape:
{"narrative": "<2-4 sentences on the developer's pushback style and recurring themes>",
 "highlights": [{"id": <sample id>, "why": "<one short clause on why it is representative>"}]}
Pick 8-12 highlights, only from the provided sample ids, favoring variety across feedback kinds.
"""

CSS = """
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
h1,h2{font-weight:600}
header.top{padding:24px;border-bottom:1px solid var(--border)}
header.top .sub{color:var(--muted)}
section{padding:16px 24px}
.stat-cards{display:flex;gap:12px;flex-wrap:wrap}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px}
.stat .n{font-size:20px;font-weight:600}
.stat .l{color:var(--muted);font-size:12px}
table.dist{border-collapse:collapse;margin-top:14px}
table.dist td{padding:2px 10px 2px 0;white-space:nowrap}
.bar{display:inline-block;height:10px;background:var(--accent);border-radius:3px;vertical-align:middle}
.months{display:flex;gap:3px;align-items:flex-end;margin-top:14px}
.mcol{display:flex;flex-direction:column;align-items:center;justify-content:flex-end}
.mcol .m{width:22px;background:var(--accent);border-radius:3px 3px 0 0}
.mcol span{font-size:9px;color:var(--muted);margin-top:3px}
.narrative{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 18px;max-width:80ch;margin-top:14px}
#controls{position:sticky;top:0;background:var(--bg);display:flex;gap:8px;align-items:center;
flex-wrap:wrap;border-bottom:1px solid var(--border);z-index:2}
.kind-btn{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:14px;
padding:4px 12px;cursor:pointer;font:inherit}
.kind-btn.active{background:var(--accent);color:#0d1117;border-color:var(--accent)}
#search{flex:1;min-width:200px;background:var(--panel);color:var(--fg);border:1px solid var(--border);
border-radius:6px;padding:6px 10px;font:inherit}
#count{color:var(--muted)}
label.noise{color:var(--muted);display:flex;gap:4px;align-items:center;cursor:pointer}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:12px 0}
.card header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#21262d;border:1px solid var(--border)}
.badge-transcript_message{color:#8b949e}.badge-review_comment{color:#7ee787}.badge-plan_review{color:#d2a8ff}
.badge-interrupt_rejection{color:#ff7b72}.badge-superset_issue{color:#ffa657}
time{color:var(--muted);font-size:12px}
.chip{font-size:11px;color:var(--muted);background:#21262d;border-radius:6px;padding:1px 6px}
.text pre{white-space:pre-wrap;word-break:break-word;margin:0;font:inherit}
details.ctx{margin-top:10px}
details.ctx summary{color:var(--accent);cursor:pointer}
.turn{border-left:2px solid var(--border);padding:4px 0 4px 10px;margin:6px 0}
.turn .role{font-size:10px;text-transform:uppercase;color:var(--muted)}
.turn .tools{font-size:10px;color:var(--accent);margin-left:6px}
.turn pre{white-space:pre-wrap;word-break:break-word;margin:2px 0 0;font:inherit;color:var(--muted)}
.turn-user pre{color:var(--fg)}
.turn-trigger{border-left-color:var(--accent)}
.turn-trigger .role::after{content:" \\2190 pushed back on";color:var(--accent)}
.why{color:var(--accent);font-style:italic;margin:0 0 6px}
.highlight{margin:12px 0}
.lineage{display:flex;flex-direction:column;gap:14px}
.stage{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px}
.stage h3{margin:0 0 10px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stage-detector{border-left:3px solid var(--muted)}
.stage-judge{border-left:3px solid var(--accent)}
.stage-auditor{border-left:3px solid #d2a8ff}
.stage-refiner{border-left:3px solid #7ee787}
.stage-golden{border-left:3px solid #ffa657}
.verdict,.pair{border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin:8px 0}
.vhead{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.vsum,.vrat,.paction,.pcomplaint{white-space:pre-wrap;word-break:break-word;margin:2px 0;font:inherit}
.vrat,.pcomplaint{color:var(--muted)}
.pverbatim{border-left:2px solid #7ee787;margin:6px 0;padding:2px 0 2px 10px;color:var(--fg)}
.orig pre{white-space:pre-wrap;word-break:break-word;margin:0 0 8px;color:var(--muted)}
mark{background:#7ee78733;color:var(--fg);border-radius:3px}
.flip{font-size:11px;color:#ffa657}
.agree{font-size:11px;color:#7ee787}
.disagree{font-size:11px;color:#ff7b72}
.muted{color:var(--muted)}
.badge.pass{color:#7ee787}.badge.fail{color:#ff7b72}
.cat-wrong_approach{color:#ff7b72}.cat-incorrect_change{color:#ffa657}.cat-unwanted_action{color:#f0883e}
.cat-style_violation{color:#d2a8ff}.cat-premature{color:#79c0ff}
.cat-operational_directive,.cat-status_update,.cat-new_task,.cat-question,.cat-other{color:#8b949e}
"""


@dataclass(frozen=True, slots=True)
class Sample:
    """One stored feedback event, decoded from a :meth:`FeedbackStore.events` row.

    Attributes:
        id: The event's database id.
        source_kind: Which detector produced it.
        occurred_at: The ISO timestamp of the feedback.
        text: The verbatim pushback text.
        payload: The detector-specific metadata, decoded from ``payload_json``.
        context: The conversational window around the feedback.
        origin_path: The transcript file the event came from.
        session_id: The session the event came from.
        signal: The de-noising confidence signal, decoded from the payload.
    """

    id: int
    source_kind: str
    occurred_at: str
    text: str
    payload: Mapping[str, Any]
    context: ContextSnapshot
    origin_path: str | None
    session_id: str | None
    signal: CandidateSignal | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Sample:
        """Decodes a :meth:`FeedbackStore.events` row into a :class:`Sample`."""
        payload = json.loads(str(row["payload_json"])) if row["payload_json"] else {}
        return cls(
            id=int(str(row["id"])),
            source_kind=str(row["source_kind"]),
            occurred_at=str(row["occurred_at"]),
            text=str(row["text"]),
            payload=payload,
            context=ContextSnapshot.from_json(str(row["context_json"])),
            origin_path=str(row["origin_path"]) if row["origin_path"] else None,
            session_id=str(row["session_id"]) if row["session_id"] else None,
            signal=from_payload(payload.get("signal")),
        )


@dataclass(frozen=True, slots=True)
class VerdictRow:
    """One triage verdict — a judge or auditor call on a candidate.

    Attributes:
        role: Who produced it, ``judge`` or ``auditor``.
        prompt_version: The prompt version that produced it.
        model: The resolved model name that produced it.
        category: The chosen category.
        is_pushback: Whether the category counts as pushback.
        what_claude_did: The one-line normalization of the action under review.
        confidence: The verdict's confidence, ``0``–``1``.
        rationale: The short justification for the call.
        judged_at: The ISO timestamp the verdict was recorded.
    """

    role: str
    prompt_version: int
    model: str
    category: str
    is_pushback: bool
    what_claude_did: str
    confidence: float
    rationale: str
    judged_at: str

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> VerdictRow:
        """Decodes a :meth:`FeedbackStore.lineage` verdict row into a :class:`VerdictRow`."""
        return cls(
            role=str(row["role"]),
            prompt_version=int(str(row["prompt_version"])),
            model=str(row["model"]),
            category=str(row["category"]),
            is_pushback=bool(row["is_pushback"]),
            what_claude_did=str(row["what_claude_did"]),
            confidence=float(str(row["confidence"])),
            rationale=str(row["rationale"]),
            judged_at=str(row["judged_at"]),
        )


@dataclass(frozen=True, slots=True)
class RefinedPairRow:
    """One atomic training pair distilled by the refiner from a single complaint.

    Attributes:
        pair_index: The pair's position within the message's split.
        action: The faithful re-synthesis of what the assistant did.
        complaint_verbatim: The exact span of the user's message voicing the complaint.
        complaint: The one-sentence distillation of the objection.
        prompt_version: The refine prompt version that produced it.
        model: The resolved model name that produced it.
    """

    pair_index: int
    action: str
    complaint_verbatim: str
    complaint: str
    prompt_version: int
    model: str

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> RefinedPairRow:
        """Decodes a :meth:`FeedbackStore.lineage` pair row into a :class:`RefinedPairRow`."""
        return cls(
            pair_index=int(str(row["pair_index"])),
            action=str(row["action"]),
            complaint_verbatim=str(row["complaint_verbatim"]),
            complaint=str(row["complaint"]),
            prompt_version=int(str(row["prompt_version"])),
            model=str(row["model"]),
        )


@dataclass(frozen=True, slots=True)
class Lineage:
    """The full pipeline trail for one candidate, from detector hit to refined pairs.

    Attributes:
        sample: The raw mined event — detector evidence and conversational context.
        dedup_key: The content-derived key joining every stage.
        verdicts: Every judge and auditor verdict recorded against the key.
        pairs: The latest refinement generation's atomic pairs, by ``pair_index``.
    """

    sample: Sample
    dedup_key: str
    verdicts: tuple[VerdictRow, ...]
    pairs: tuple[RefinedPairRow, ...]

    @classmethod
    def from_lineage(cls, data: Mapping[str, Any]) -> Lineage:
        """Builds a :class:`Lineage` from a :meth:`FeedbackStore.lineage` result."""
        return cls(
            sample=Sample.from_row(data),
            dedup_key=str(data["dedup_key"]),
            verdicts=tuple(VerdictRow.from_row(row) for row in data["verdicts"]),
            pairs=tuple(RefinedPairRow.from_row(row) for row in data["pairs"]),
        )

    @property
    def judge_verdicts(self) -> tuple[VerdictRow, ...]:
        return tuple(sorted((v for v in self.verdicts if v.role == "judge"), key=lambda v: v.prompt_version))

    @property
    def auditor_verdict(self) -> VerdictRow | None:
        return max((v for v in self.verdicts if v.role == "auditor"), key=lambda v: v.prompt_version, default=None)

    @property
    def final(self) -> VerdictRow | None:
        return self.judge_verdicts[-1] if self.judge_verdicts else None

    @property
    def flipped(self) -> bool:
        return len({v.is_pushback for v in self.judge_verdicts}) > 1

    @property
    def agreement(self) -> Literal["agree", "disagree"] | None:
        match (self.final, self.auditor_verdict):
            case (None, _) | (_, None):
                return None
            case (judge, auditor):
                return "agree" if judge.is_pushback == auditor.is_pushback else "disagree"

    @property
    def status(self) -> Literal["refined", "accepted", "noise", "unjudged"]:
        if self.pairs:
            return "refined"
        match self.final:
            case None:
                return "unjudged"
            case verdict if verdict.is_pushback:
                return "accepted"
            case _:
                return "noise"


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregate counts describing the whole corpus.

    Attributes:
        total: The total number of samples.
        by_kind: Sample counts keyed by source kind, most common first.
        noise: The number of low-signal samples (bare interrupt markers, hook
            errors, and near-empty messages).
        sessions: The number of distinct sessions.
        projects: The number of distinct originating projects.
        first: The earliest sample date (``YYYY-MM-DD``).
        last: The latest sample date (``YYYY-MM-DD``).
        by_month: Sample counts keyed by ``YYYY-MM``, in chronological order.
    """

    total: int
    by_kind: Mapping[str, int]
    noise: int
    sessions: int
    projects: int
    first: str
    last: str
    by_month: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Stage-by-stage counts describing how the corpus flows through the pipeline.

    Attributes:
        accepted: Judge-accepted events (the refiner's input pool).
        refined: Accepted events split into atomic pairs.
        pending: Accepted events not yet refined.
        noise_judged: Events the judge labeled non-pushback.
        unjudged: Events without a judge verdict.
        total_pairs: Atomic ``{action, complaint}`` pairs across the corpus.
        pairs_per_event: Mean pairs per refined event.
        by_category: Accepted-event counts keyed by judge category.
        audited: Accepted-or-rejected events carrying an auditor verdict.
        agree: Audited events where the auditor matched the judge's side.
        disagree: Audited events where the auditor differed.
        flips: Events whose judge side changed across prompt versions.
        golden_total: Events present in the golden regression fixture.
        golden_pass: Golden events whose latest judge matches the frozen label.
    """

    accepted: int
    refined: int
    pending: int
    noise_judged: int
    unjudged: int
    total_pairs: int
    pairs_per_event: float
    by_category: Mapping[str, int]
    audited: int
    agree: int
    disagree: int
    flips: int
    golden_total: int
    golden_pass: int


@dataclass(frozen=True, slots=True)
class Highlight:
    """A standout sample chosen for the summary, with an optional rationale.

    Attributes:
        event_id: The id of the highlighted sample.
        why: A short clause on why it is representative, when one was written.
    """

    event_id: int
    why: str | None = None


@dataclass(frozen=True, slots=True)
class Summary:
    """The corpus overview rendered above the sample list.

    Attributes:
        stats: The aggregate corpus counts.
        highlights: The standout samples chosen for the summary.
        narrative: A prose description of the developer's pushback style, when the
            ``claude`` CLI produced one.
    """

    stats: CorpusStats
    highlights: tuple[Highlight, ...]
    narrative: str | None


def is_noise(sample: Sample) -> bool:
    return effective_confidence(sample.signal) < NOISE_FLOOR


def project_label(origin_path: str) -> str:
    name = Path(origin_path).parent.name
    return next(
        (name.rsplit(marker, 1)[-1] for marker in ("-Code-", "-projects-", "-worktrees-") if marker in name),
        name.lstrip("-"),
    )


def corpus_stats(samples: Sequence[Sample]) -> CorpusStats:
    times = sorted(s.occurred_at for s in samples)
    return CorpusStats(
        total=len(samples),
        by_kind=dict(Counter(s.source_kind for s in samples).most_common()),
        noise=sum(is_noise(s) for s in samples),
        sessions=len({s.session_id for s in samples if s.session_id}),
        projects=len({Path(s.origin_path).parent.name for s in samples if s.origin_path}),
        first=times[0][:10] if times else "",
        last=times[-1][:10] if times else "",
        by_month=dict(sorted(Counter(s.occurred_at[:7] for s in samples).items())),
    )


def candidate_status(row: Mapping[str, object]) -> Literal["refined", "accepted", "noise", "unjudged"]:
    """Classifies one :meth:`FeedbackStore.candidates` row by how far it reached."""
    match row["is_pushback"]:
        case None:
            return "unjudged"
        case 0:
            return "noise"
        case _:
            return "refined" if row["pair_count"] else "accepted"


def golden_label(is_pushback: object) -> str:
    return "pushback" if is_pushback else "noise"


def golden_status(
    dedup_key: str, final: VerdictRow | None, golden_map: Mapping[str, GoldenRow]
) -> Literal["pass", "fail"] | None:
    """Returns whether the latest judge matches the frozen golden label, or ``None`` off-fixture."""
    if (gold := golden_map.get(dedup_key)) is None or final is None:
        return None
    return "pass" if final.is_pushback == gold.expected else "fail"


def pipeline_stats(candidates: Sequence[Mapping[str, object]], *, golden_map: Mapping[str, GoldenRow]) -> PipelineStats:
    statuses = Counter(candidate_status(row) for row in candidates)
    refined_rows = [row for row in candidates if candidate_status(row) == "refined"]
    total_pairs = sum(int(str(row["pair_count"])) for row in refined_rows)
    audited = [row for row in candidates if row["auditor_is_pushback"] is not None and row["is_pushback"] is not None]
    agree = sum(bool(row["auditor_is_pushback"]) == bool(row["is_pushback"]) for row in audited)
    golden = [(row, golden_map[key]) for row in candidates if (key := str(row["dedup_key"])) in golden_map]
    return PipelineStats(
        accepted=statuses["accepted"] + statuses["refined"],
        refined=statuses["refined"],
        pending=statuses["accepted"],
        noise_judged=statuses["noise"],
        unjudged=statuses["unjudged"],
        total_pairs=total_pairs,
        pairs_per_event=total_pairs / len(refined_rows) if refined_rows else 0.0,
        by_category=dict(
            Counter(str(row["category"]) for row in candidates if row["is_pushback"]).most_common()
        ),
        audited=len(audited),
        agree=agree,
        disagree=len(audited) - agree,
        flips=sum(bool(row["flipped"]) for row in candidates),
        golden_total=len(golden),
        golden_pass=sum(bool(row["is_pushback"]) == gold.expected for row, gold in golden),
    )


def candidate_pool(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    pool: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if not is_noise(sample):
            pool[sample.source_kind].append(sample)
    return {
        kind: sorted(items, key=lambda s: len(s.text), reverse=True)[:HIGHLIGHT_POOL_PER_KIND]
        for kind, items in pool.items()
    }


def heuristic_highlight_ids(pool: Mapping[str, Sequence[Sample]]) -> list[int]:
    rows = [s for group in zip_longest(*pool.values()) for s in group if s is not None]
    return [s.id for s in rows[:HEURISTIC_HIGHLIGHTS]]


def summary_prompt(pool: Mapping[str, Sequence[Sample]], stats: CorpusStats) -> str:
    return "\n".join(
        [
            f"Corpus: {stats.total} samples across {stats.sessions} sessions, {stats.first} to {stats.last}.",
            "By kind: " + ", ".join(f"{kind}={n}" for kind, n in stats.by_kind.items()),
            "",
            "Feedback samples (id, kind, text):",
            *(
                f"[{s.id}] ({kind}) {' '.join(s.text.split())[:SAMPLE_TEXT_LIMIT]}"
                for kind, group in pool.items()
                for s in group
            ),
        ]
    )


def parse_summary_json(raw: str) -> tuple[str, list[dict[str, Any]]] | None:
    if not (match := re.search(r"\{.*\}", raw, re.DOTALL)):
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    narrative, picks = data.get("narrative"), data.get("highlights")
    if not isinstance(narrative, str) or not isinstance(picks, list):
        return None
    return narrative, [p for p in picks if isinstance(p, dict) and isinstance(p.get("id"), int)]


async def llm_summary(
    pool: Mapping[str, Sequence[Sample]], stats: CorpusStats, model: str
) -> tuple[str, tuple[Highlight, ...]] | None:
    try:
        raw = await run_claude(summary_prompt(pool, stats), system=SUMMARY_SYSTEM, model=model)
    except subprocess.SubprocessError:
        return None
    if (parsed := parse_summary_json(raw)) is None:
        return None
    narrative, picks = parsed
    valid = {s.id for group in pool.values() for s in group}
    highlights = tuple(Highlight(pick["id"], pick.get("why")) for pick in picks if pick["id"] in valid)
    return (narrative, highlights) if highlights else None


async def build_summary(samples: Sequence[Sample], *, use_llm: bool, model: str) -> Summary:
    """Builds the corpus :class:`Summary`, using the ``claude`` CLI when allowed.

    When ``use_llm`` is set and ``claude`` is on the path, the narrative and
    highlights come from the model; on any failure to produce or parse a result the
    summary falls back to deterministic heuristics, so the export never depends on
    the model succeeding.

    Args:
        samples: The full corpus to summarize.
        use_llm: Whether to consult the ``claude`` CLI for the narrative.
        model: The model to run when consulting ``claude``.

    Returns:
        The assembled :class:`Summary`.
    """
    stats, pool = corpus_stats(samples), candidate_pool(samples)
    if use_llm and claude_available() and (result := await llm_summary(pool, stats, model)) is not None:
        return Summary(stats=stats, highlights=result[1], narrative=result[0])
    return Summary(stats=stats, highlights=tuple(map(Highlight, heuristic_highlight_ids(pool))), narrative=None)


def truncate(text: str, limit: int = CONTEXT_TURN_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_turn(turn: ContextTurn, *, is_trigger: bool = False) -> str:
    cls = f"turn turn-{turn.role}" + (" turn-trigger" if is_trigger else "")
    tools = f'<span class="tools">{escape(" ".join(turn.tool_calls))}</span>' if turn.tool_calls else ""
    return (
        f'<div class="{cls}"><span class="role">{escape(turn.role)}</span>{tools}'
        f"<pre>{escape(truncate(turn.text))}</pre></div>"
    )


def render_context(ctx: ContextSnapshot) -> str:
    turns = [render_turn(turn, is_trigger=turn == ctx.trigger) for turn in ctx.before]
    if ctx.trigger is not None and ctx.trigger not in ctx.before:
        turns.append(render_turn(ctx.trigger, is_trigger=True))
    turns.extend(render_turn(turn) for turn in ctx.after)
    if not turns:
        return ""
    return f'<details class="ctx"><summary>context ({len(turns)} turns)</summary>{"".join(turns)}</details>'


def meta_chips(sample: Sample) -> str:
    payload = sample.payload
    chips = [str(payload[key]) for key in ("detector", "format", "tool", "severity", "track") if payload.get(key)]
    if file := payload.get("file"):
        line = payload.get("line_start") or payload.get("line")
        chips.append(f"{file}:{line}" if line else str(file))
    if sample.origin_path:
        chips.append(project_label(sample.origin_path))
    return "".join(f'<span class="chip">{escape(chip)}</span>' for chip in chips)


def highlight_spans(text: str, spans: Sequence[str]) -> str:
    escaped = escape(text)
    for span in spans:
        escaped = escaped.replace(escape(span), f"<mark>{escape(span)}</mark>")
    return escaped


def render_verdict_stage(verdict: VerdictRow, *, flipped: bool = False) -> str:
    flag = '<span class="flip">flipped across versions</span>' if flipped else ""
    return (
        f'<div class="verdict stage-{escape(verdict.role)}"><div class="vhead">'
        f'<span class="badge cat-{escape(verdict.category)}">{escape(verdict.category)}</span>'
        f'<span class="chip">{escape(verdict.role)} v{verdict.prompt_version} · {escape(verdict.model)}</span>'
        f'<span class="chip">conf {verdict.confidence:.2f}</span>'
        f'<span class="chip">{golden_label(verdict.is_pushback)}</span>{flag}</div>'
        f'<pre class="vsum">{escape(truncate(verdict.what_claude_did))}</pre>'
        f'<pre class="vrat">{escape(truncate(verdict.rationale))}</pre></div>'
    )


def render_refiner_stage(pairs: Sequence[RefinedPairRow], original: str) -> str:
    if not pairs:
        return '<p class="muted">not yet refined</p>'
    cards = "".join(
        f'<div class="pair"><div class="vhead"><span class="chip">pair {pair.pair_index}</span>'
        f'<span class="chip">v{pair.prompt_version} · {escape(pair.model)}</span></div>'
        f'<pre class="paction">{escape(truncate(pair.action))}</pre>'
        f'<blockquote class="pverbatim">{escape(pair.complaint_verbatim)}</blockquote>'
        f'<pre class="pcomplaint">{escape(pair.complaint)}</pre></div>'
        for pair in pairs
    )
    original_html = highlight_spans(original, [pair.complaint_verbatim for pair in pairs])
    return f'<div class="orig"><pre>{original_html}</pre></div>{cards}'


def render_lineage_detail(lineage: Lineage, golden_map: Mapping[str, GoldenRow]) -> str:
    """Renders one candidate's full pipeline trail as a five-stage rail."""
    sample = lineage.sample
    judge_html = "".join(
        render_verdict_stage(verdict, flipped=lineage.flipped) for verdict in lineage.judge_verdicts
    )
    match lineage.auditor_verdict:
        case None:
            auditor_html = '<p class="muted">not audited</p>'
        case auditor:
            agree = lineage.agreement
            auditor_html = render_verdict_stage(auditor) + f'<span class="{agree}">{agree} with judge</span>'
    match golden_status(lineage.dedup_key, lineage.final, golden_map):
        case None:
            golden_html = '<p class="muted">not in golden set</p>'
        case verdict:
            expected = golden_label(golden_map[lineage.dedup_key].expected)
            golden_html = f'<span class="badge {verdict}">golden {verdict} · expected {escape(expected)}</span>'
    return "".join(
        [
            '<div class="lineage">',
            '<section class="stage stage-detector"><h3>1 · detector</h3>',
            f'<header class="card-head"><span class="badge badge-{escape(sample.source_kind)}">'
            f"{escape(sample.source_kind)}</span><time>{escape(sample.occurred_at[:19])}</time>"
            f"{meta_chips(sample)}</header>",
            f'<div class="text"><pre>{escape(sample.text)}</pre></div>{render_context(sample.context)}</section>',
            '<section class="stage stage-judge"><h3>2 · judge</h3>',
            judge_html or '<p class="muted">unjudged</p>',
            "</section>",
            f'<section class="stage stage-auditor"><h3>3 · auditor</h3>{auditor_html}</section>',
            '<section class="stage stage-refiner"><h3>4 · refiner — atomic pairs</h3>',
            render_refiner_stage(lineage.pairs, sample.text),
            "</section>",
            f'<section class="stage stage-golden"><h3>5 · golden gate</h3>{golden_html}</section>',
            "</div>",
        ]
    )
