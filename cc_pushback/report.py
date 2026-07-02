"""Decode the feedback corpus into the data model the dashboard serves.

Holds the data model the dashboard serves — :class:`Sample`, :class:`VerdictRow`,
:class:`RefinedPairRow` with its :class:`EvidenceRow` (read from the shared
correction ledger), and the :class:`Lineage` that stitches a candidate's whole
pipeline trail together — plus the corpus :class:`Summary` (written by the
``claude`` CLI). The FastAPI surface and the JSON shapes it serves — including
the client-rendered lineage — live in :mod:`cc_pushback.dashboard`.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript.context import ContextWindow
from cc_transcript.mining import NOISE_FLOOR
from cc_transcript.mining.confidence import from_payload

from cc_pushback.claude import run_claude

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, Literal

    from cc_transcript.corrections import Correction
    from cc_transcript.mining import CandidateSignal

    from cc_pushback.evaluate import GoldenRow

CONTEXT_TURN_LIMIT = 700
SAMPLE_TEXT_LIMIT = 400
HIGHLIGHT_POOL_PER_KIND = 8

SUMMARY_SYSTEM = """\
You analyze a developer's "pushback" — the corrective feedback they give an AI coding assistant.
You receive corpus statistics and a numbered pool of real feedback samples.
Return ONLY a JSON object, with no prose around it, of exactly this shape:
{"narrative": "<2-4 sentences on the developer's pushback style and recurring themes>",
 "highlights": [{"id": <sample id>, "why": "<one short clause on why it is representative>"}]}
Pick 8-12 highlights, only from the provided sample ids, favoring variety across feedback kinds.
"""


@dataclass(frozen=True, slots=True)
class Sample:
    """One stored feedback event, decoded from a :meth:`FeedbackStore.candidates` row.

    Attributes:
        id: The event's database id.
        source_kind: Which detector produced it.
        occurred_at: The ISO timestamp of the feedback.
        text: The verbatim pushback text.
        payload: The detector-specific metadata, decoded from ``payload_json``.
        window: The durable context window around the feedback.
        origin_path: The transcript file the event came from — a display hint only.
        session_id: The session the event came from.
        signal: The de-noising confidence signal, decoded from the payload.
    """

    id: int
    source_kind: str
    occurred_at: str
    text: str
    payload: Mapping[str, Any]
    window: ContextWindow
    origin_path: str | None
    session_id: str | None
    signal: CandidateSignal

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Sample:
        """Decodes a :meth:`FeedbackStore.candidates` row into a :class:`Sample`."""
        payload = json.loads(str(row["payload_json"])) if row["payload_json"] else {}
        return cls(
            id=int(str(row["id"])),
            source_kind=str(row["source_kind"]),
            occurred_at=str(row["occurred_at"]),
            text=str(row["text"]),
            payload=payload,
            window=ContextWindow.from_json(str(row["context_json"])),
            origin_path=str(row["origin_path"]) if row["origin_path"] else None,
            session_id=str(row["session_id"]) if row["session_id"] else None,
            signal=from_payload(payload["signal"]),
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
class EvidenceRow:
    """One refined pair's grounding code evidence, read from the shared ledger.

    Attributes:
        file_path: The file the incorrect edit touched.
        incorrect: The incorrect edit's verbatim ``(old, new)`` content.
        correct: The correction's verbatim ``(old, new)`` content, when one exists.
        source: Where the correction came from — ``session`` or ``git`` — when one exists.
    """

    file_path: str
    incorrect: tuple[str, str]
    correct: tuple[str, str] | None
    source: str | None

    @classmethod
    def from_correction(cls, correction: Correction) -> EvidenceRow:
        """Decodes a shared-ledger :class:`~cc_transcript.corrections.Correction` row."""
        return cls(
            file_path=correction.incorrect_file,
            incorrect=(correction.incorrect_old, correction.incorrect_new),
            correct=None
            if correction.correction_origin is None
            else (str(correction.correction_old), str(correction.correction_new)),
            source=correction.correction_origin,
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
        evidence: The enrich stage's code evidence from the shared ledger, when the
            pair's anchor carries a correction.
    """

    pair_index: int
    action: str
    complaint_verbatim: str
    complaint: str
    prompt_version: int
    model: str
    evidence: EvidenceRow | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, object], *, evidence: EvidenceRow | None = None) -> RefinedPairRow:
        """Decodes a :meth:`FeedbackStore.lineage` pair row into a :class:`RefinedPairRow`.

        The pair's grounding ``evidence`` comes from the shared ledger, resolved by
        the caller from the pair's pushback anchor.
        """
        return cls(
            pair_index=int(str(row["pair_index"])),
            action=str(row["action"]),
            complaint_verbatim=str(row["complaint_verbatim"]),
            complaint=str(row["complaint"]),
            prompt_version=int(str(row["prompt_version"])),
            model=str(row["model"]),
            evidence=evidence,
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
    def from_lineage(
        cls, data: Mapping[str, Any], *, evidence_of: Callable[[Mapping[str, object]], EvidenceRow | None]
    ) -> Lineage:
        """Builds a :class:`Lineage` from a :meth:`FeedbackStore.lineage` result.

        Each pair's grounding evidence comes from ``evidence_of``, which resolves the
        pair's pushback anchor against the shared correction ledger.
        """
        return cls(
            sample=Sample.from_row(data),
            dedup_key=str(data["dedup_key"]),
            verdicts=tuple(VerdictRow.from_row(row) for row in data["verdicts"]),
            pairs=tuple(RefinedPairRow.from_row(row, evidence=evidence_of(row)) for row in data["pairs"]),
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
        by_category_kind: Accepted-event counts keyed by category then source kind,
            categories ordered by descending total.
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
    by_category_kind: Mapping[str, Mapping[str, int]]
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
        narrative: The prose description of the developer's pushback style,
            written by the ``claude`` CLI.
    """

    stats: CorpusStats
    highlights: tuple[Highlight, ...]
    narrative: str


def is_noise(sample: Sample) -> bool:
    return sample.signal.confidence < NOISE_FLOOR


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
    composition: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidates:
        if row["is_pushback"]:
            composition[str(row["category"])][str(row["source_kind"])] += 1
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
        by_category_kind={
            category: dict(kinds.most_common())
            for category, kinds in sorted(composition.items(), key=lambda item: -item[1].total())
        },
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


def parse_summary_json(raw: str) -> tuple[str, list[dict[str, Any]]]:
    if not (match := re.search(r"\{.*\}", raw, re.DOTALL)):
        raise ValueError(f"no JSON object in summary output: {raw[:200]}")
    match json.loads(match.group(0)):
        case {"narrative": str(narrative), "highlights": list(picks)}:
            return narrative, [p for p in picks if isinstance(p, dict) and isinstance(p.get("id"), int)]
        case data:
            raise ValueError(f"summary JSON missing narrative/highlights: {data}")


async def build_summary(samples: Sequence[Sample], *, model: str) -> Summary:
    """Builds the corpus :class:`Summary` via the ``claude`` CLI.

    The narrative and highlights come from the model; any subprocess or parse
    failure raises.

    Args:
        samples: The full corpus to summarize.
        model: The model to run.

    Returns:
        The assembled :class:`Summary`.
    """
    stats, pool = corpus_stats(samples), candidate_pool(samples)
    narrative, picks = parse_summary_json(
        await run_claude(summary_prompt(pool, stats), system=SUMMARY_SYSTEM, model=model)
    )
    valid = {s.id for group in pool.values() for s in group}
    if not (highlights := tuple(Highlight(pick["id"], pick.get("why")) for pick in picks if pick["id"] in valid)):
        raise ValueError("summary returned no valid highlight ids")
    return Summary(stats=stats, highlights=highlights, narrative=narrative)


def truncate(text: str, limit: int = CONTEXT_TURN_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
