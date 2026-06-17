"""The FastAPI dashboard: routes and JSON shapes over the feedback store.

Serves the static client (the files under ``assets/``) and a JSON API — the
refined pairs (the pipeline's deliverable), every candidate behind them, the
corpus stats, and one candidate's full lineage as structured JSON the client
renders into a five-stage rail. The data model lives in :mod:`cc_pushback.report`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from cc_pushback import report
from cc_pushback.evaluate import load_golden

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cc_transcript.context import ContextWindow, TurnRef

    from cc_pushback.evaluate import GoldenRow
    from cc_pushback.report import EvidenceRow, Lineage, RefinedPairRow, Sample, Summary, VerdictRow
    from cc_pushback.store import FeedbackStore

LIST_TEXT_LIMIT = 280
ASSETS = Path(__file__).with_name("assets")


def project_of(origin_path: object) -> str | None:
    return report.project_label(str(origin_path)) if origin_path else None


def language_of(file_path: str | None) -> str | None:
    if not file_path:
        return None
    return (Path(file_path).suffix.lstrip(".") or Path(file_path).name).lower()


def edit_json(old: str, new: str) -> dict[str, str]:
    return {"old": report.truncate(old, LIST_TEXT_LIMIT), "new": report.truncate(new, LIST_TEXT_LIMIT)}


def serialize_evidence(row: Mapping[str, object]) -> dict[str, object] | None:
    if (evidence := report.EvidenceRow.from_row(row)) is None:
        return None
    return {
        "file_path": evidence.file_path,
        "source": evidence.source,
        "incorrect": edit_json(*evidence.incorrect),
        "correct": None if evidence.correct is None else edit_json(*evidence.correct),
    }


def serialize_pair(row: Mapping[str, object]) -> dict[str, object]:
    evidence = serialize_evidence(row)
    return {
        "dedup_key": row["dedup_key"],
        "pair_index": row["pair_index"],
        "action": report.truncate(str(row["action"]), LIST_TEXT_LIMIT),
        "complaint_verbatim": report.truncate(str(row["complaint_verbatim"]), LIST_TEXT_LIMIT),
        "complaint": row["complaint"],
        "category": row["category"],
        "source_kind": row["source_kind"],
        "project": project_of(row["origin_path"]),
        "occurred_at": str(row["occurred_at"])[:19],
        "evidence": evidence,
        "language": language_of(str(evidence["file_path"])) if evidence else None,
    }


def serialize_candidate(row: Mapping[str, object], golden_map: Mapping[str, GoldenRow]) -> dict[str, object]:
    key = str(row["dedup_key"])
    audited = row["auditor_is_pushback"] is not None and row["is_pushback"] is not None
    in_golden = key in golden_map and row["is_pushback"] is not None
    return {
        "dedup_key": key,
        "source_kind": row["source_kind"],
        "occurred_at": str(row["occurred_at"])[:19],
        "project": project_of(row["origin_path"]),
        "status": report.candidate_status(row),
        "category": row["category"],
        "confidence": row["confidence"],
        "pair_count": row["pair_count"],
        "flipped": bool(row["flipped"]),
        "agreement": ("agree" if bool(row["auditor_is_pushback"]) == bool(row["is_pushback"]) else "disagree")
        if audited
        else None,
        "golden": ("pass" if bool(row["is_pushback"]) == golden_map[key].expected else "fail") if in_golden else None,
        "text": report.truncate(str(row["text"]), LIST_TEXT_LIMIT),
    }


def lineage_meta(sample: Sample) -> list[str]:
    payload = sample.payload
    chips = [str(payload[key]) for key in ("detector", "format", "tool", "severity", "track") if payload.get(key)]
    if file := payload.get("file"):
        line = payload.get("line_start") or payload.get("line")
        chips.append(f"{file}:{line}" if line else str(file))
    if sample.origin_path:
        chips.append(report.project_label(sample.origin_path))
    return chips


def serialize_turn(ref: TurnRef, *, is_trigger: bool) -> dict[str, object]:
    return {
        "role": ref.role,
        "preview": report.truncate(ref.preview),
        "tool_calls": len(ref.tool_digests),
        "is_trigger": is_trigger,
    }


def serialize_context(window: ContextWindow) -> dict[str, object] | None:
    turns = [
        *(serialize_turn(ref, is_trigger=False) for ref in window.before),
        *(() if window.trigger is None else (serialize_turn(window.trigger, is_trigger=True),)),
        *(serialize_turn(ref, is_trigger=False) for ref in window.after),
    ]
    return {"turns": turns} if turns else None


def serialize_verdict(verdict: VerdictRow, *, flipped: bool) -> dict[str, object]:
    return {
        "role": verdict.role,
        "category": verdict.category,
        "prompt_version": verdict.prompt_version,
        "model": verdict.model,
        "confidence": verdict.confidence,
        "is_pushback": verdict.is_pushback,
        "what_claude_did": report.truncate(verdict.what_claude_did),
        "rationale": report.truncate(verdict.rationale),
        "flipped": flipped,
    }


def serialize_evidence_detail(evidence: EvidenceRow) -> dict[str, object]:
    return {
        "file_path": evidence.file_path,
        "source": evidence.source,
        "incorrect": {"old": evidence.incorrect[0], "new": evidence.incorrect[1]},
        "correct": None if evidence.correct is None else {"old": evidence.correct[0], "new": evidence.correct[1]},
        "note": evidence.note,
    }


def serialize_detail_pair(pair: RefinedPairRow) -> dict[str, object]:
    return {
        "pair_index": pair.pair_index,
        "prompt_version": pair.prompt_version,
        "model": pair.model,
        "action": report.truncate(pair.action),
        "complaint_verbatim": pair.complaint_verbatim,
        "complaint": pair.complaint,
        "evidence": None if pair.evidence is None else serialize_evidence_detail(pair.evidence),
    }


def serialize_lineage(lineage: Lineage, golden_map: Mapping[str, GoldenRow]) -> dict[str, object]:
    sample = lineage.sample
    match lineage.auditor_verdict:
        case None:
            auditor = None
        case verdict:
            auditor = serialize_verdict(verdict, flipped=False) | {"agreement": lineage.agreement}
    match report.golden_status(lineage.dedup_key, lineage.final, golden_map):
        case None:
            golden = None
        case status:
            golden = {"verdict": status, "expected": golden_map[lineage.dedup_key].expected}
    return {
        "detector": {
            "source_kind": sample.source_kind,
            "occurred_at": sample.occurred_at[:19],
            "meta": lineage_meta(sample),
            "text": sample.text,
            "context": serialize_context(sample.window),
        },
        "judge": [serialize_verdict(verdict, flipped=lineage.flipped) for verdict in lineage.judge_verdicts],
        "auditor": auditor,
        "refiner": {
            "original": sample.text,
            "spans": [pair.complaint_verbatim for pair in lineage.pairs],
            "pairs": [serialize_detail_pair(pair) for pair in lineage.pairs],
        },
        "golden": golden,
    }


def build_app(store: FeedbackStore, *, summary: Summary) -> FastAPI:
    """Builds the dashboard app over an open store, serving the client and JSON API.

    Args:
        store: The open feedback store the routes query live.
        summary: The corpus summary, built once; its narrative and highlights are
            cached and served at ``/api/stats``.

    Returns:
        The configured :class:`fastapi.FastAPI` application.
    """
    golden_map = {row.dedup_key: row for row in load_golden()}
    app = FastAPI(title="cc-pushback")
    app.mount("/static", StaticFiles(directory=ASSETS), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (ASSETS / "index.html").read_text()

    @app.get("/api/pairs")
    async def api_pairs() -> dict[str, object]:
        return {"pairs": [serialize_pair(row) for row in await store.pairs()]}

    @app.get("/api/candidates")
    async def api_candidates() -> dict[str, object]:
        return {"candidates": [serialize_candidate(row, golden_map) for row in await store.candidates()]}

    @app.get("/api/lineage/{dedup_key}")
    async def api_lineage(dedup_key: str) -> dict[str, object]:
        if not (data := await store.lineage(dedup_key)):
            raise HTTPException(status_code=404, detail="unknown dedup_key")
        return serialize_lineage(report.Lineage.from_lineage(data), golden_map)

    @app.get("/api/stats")
    async def api_stats() -> dict[str, object]:
        candidates = await store.candidates()
        return {
            "corpus": asdict(report.corpus_stats([report.Sample.from_row(row) for row in candidates])),
            "pipeline": asdict(report.pipeline_stats(candidates, golden_map=golden_map)),
            "narrative": summary.narrative,
            "highlights": [asdict(highlight) for highlight in summary.highlights],
        }

    return app
