from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.corrections import Correction, CorrectionLog, Origin
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import firm
from cc_transcript.mining.confidence import to_payload

from cc_steer.dashboard import build_app, language_of, serialize_lineage
from cc_steer.evaluate import GoldenRow
from cc_steer.refine import RefinedPair, Refinement
from cc_steer.report import (
    EvidenceRow,
    Lineage,
    RefinedPairRow,
    Sample,
    Summary,
    VerdictRow,
    corpus_stats,
    golden_status,
)
from cc_steer.triage import JUDGE, Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio


def vrow(role: str, version: int, category: str, *, is_steering: bool) -> VerdictRow:
    return VerdictRow(
        role=role,
        prompt_version=version,
        model="sonnet" if role == JUDGE else "opus",
        category=category,
        is_steering=is_steering,
        what_claude_did="vendored the dep",
        confidence=0.9,
        rationale="faults the vendoring",
        judged_at="2026-01-01T00:00:00",
    )


def prow(index: int, verbatim: str) -> RefinedPairRow:
    return RefinedPairRow(
        pair_index=index, action="vendored the dep", direction_verbatim=verbatim, direction=f"c{index}",
        prompt_version=1, model="sonnet",
    )


def preview_window(trigger: str | None) -> ContextWindow:
    return ContextWindow(
        anchor=EventRef(SessionId("s1"), EventUuid("u1")),
        before=(),
        trigger=None if trigger is None else TurnRef(role="assistant", refs=(), preview=trigger, tool_digests=()),
        after=(),
        fidelity="summary",
        preview_chars=200,
    )


def lineage(
    verdicts: Sequence[VerdictRow], pairs: Sequence[RefinedPairRow] = (), *, text: str = "no, dont vendor it"
) -> Lineage:
    sample = Sample(
        id=1,
        source_kind="transcript_message",
        occurred_at="2026-01-01T00:00:00",
        text=text,
        payload={},
        window=preview_window("I vendored the lib"),
        origin_path="/h-Code-proj/s.jsonl",
        session_id="s",
        signal=firm("transcript_message"),
    )
    return Lineage(sample=sample, dedup_key="k1", verdicts=tuple(verdicts), pairs=tuple(pairs))


def test_verdict_row_from_row_coerces_steering_to_bool() -> None:
    row = {
        "role": "judge", "prompt_version": 2, "model": "sonnet", "category": "wrong_approach",
        "is_steering": 1, "what_claude_did": "did x", "confidence": 0.8, "rationale": "r",
        "judged_at": "2026-01-01T00:00:00",
    }
    verdict = VerdictRow.from_row(row)
    assert verdict.is_steering is True and verdict.prompt_version == 2


def pair_row(**overrides: object) -> dict[str, object]:
    return {
        "pair_index": 1, "action": "a", "direction_verbatim": "v", "direction": "c",
        "prompt_version": 1, "model": "sonnet", "session_id": "s", "event_uuid": "u1",
    } | overrides


def git_correction(*, correction_origin: Origin | None = "git") -> Correction:
    has_fix = correction_origin is not None
    return Correction(
        ts_ms=1, session_id=SessionId("s"), source="cc-steer", anchor_uuid=EventUuid("u1"),
        incorrect_digest=None, incorrect_file="/repo/a.py", incorrect_old="bad", incorrect_new="worse",
        correction_origin=correction_origin,
        correction_old="worse" if has_fix else None,
        correction_new="good" if has_fix else None,
    )


def test_refined_pair_row_from_row_carries_no_evidence_by_default() -> None:
    pair = RefinedPairRow.from_row(pair_row())
    assert pair.pair_index == 1 and pair.direction_verbatim == "v"
    assert pair.evidence is None


def test_refined_pair_row_attaches_resolved_evidence() -> None:
    evidence = EvidenceRow.from_correction(git_correction())
    pair = RefinedPairRow.from_row(pair_row(), evidence=evidence)
    assert pair.evidence == evidence


def test_evidence_row_from_correction_decodes_a_git_fix() -> None:
    assert EvidenceRow.from_correction(git_correction()) == EvidenceRow(
        file_path="/repo/a.py", incorrect=("bad", "worse"), correct=("worse", "good"), source="git",
    )


def test_evidence_row_from_correction_has_no_correct_side_when_origin_is_none() -> None:
    evidence = EvidenceRow.from_correction(git_correction(correction_origin=None))
    assert evidence.correct is None and evidence.source is None


@pytest.mark.parametrize(
    ("path", "language"),
    [
        pytest.param("/repo/a.py", "py", id="py"),
        pytest.param("/repo/src/UI.TSX", "tsx", id="uppercase-ext"),
        pytest.param("/repo/main.rs", "rs", id="rs"),
        pytest.param("/repo/Dockerfile", "dockerfile", id="extensionless"),
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty"),
    ],
)
def test_language_of(path: str | None, language: str | None) -> None:
    assert language_of(path) == language


@pytest.mark.parametrize(
    ("verdicts", "flipped"),
    [
        pytest.param([], False, id="no-judge"),
        pytest.param([vrow(JUDGE, 1, "status_update", is_steering=False)], False, id="single-judge"),
        pytest.param(
            [vrow(JUDGE, 1, "status_update", is_steering=False), vrow(JUDGE, 2, "wrong_approach", is_steering=True)],
            True,
            id="side-change-across-versions",
        ),
    ],
)
def test_lineage_flipped(verdicts: list[VerdictRow], flipped: bool) -> None:
    assert lineage(verdicts).flipped is flipped


def test_lineage_final_is_latest_judge_version() -> None:
    lin = lineage(
        [vrow(JUDGE, 1, "status_update", is_steering=False), vrow(JUDGE, 2, "wrong_approach", is_steering=True)]
    )
    assert lin.final is not None and lin.final.category == "wrong_approach"


@pytest.mark.parametrize(
    ("auditor_steering", "agreement"),
    [
        pytest.param(True, "agree", id="agree"),
        pytest.param(False, "disagree", id="disagree"),
        pytest.param(None, None, id="unaudited"),
    ],
)
def test_lineage_agreement(auditor_steering: bool | None, agreement: str | None) -> None:
    verdicts = [vrow(JUDGE, 1, "wrong_approach", is_steering=True)]
    if auditor_steering is not None:
        verdicts.append(vrow("auditor", 1, "wrong_approach", is_steering=auditor_steering))
    assert lineage(verdicts).agreement == agreement


def test_golden_status() -> None:
    gold = GoldenRow(dedup_key="k1", source_kind="transcript_message", text="t", expected=True, note="n")
    golden_map = {"k1": gold}
    steering = vrow(JUDGE, 1, "wrong_approach", is_steering=True)
    noise = vrow(JUDGE, 1, "status_update", is_steering=False)
    assert golden_status("k1", steering, golden_map) == "pass"
    assert golden_status("k1", noise, golden_map) == "fail"
    assert golden_status("other", steering, golden_map) is None
    assert golden_status("k1", None, golden_map) is None


def test_serialize_lineage_shapes_five_stages_and_keeps_raw_text() -> None:
    lin = lineage(
        [vrow(JUDGE, 1, "status_update", is_steering=False), vrow(JUDGE, 2, "wrong_approach", is_steering=True)],
        [prow(0, "dont vendor")],
        text="<script>alert(1)</script> dont vendor it",
    )
    auditor = lin.verdicts + (vrow("auditor", 2, "status_update", is_steering=False),)
    lin = Lineage(sample=lin.sample, dedup_key=lin.dedup_key, verdicts=auditor, pairs=lin.pairs)
    data = serialize_lineage(lin, {})

    assert set(data) == {"detector", "judge", "auditor", "refiner", "golden"}
    assert [verdict["prompt_version"] for verdict in data["judge"]] == [1, 2]
    assert all(verdict["flipped"] for verdict in data["judge"])  # judge side changed v1 -> v2
    assert data["judge"][-1]["category"] == "wrong_approach"
    assert data["auditor"]["agreement"] == "disagree"  # auditor noise vs final steering
    assert data["refiner"]["spans"] == ["dont vendor"]  # the direction the client highlights in the original
    assert data["refiner"]["pairs"][0]["evidence"] is None  # unenriched pair, no diff
    assert data["golden"] is None
    assert data["detector"]["context"]["turns"][0]["is_trigger"] is True
    # raw, un-escaped text — the browser escapes at render time, not the server
    assert data["detector"]["text"] == "<script>alert(1)</script> dont vendor it"


def test_serialize_lineage_keeps_the_full_untruncated_diff() -> None:
    long_call = "danger(" + "x" * 900 + ")"
    pair = RefinedPairRow(
        pair_index=0, action="vendored the dep", direction_verbatim="dont vendor", direction="c0",
        prompt_version=1, model="sonnet",
        evidence=EvidenceRow(
            file_path="/repo/a.py",
            incorrect=(f"if x < 1:\n    {long_call}", "safe()"),
            correct=("safe()", "safer()"),
            source="git",
        ),
    )
    data = serialize_lineage(lineage([vrow(JUDGE, 1, "wrong_approach", is_steering=True)], [pair]), {})
    evidence = data["refiner"]["pairs"][0]["evidence"]
    assert evidence["file_path"] == "/repo/a.py" and evidence["source"] == "git"
    assert evidence["incorrect"]["old"] == f"if x < 1:\n    {long_call}"  # full, untruncated, unescaped
    assert "x" * 900 in evidence["incorrect"]["old"]
    assert evidence["correct"] == {"old": "safe()", "new": "safer()"}


async def seed(store: FeedbackStore) -> None:
    conn = store.store.conn
    trigger = preview_window("I vendored it").to_json()
    empty = preview_window(None).to_json()
    payload_json = json.dumps({"signal": to_payload(firm("transcript_message"))})
    rows = [
        ("k1", "no, dont vendor it; bake it in", trigger),
        ("k2", "run the tests not the build", empty),
        ("k3", "thanks, looks good", empty),
    ]
    for i, (key, text, ctx) in enumerate(rows):
        await conn.execute(
            "INSERT INTO feedback_events (dedup_key, source_kind, session_id, event_uuid, "
            "occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (key, "transcript_message", "s", f"u{i}", f"2026-01-0{i + 1}T00:00:00",
             text, payload_json, ctx, "0.1", "2026-01-01T00:00:00", "/h-Code-proj/s.jsonl"),
        )

    def verdict(category: str) -> Verdict:
        return Verdict.model_validate(
            {"category": category, "what_claude_did": "did x", "confidence": 0.9, "rationale": "r"}
        )

    await store.record_verdict(
        "k1", verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        "k1", verdict("status_update"), role="auditor", prompt_version=1, model="opus", fidelity="full"
    )
    await store.record_verdict(
        "k2", verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        "k3", verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    pair = RefinedPair(action="vendored it", direction_verbatim="dont vendor it", direction="do not vendor")
    await store.record_refinement("k1", Refinement(pairs=[pair]), prompt_version=1, model="sonnet")


def k1_correction(
    *,
    incorrect: tuple[str, str],
    correct: tuple[str, str] | None,
    file: str = "/repo/a.py",
    source: Origin | None = "git",
) -> Correction:
    return Correction(
        ts_ms=1, session_id=SessionId("s"), source="cc-steer", anchor_uuid=EventUuid("u0"),
        incorrect_digest=None, incorrect_file=file, incorrect_old=incorrect[0], incorrect_new=incorrect[1],
        correction_origin=source,
        correction_old=None if correct is None else correct[0],
        correction_new=None if correct is None else correct[1],
    )


def enrich_k1(correction: Correction) -> None:
    """Grounds k1's steering anchor (session ``s``, uuid ``u0``) in the shared ledger."""
    CorrectionLog.open().append(correction)


async def client(store: FeedbackStore) -> httpx.AsyncClient:
    stats = corpus_stats([Sample.from_row(row) for row in await store.candidates()])
    summary = Summary(stats=stats, highlights=(), narrative="Terse and direct.")
    transport = httpx.ASGITransport(app=build_app(store, summary=summary))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_api_pairs_returns_atomic_rows(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        body = (await http.get("/api/pairs")).json()
    assert len(body["pairs"]) == 1
    pair = body["pairs"][0]
    assert pair["dedup_key"] == "k1" and pair["category"] == "wrong_approach"
    assert pair["direction"] == "do not vendor" and pair["project"] == "proj"
    assert pair["direction_verbatim"] == "dont vendor it"
    assert pair["evidence"] is None  # unenriched pair, card unchanged
    assert pair["language"] is None  # no evidence, no language facet value


async def test_api_pairs_carries_code_evidence(store: FeedbackStore) -> None:
    await seed(store)
    enrich_k1(k1_correction(incorrect=("x = eval(s)", "y = eval(t)"), correct=("y = eval(t)", "y = json.loads(t)")))
    async with await client(store) as http:
        pair = (await http.get("/api/pairs")).json()["pairs"][0]
    assert pair["evidence"] == {
        "file_path": "/repo/a.py",
        "source": "git",
        "incorrect": {"old": "x = eval(s)", "new": "y = eval(t)"},
        "correct": {"old": "y = eval(t)", "new": "y = json.loads(t)"},
    }
    assert pair["language"] == "py"  # derived from the evidence file extension


async def test_api_pairs_clips_evidence_sides_for_the_list(store: FeedbackStore) -> None:
    await seed(store)
    enrich_k1(k1_correction(incorrect=("a" * 400, "b"), correct=None, source=None))
    async with await client(store) as http:
        evidence = (await http.get("/api/pairs")).json()["pairs"][0]["evidence"]
    assert evidence["incorrect"]["old"] == "a" * 280 + "…"
    assert evidence["correct"] is None and evidence["source"] is None


async def test_api_pairs_no_ledger_correction_keeps_evidence_none(store: FeedbackStore) -> None:
    await seed(store)  # k1's anchor carries no correction
    async with await client(store) as http:
        pair = (await http.get("/api/pairs")).json()["pairs"][0]
    assert pair["evidence"] is None
    assert pair["direction_verbatim"] == "dont vendor it"  # the card payload is otherwise unchanged


async def test_api_lineage_shows_the_full_diff(store: FeedbackStore) -> None:
    await seed(store)
    long_new = "json.loads(" + "x" * 900 + ")"
    enrich_k1(k1_correction(incorrect=("bad < worse\nstill bad", long_new), correct=None, source=None))
    async with await client(store) as http:
        data = (await http.get("/api/lineage/k1")).json()
    evidence = data["refiner"]["pairs"][0]["evidence"]
    assert evidence["incorrect"]["old"] == "bad < worse\nstill bad"  # raw, un-split; the client renders the diff
    assert "x" * 900 in evidence["incorrect"]["new"]  # the full diff, untruncated
    assert evidence["correct"] is None  # no correction
    assert evidence["source"] is None  # not a git fix


async def test_api_candidates_covers_every_status(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        candidates = (await http.get("/api/candidates")).json()["candidates"]
    by_key = {row["dedup_key"]: row for row in candidates}
    assert by_key["k1"]["status"] == "refined" and by_key["k1"]["agreement"] == "disagree"
    assert by_key["k2"]["status"] == "accepted"
    assert by_key["k3"]["status"] == "noise"


async def test_api_lineage_renders_detail_and_404s(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        ok = await http.get("/api/lineage/k1")
        missing = await http.get("/api/lineage/nope")
    assert ok.status_code == 200
    data = ok.json()
    assert data["detector"]["context"]["turns"][0]["is_trigger"] is True
    assert data["refiner"]["pairs"][0]["direction"] == "do not vendor"
    assert data["judge"][0]["category"] == "wrong_approach"
    assert missing.status_code == 404


async def test_api_stats_shape(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        stats = (await http.get("/api/stats")).json()
    assert stats["narrative"] == "Terse and direct."
    assert stats["pipeline"]["refined"] == 1 and stats["pipeline"]["accepted"] == 2
    assert stats["pipeline"]["noise_judged"] == 1 and stats["pipeline"]["total_pairs"] == 1
    assert stats["pipeline"]["by_category_kind"] == {"wrong_approach": {"transcript_message": 2}}
    assert stats["corpus"]["total"] == 3


async def test_root_serves_index_and_static_assets(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        page = await http.get("/")
        main_js = await http.get("/static/js/main.js")
        filters_js = await http.get("/static/js/filters.js")
        cards_js = await http.get("/static/js/cards.js")
        lineage_js = await http.get("/static/js/lineage.js")
        base_css = await http.get("/static/base.css")
        dash_css = await http.get("/static/dashboard.css")
    assert page.status_code == 200
    for node in ('id="filters"', 'id="list"', 'id="search"', 'id="active"', 'id="detail"',
                 'id="backdrop"', 'id="stats-toggle"'):
        assert node in page.text  # the static body skeleton
    for ref in ('href="/static/base.css"', 'href="/static/dashboard.css"', 'src="/static/js/main.js"'):
        assert ref in page.text  # the page links its assets rather than inlining them
    assert main_js.status_code == 200 and "text/javascript" in main_js.headers["content-type"]
    assert all(t in filters_js.text for t in ("GROUPS", "renderFacets", "matchRow", '"Language"', "has code"))
    assert "evidenceHtml" in cards_js.text  # list-card evidence renderer
    assert "lineageHtml" in lineage_js.text and "stage-detector" in lineage_js.text  # client lineage renderer
    assert base_css.status_code == 200 and ".pane .del" in base_css.text and "chip-git" in base_css.text
    assert dash_css.status_code == 200 and "#filters" in dash_css.text and "#detail" in dash_css.text
