from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import firm
from cc_transcript.mining.confidence import to_payload

from cc_pushback.dashboard import build_app, language_of
from cc_pushback.enrich import CodeEvidence, EditSide
from cc_pushback.evaluate import GoldenRow
from cc_pushback.refine import RefinedPair, Refinement
from cc_pushback.report import (
    EvidenceRow,
    Lineage,
    RefinedPairRow,
    Sample,
    VerdictRow,
    build_summary,
    golden_status,
    render_lineage_detail,
)
from cc_pushback.triage import JUDGE, Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_pushback.store import FeedbackStore

pytestmark = pytest.mark.anyio


def vrow(role: str, version: int, category: str, *, is_pushback: bool) -> VerdictRow:
    return VerdictRow(
        role=role,
        prompt_version=version,
        model="sonnet" if role == JUDGE else "opus",
        category=category,
        is_pushback=is_pushback,
        what_claude_did="vendored the dep",
        confidence=0.9,
        rationale="faults the vendoring",
        judged_at="2026-01-01T00:00:00",
    )


def prow(index: int, verbatim: str) -> RefinedPairRow:
    return RefinedPairRow(
        pair_index=index, action="vendored the dep", complaint_verbatim=verbatim, complaint=f"c{index}",
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


def test_verdict_row_from_row_coerces_pushback_to_bool() -> None:
    row = {
        "role": "judge", "prompt_version": 2, "model": "sonnet", "category": "wrong_approach",
        "is_pushback": 1, "what_claude_did": "did x", "confidence": 0.8, "rationale": "r",
        "judged_at": "2026-01-01T00:00:00",
    }
    verdict = VerdictRow.from_row(row)
    assert verdict.is_pushback is True and verdict.prompt_version == 2


def pair_row(**overrides: object) -> dict[str, object]:
    return {
        "pair_index": 1, "action": "a", "complaint_verbatim": "v", "complaint": "c",
        "prompt_version": 1, "model": "sonnet",
        "evidence_kind": None, "evidence_file_path": None, "incorrect_old": None, "incorrect_new": None,
        "correct_old": None, "correct_new": None, "evidence_note": None, "evidence_source": None,
    } | overrides


def test_refined_pair_row_from_row() -> None:
    pair = RefinedPairRow.from_row(pair_row())
    assert pair.pair_index == 1 and pair.complaint_verbatim == "v"
    assert pair.evidence is None


def test_refined_pair_row_decodes_code_evidence() -> None:
    pair = RefinedPairRow.from_row(pair_row(
        evidence_kind="code", evidence_file_path="/repo/a.py",
        incorrect_old="bad", incorrect_new="worse", correct_old="worse", correct_new="good",
        evidence_note="faults the edit", evidence_source="git",
    ))
    assert pair.evidence == EvidenceRow(
        file_path="/repo/a.py", incorrect=("bad", "worse"), correct=("worse", "good"),
        note="faults the edit", source="git",
    )


def test_evidence_row_is_none_for_no_code() -> None:
    assert EvidenceRow.from_row(pair_row(evidence_kind="no_code", evidence_note="not about code")) is None


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
    ("verdicts", "pairs", "status", "flipped"),
    [
        pytest.param([], [], "unjudged", False, id="no-judge"),
        pytest.param([vrow(JUDGE, 1, "status_update", is_pushback=False)], [], "noise", False, id="noise"),
        pytest.param([vrow(JUDGE, 1, "wrong_approach", is_pushback=True)], [], "accepted", False, id="accepted"),
        pytest.param(
            [vrow(JUDGE, 1, "status_update", is_pushback=False), vrow(JUDGE, 2, "wrong_approach", is_pushback=True)],
            [prow(0, "dont")],
            "refined",
            True,
            id="refined-and-flipped",
        ),
    ],
)
def test_lineage_status_and_flipped(
    verdicts: list[VerdictRow], pairs: list[RefinedPairRow], status: str, flipped: bool
) -> None:
    lin = lineage(verdicts, pairs)
    assert lin.status == status
    assert lin.flipped is flipped


def test_lineage_final_is_latest_judge_version() -> None:
    lin = lineage(
        [vrow(JUDGE, 1, "status_update", is_pushback=False), vrow(JUDGE, 2, "wrong_approach", is_pushback=True)]
    )
    assert lin.final is not None and lin.final.category == "wrong_approach"


@pytest.mark.parametrize(
    ("auditor_pushback", "agreement"),
    [
        pytest.param(True, "agree", id="agree"),
        pytest.param(False, "disagree", id="disagree"),
        pytest.param(None, None, id="unaudited"),
    ],
)
def test_lineage_agreement(auditor_pushback: bool | None, agreement: str | None) -> None:
    verdicts = [vrow(JUDGE, 1, "wrong_approach", is_pushback=True)]
    if auditor_pushback is not None:
        verdicts.append(vrow("auditor", 1, "wrong_approach", is_pushback=auditor_pushback))
    assert lineage(verdicts).agreement == agreement


def test_golden_status() -> None:
    gold = GoldenRow(dedup_key="k1", source_kind="transcript_message", text="t", expected=True, note="n")
    golden_map = {"k1": gold}
    pushback = vrow(JUDGE, 1, "wrong_approach", is_pushback=True)
    noise = vrow(JUDGE, 1, "status_update", is_pushback=False)
    assert golden_status("k1", pushback, golden_map) == "pass"
    assert golden_status("k1", noise, golden_map) == "fail"
    assert golden_status("other", pushback, golden_map) is None
    assert golden_status("k1", None, golden_map) is None


def test_render_lineage_detail_has_five_stages_and_escapes() -> None:
    lin = lineage(
        [vrow(JUDGE, 1, "status_update", is_pushback=False), vrow(JUDGE, 2, "wrong_approach", is_pushback=True)],
        [prow(0, "dont vendor")],
        text="<script>alert(1)</script> dont vendor it",
    )
    auditor = lin.verdicts + (vrow("auditor", 2, "status_update", is_pushback=False),)
    lin = Lineage(sample=lin.sample, dedup_key=lin.dedup_key, verdicts=auditor, pairs=lin.pairs)
    html = render_lineage_detail(lin, {})

    for token in ("stage-detector", "stage-judge", "stage-auditor", "stage-refiner", "stage-golden"):
        assert token in html
    assert "flipped across versions" in html  # judge side changed v1 -> v2
    assert "disagree with judge" in html  # auditor said noise, final judge pushback
    assert "wrong_approach" in html
    assert "<mark>dont vendor</mark>" in html  # complaint_verbatim highlighted in the original
    assert "not in golden set" in html
    assert "&lt;script&gt;alert(1)" in html and "<script>alert(1)" not in html
    assert 'class="evidence"' not in html and 'class="pane"' not in html  # unenriched pair, no diff section


def test_render_lineage_detail_renders_the_full_untruncated_diff() -> None:
    long_call = "danger(" + "x" * 900 + ")"
    pair = RefinedPairRow(
        pair_index=0, action="vendored the dep", complaint_verbatim="dont vendor", complaint="c0",
        prompt_version=1, model="sonnet",
        evidence=EvidenceRow(
            file_path="/repo/a.py",
            incorrect=(f"if x < 1:\n    {long_call}", "safe()"),
            correct=("safe()", "safer()"),
            note="faults the danger call",
            source="git",
        ),
    )
    html = render_lineage_detail(lineage([vrow(JUDGE, 1, "wrong_approach", is_pushback=True)], [pair]), {})
    assert '<div class="del">if x &lt; 1:</div>' in html  # escaped, split per line
    assert "x" * 900 in html  # full content, no truncation
    assert html.count('class="pane"') == 2  # incorrect and correct panes
    assert '<div class="plabel">incorrect</div>' in html and '<div class="plabel">correct</div>' in html
    assert '<div class="ins">safer()</div>' in html
    assert '<span class="chip">/repo/a.py</span>' in html
    assert '<span class="chip chip-git">git</span>' in html


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
    pair = RefinedPair(action="vendored it", complaint_verbatim="dont vendor it", complaint="do not vendor")
    await store.record_refinement("k1", Refinement(pairs=[pair]), prompt_version=1, model="sonnet")


async def enrich_k1(
    store: FeedbackStore, evidence: CodeEvidence, *, pair_index: int = 0, source: str | None = "git"
) -> None:
    await store.record_evidence(
        "k1", evidence, refine_version=1, refine_model="sonnet", pair_index=pair_index,
        enrich_version=1, enrich_model="haiku", extractor_version=1, source=source,
    )


async def client(store: FeedbackStore) -> httpx.AsyncClient:
    summary = await build_summary([Sample.from_row(row) for row in await store.candidates()], use_llm=False, model="m")
    transport = httpx.ASGITransport(app=build_app(store, summary=summary))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_api_pairs_returns_atomic_rows(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        body = (await http.get("/api/pairs")).json()
    assert len(body["pairs"]) == 1
    pair = body["pairs"][0]
    assert pair["dedup_key"] == "k1" and pair["category"] == "wrong_approach"
    assert pair["complaint"] == "do not vendor" and pair["project"] == "proj"
    assert pair["complaint_verbatim"] == "dont vendor it"
    assert pair["evidence"] is None  # unenriched pair, card unchanged
    assert pair["language"] is None  # no evidence, no language facet value


async def test_api_pairs_carries_code_evidence(store: FeedbackStore) -> None:
    await seed(store)
    await enrich_k1(
        store,
        CodeEvidence(
            kind="code", file_path="/repo/a.py",
            incorrect_edit=EditSide(old="x = eval(s)", new="y = eval(t)"),
            correct_edit=EditSide(old="y = eval(t)", new="y = json.loads(t)"),
            note="faults the eval",
        ),
    )
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
    await enrich_k1(
        store,
        CodeEvidence(
            kind="code", file_path="/repo/a.py",
            incorrect_edit=EditSide(old="a" * 400, new="b"),
            correct_edit=None,
            note="long edit",
        ),
        source=None,
    )
    async with await client(store) as http:
        evidence = (await http.get("/api/pairs")).json()["pairs"][0]["evidence"]
    assert evidence["incorrect"]["old"] == "a" * 280 + "…"
    assert evidence["correct"] is None and evidence["source"] is None


async def test_api_pairs_no_code_sentinel_keeps_evidence_none(store: FeedbackStore) -> None:
    await seed(store)
    await enrich_k1(store, CodeEvidence(kind="no_code", note="not about code"), pair_index=-1, source=None)
    async with await client(store) as http:
        pair = (await http.get("/api/pairs")).json()["pairs"][0]
    assert pair["evidence"] is None
    assert pair["complaint_verbatim"] == "dont vendor it"  # the card payload is otherwise unchanged


async def test_api_lineage_shows_the_full_diff(store: FeedbackStore) -> None:
    await seed(store)
    long_new = "json.loads(" + "x" * 900 + ")"
    await enrich_k1(
        store,
        CodeEvidence(
            kind="code", file_path="/repo/a.py",
            incorrect_edit=EditSide(old="bad < worse\nstill bad", new=long_new),
            correct_edit=None,
            note="grounds the complaint",
        ),
        source=None,
    )
    async with await client(store) as http:
        html = (await http.get("/api/lineage/k1")).json()["detail_html"]
    assert '<div class="del">bad &lt; worse</div><div class="del">still bad</div>' in html
    assert "x" * 900 in html  # the full diff, untruncated
    assert html.count('class="pane"') == 1  # no correction, incorrect pane only
    assert "chip-git" not in html


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
    html = ok.json()["detail_html"]
    assert "stage-refiner" in html and "turn-trigger" in html and "do not vendor" in html
    assert missing.status_code == 404


async def test_api_stats_shape(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        stats = (await http.get("/api/stats")).json()
    assert stats["narrative"] is None
    assert stats["pipeline"]["refined"] == 1 and stats["pipeline"]["accepted"] == 2
    assert stats["pipeline"]["noise_judged"] == 1 and stats["pipeline"]["total_pairs"] == 1
    assert stats["pipeline"]["by_category_kind"] == {"wrong_approach": {"transcript_message": 2}}
    assert stats["corpus"]["total"] == 3


async def test_root_serves_shell(store: FeedbackStore) -> None:
    await seed(store)
    async with await client(store) as http:
        page = await http.get("/")
    assert page.status_code == 200
    for node in ('id="filters"', 'id="list"', 'id="search"', 'id="active"', 'id="detail"',
                 'id="backdrop"', 'id="stats-toggle"'):
        assert node in page.text
    for token in ("GROUPS", "renderFacets", "chipsHtml", "matchRow", "'Language'", "has code"):
        assert token in page.text  # faceted sidebar + contextual language/evidence facets
    for token in ("evidenceHtml", "details.diff", "chip-git", ".pane .del", ".pane .ins"):
        assert token in page.text  # evidence diff surfaces unchanged
