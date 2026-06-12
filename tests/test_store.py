from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from cc_transcript.mining import FEEDBACK_DDL as BASE_FEEDBACK_DDL
from cc_transcript.mining import DedupKey

from cc_pushback.detectors import detect
from cc_pushback.enrich import CodeEvidence, EditSide
from cc_pushback.refine import RefinedPair, Refinement
from cc_pushback.store import FEEDBACK_DDL, TRIAGE_DDL
from cc_pushback.triage import JUDGE, Verdict
from tests.builders import assistant_tool_use, denial_result, interrupt_result, parse, user_text

if TYPE_CHECKING:
    from cc_pushback.models import FeedbackCandidate
    from cc_pushback.store import FeedbackStore

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"

# The expected TRIAGE_DDL, frozen verbatim. TRIAGE_DDL is composed from the judge
# package's verdicts_ddl() (pinned to cc-pushback's column names) plus
# TRIAGE_VIEWS_DDL; this byte-for-byte equality pins the composed schema, fidelity
# column included.
ORIGINAL_TRIAGE_DDL = """
CREATE TABLE IF NOT EXISTS triage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  role TEXT NOT NULL,
  prompt_version INTEGER NOT NULL,
  model TEXT NOT NULL,
  category TEXT NOT NULL,
  is_pushback INTEGER NOT NULL,
  what_claude_did TEXT NOT NULL,
  confidence REAL NOT NULL,
  rationale TEXT NOT NULL,
  fidelity TEXT NOT NULL CHECK(fidelity IN ('full','summary')),
  judged_at TEXT NOT NULL,
  UNIQUE(dedup_key, role, prompt_version, model)
);
CREATE INDEX IF NOT EXISTS idx_triage_dedup ON triage(dedup_key);
DROP VIEW IF EXISTS training_pairs;
DROP VIEW IF EXISTS accepted_pushback;
CREATE VIEW accepted_pushback AS
WITH latest AS (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge'
)
SELECT
  e.id AS event_id,
  e.dedup_key,
  e.source_kind,
  e.text,
  e.context_json,
  t.category,
  t.what_claude_did,
  e.origin_path
FROM feedback_events e
JOIN latest t ON t.dedup_key = e.dedup_key AND t.rn = 1
WHERE t.is_pushback = 1;
"""


@pytest.mark.unit
def test_composed_triage_ddl_matches_original_literal() -> None:
    assert TRIAGE_DDL == ORIGINAL_TRIAGE_DDL


@pytest.mark.unit
def test_feedback_ddl_extends_the_platform_table_with_origin_path() -> None:
    assert "origin_path TEXT" not in BASE_FEEDBACK_DDL
    assert "origin_path TEXT" in FEEDBACK_DDL  # the .replace() anchor matched


async def seeded_keys(store: FeedbackStore) -> list[DedupKey]:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    rows = await store.unjudged(role=JUDGE, prompt_version=1, model="sonnet")
    return [DedupKey(str(row["dedup_key"])) for row in rows]


def verdict(category: str, *, confidence: float = 0.9) -> Verdict:
    return Verdict.model_validate(
        {"category": category, "what_claude_did": "ran a tool", "confidence": confidence, "rationale": "r"}
    )


def refinement(*complaints: str) -> Refinement:
    return Refinement(
        pairs=[
            RefinedPair(action="ran a tool", complaint_verbatim=text, complaint=f"distilled: {text}")
            for text in (complaints or ("stop that",))
        ]
    )


def sample_candidates() -> list[FeedbackCandidate]:
    events = parse(
        [
            assistant_tool_use("t1", "Write", {"file_path": "/a.py", "content": "x = 1"}),
            denial_result("t1", said="don't do that"),
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
            user_text("run the tests instead, not the build"),
        ]
    )
    return detect(events)


@pytest.mark.integration
async def test_record_file_scan_is_idempotent(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    assert len(candidates) >= 2
    first = await store.record_file_scan(FILE, 1.0, candidates)
    second = await store.record_file_scan(FILE, 2.0, candidates)
    assert first == len(candidates)
    assert second == 0
    assert (await store.stats()).total == len(candidates)


@pytest.mark.integration
async def test_record_file_scan_records_mtime(store: FeedbackStore) -> None:
    await store.record_file_scan(FILE, 7.0, sample_candidates())
    assert await store.file_mtimes() == {FILE: 7.0}


@pytest.mark.integration
async def test_record_file_scan_is_atomic_on_failure(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(path: str, mtime: float) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store.store, "record_file", boom)
    with pytest.raises(RuntimeError):
        await store.record_file_scan(FILE, 1.0, sample_candidates())
    assert (await store.stats()).total == 0
    assert await store.file_mtimes() == {}


@pytest.mark.integration
async def test_stats_counts_by_source_kind(store: FeedbackStore) -> None:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    by_source = (await store.stats()).by_source
    assert by_source.get("interrupt_rejection", 0) >= 2
    assert by_source.get("transcript_message", 0) >= 1


@pytest.mark.integration
async def test_events_returns_full_rows_newest_first(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    await store.record_file_scan(FILE, 1.0, candidates)
    rows = await store.events()
    assert len(rows) == len(candidates)
    assert set(rows[0]) == {
        "id",
        "source_kind",
        "occurred_at",
        "text",
        "payload_json",
        "context_json",
        "event_uuid",
        "session_id",
    }
    assert all(row["context_json"] for row in rows)
    assert [str(row["occurred_at"]) for row in rows] == sorted((str(row["occurred_at"]) for row in rows), reverse=True)


@pytest.mark.integration
async def test_record_file_scan_stores_the_origin_path_hint(store: FeedbackStore) -> None:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    cur = await store.store.conn.execute("SELECT DISTINCT origin_path FROM feedback_events")
    assert [row["origin_path"] async for row in cur] == [FILE]


@pytest.mark.integration
async def test_full_fidelity_verdict_replaces_a_summary_one(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="summary"
    )
    await store.record_verdict(
        key, verdict("incorrect_change"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(key, verdict("new_task"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full")
    cur = await store.store.conn.execute("SELECT category, fidelity FROM triage WHERE dedup_key = ?", (key,))
    rows = [(row["category"], row["fidelity"]) async for row in cur]
    assert rows == [("incorrect_change", "full")]  # full replaced summary; the second full was a no-op


@pytest.mark.integration
async def test_record_verdict_is_idempotent(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(key, verdict("new_task"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full")
    rows = await store.judged(role=JUDGE, prompt_version=1)
    assert [str(row["category"]) for row in rows if row["dedup_key"] == key] == ["wrong_approach"]


@pytest.mark.integration
async def test_unjudged_honors_version_model_and_limit(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(
        keys[0], verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    remaining = await store.unjudged(role=JUDGE, prompt_version=1, model="sonnet")
    assert keys[0] not in {str(row["dedup_key"]) for row in remaining}
    assert len(remaining) == len(keys) - 1
    assert len(await store.unjudged(role=JUDGE, prompt_version=2, model="sonnet")) == len(keys)
    assert len(await store.unjudged(role=JUDGE, prompt_version=1, model="haiku")) == len(keys)
    assert len(await store.unjudged(role=JUDGE, prompt_version=1, model="sonnet", limit=1)) == 1


@pytest.mark.integration
async def test_accepted_pushback_filters_noise_and_latest_judge_wins(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(
        keys[0], verdict("unwanted_action"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        keys[1], verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    accepted = await store.unrefined(prompt_version=1, model="sonnet")
    assert [str(row["dedup_key"]) for row in accepted] == [keys[0]]
    await store.record_verdict(
        keys[0], verdict("operational_directive"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        keys[1], verdict("incorrect_change"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    flipped = await store.unrefined(prompt_version=1, model="sonnet")
    assert [str(row["dedup_key"]) for row in flipped] == [keys[1]]


@pytest.mark.integration
async def test_auditor_only_event_is_not_accepted_pushback(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role="auditor", prompt_version=1, model="opus", fidelity="full"
    )
    assert await store.unrefined(prompt_version=1, model="sonnet") == []
    assert await store.pairs() == []


@pytest.mark.integration
async def test_record_refinement_is_idempotent(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    rows = await store.pairs()
    assert len(rows) == 2
    assert [int(row["pair_index"]) for row in rows] == [0, 1]
    assert {str(row["complaint_verbatim"]) for row in rows} == {"use a generator", "stop hardcoding"}


@pytest.mark.integration
async def test_unrefined_honors_version_model_and_limit(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    for key in keys:
        await store.record_verdict(
            key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
        )
    await store.record_refinement(keys[0], refinement("x"), prompt_version=1, model="sonnet")
    remaining = await store.unrefined(prompt_version=1, model="sonnet")
    assert keys[0] not in {str(row["dedup_key"]) for row in remaining}
    assert len(remaining) == len(keys) - 1
    assert len(await store.unrefined(prompt_version=2, model="sonnet")) == len(keys)
    assert len(await store.unrefined(prompt_version=1, model="haiku")) == len(keys)
    assert len(await store.unrefined(prompt_version=1, model="sonnet", limit=1)) == 1


@pytest.mark.integration
async def test_candidates_reports_status_pair_count_auditor_and_flip(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    assert len(keys) >= 2
    await store.record_verdict(
        keys[0], verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        keys[0], verdict("wrong_approach"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        keys[0], verdict("status_update"), role="auditor", prompt_version=2, model="opus", fidelity="full"
    )
    await store.record_refinement(keys[0], refinement("a", "b"), prompt_version=1, model="sonnet")
    await store.record_verdict(
        keys[1], verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )

    rows = {str(row["dedup_key"]): row for row in await store.candidates()}
    assert len(rows) == len(keys)

    accepted = rows[keys[0]]
    assert accepted["is_pushback"] == 1 and accepted["judge_version"] == 2  # latest judge (v2) wins
    assert accepted["pair_count"] == 2
    assert accepted["flipped"] == 1  # noise (v1) -> pushback (v2)
    assert accepted["auditor_is_pushback"] == 0  # auditor disagreed, called it noise

    noise = rows[keys[1]]
    assert noise["is_pushback"] == 0 and noise["pair_count"] is None
    assert noise["flipped"] == 0 and noise["auditor_is_pushback"] is None

    for row in (rows[key] for key in keys[2:]):
        assert row["is_pushback"] is None and row["pair_count"] is None and row["flipped"] == 0


@pytest.mark.integration
async def test_lineage_returns_all_verdicts_and_latest_pairs(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        key, verdict("status_update"), role="auditor", prompt_version=2, model="opus", fidelity="full"
    )
    await store.record_refinement(key, refinement("x", "y"), prompt_version=1, model="sonnet")

    lineage = await store.lineage(key)
    assert str(lineage["dedup_key"]) == key
    verdicts = [(str(v["role"]), int(str(v["prompt_version"]))) for v in lineage["verdicts"]]
    assert verdicts == [("auditor", 2), ("judge", 1), ("judge", 2)]
    assert [int(str(p["pair_index"])) for p in lineage["pairs"]] == [0, 1]
    assert {str(p["complaint_verbatim"]) for p in lineage["pairs"]} == {"x", "y"}
    assert await store.lineage("nope") == {}


@pytest.mark.integration
async def test_refined_pairs_latest_generation_wins(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_refinement(key, refinement("a", "b"), prompt_version=1, model="sonnet")
    await store.record_refinement(key, refinement("c"), prompt_version=2, model="sonnet")
    rows = await store.pairs()
    assert [str(row["complaint_verbatim"]) for row in rows] == ["c"]
    assert rows[0]["prompt_version"] == 2
    assert rows[0]["category"] == "wrong_approach"
    assert rows[0]["action"] == "ran a tool"


def code_evidence(note: str = "n") -> CodeEvidence:
    return CodeEvidence(
        kind="code",
        file_path="/a.py",
        incorrect_edit=EditSide(old="bad", new="worse"),
        correct_edit=EditSide(old="worse", new="good"),
        note=note,
    )


async def seeded_refined_pair(store: FeedbackStore) -> DedupKey:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    return key


async def record_evidence(
    store: FeedbackStore, key: DedupKey, evidence: CodeEvidence, *, pair_index: int = 0, extractor_version: int = 1
) -> None:
    await store.record_evidence(
        key,
        evidence,
        refine_version=1,
        refine_model="sonnet",
        pair_index=pair_index,
        enrich_version=1,
        enrich_model="haiku",
        extractor_version=extractor_version,
        source="session" if evidence.correct_edit else None,
    )


@pytest.mark.integration
async def test_record_evidence_is_idempotent_and_round_trips(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    await record_evidence(store, key, code_evidence())
    await record_evidence(store, key, code_evidence(note="ignored duplicate"))
    cur = await store.store.conn.execute("SELECT COUNT(*) AS n FROM pair_evidence")
    assert [row["n"] async for row in cur] == [1]
    by_index = {int(str(row["pair_index"])): row for row in await store.pairs()}
    enriched = by_index[0]
    assert (enriched["evidence_kind"], enriched["evidence_file_path"], enriched["evidence_note"]) == (
        "code",
        "/a.py",
        "n",
    )
    assert (enriched["incorrect_old"], enriched["incorrect_new"]) == ("bad", "worse")
    assert (enriched["correct_old"], enriched["correct_new"]) == ("worse", "good")
    assert (enriched["evidence_source"], enriched["enrich_version"], enriched["extractor_version"]) == ("session", 1, 1)
    assert by_index[1]["evidence_kind"] is None  # the sibling pair stays unenriched


@pytest.mark.integration
async def test_no_code_sentinel_covers_every_pair_and_settles_unenriched(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    assert len(await store.unenriched(enrich_version=1, enrich_model="haiku", extractor_version=1)) == 2
    await record_evidence(store, key, CodeEvidence(kind="no_code", note="nothing here"), pair_index=-1)
    assert await store.unenriched(enrich_version=1, enrich_model="haiku", extractor_version=1) == []
    assert [(row["pair_index"], row["evidence_kind"], row["evidence_note"]) for row in await store.pairs()] == [
        (0, "no_code", "nothing here"),
        (1, "no_code", "nothing here"),
    ]


@pytest.mark.integration
async def test_unenriched_keys_on_the_full_enrich_generation(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    await record_evidence(store, key, code_evidence())
    remaining = await store.unenriched(enrich_version=1, enrich_model="haiku", extractor_version=1)
    assert [int(str(row["pair_index"])) for row in remaining] == [1]
    assert set(remaining[0]) == {
        "dedup_key",
        "refine_version",
        "refine_model",
        "pair_index",
        "action",
        "complaint",
        "complaint_verbatim",
        "source_kind",
        "session_id",
        "event_uuid",
    }
    assert len(await store.unenriched(enrich_version=2, enrich_model="haiku", extractor_version=1)) == 2
    assert len(await store.unenriched(enrich_version=1, enrich_model="opus", extractor_version=1)) == 2
    assert len(await store.unenriched(enrich_version=1, enrich_model="haiku", extractor_version=2)) == 2
    assert len(await store.unenriched(enrich_version=1, enrich_model="haiku", extractor_version=1, limit=1)) == 1


@pytest.mark.integration
async def test_pair_evidence_latest_prefers_the_newest_extractor_generation(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    await record_evidence(store, key, code_evidence(note="old extraction"), extractor_version=1)
    await record_evidence(store, key, code_evidence(note="new extraction"), extractor_version=2)
    row = {int(str(r["pair_index"])): r for r in await store.pairs()}[0]
    assert (row["evidence_note"], row["extractor_version"]) == ("new extraction", 2)


@pytest.mark.integration
async def test_exact_pair_evidence_beats_the_sentinel_within_a_generation(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    await record_evidence(store, key, code_evidence())
    await record_evidence(store, key, CodeEvidence(kind="no_code", note="expired later"), pair_index=-1)
    by_index = {int(str(row["pair_index"])): row for row in await store.pairs()}
    assert by_index[0]["evidence_kind"] == "code"
    assert (by_index[1]["evidence_kind"], by_index[1]["evidence_note"]) == ("no_code", "expired later")


@pytest.mark.integration
async def test_lineage_pairs_carry_evidence_columns(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    await record_evidence(store, key, code_evidence())
    pairs = (await store.lineage(key))["pairs"]
    assert [(p["pair_index"], p["evidence_kind"]) for p in pairs] == [(0, "code"), (1, None)]
    assert pairs[0]["evidence_source"] == "session"


@pytest.mark.integration
async def test_triage_stats_counts_by_category(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(
        keys[0], verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        keys[1], verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    stats = await store.triage_stats(prompt_version=1)
    assert (stats.total, stats.judged, stats.accepted) == (len(keys), 2, 1)
    assert stats.by_category == {"wrong_approach": 1, "status_update": 1}


@pytest.mark.integration
async def test_dedup_keys_returns_every_event_key(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    assert await store.dedup_keys() == set(keys)
