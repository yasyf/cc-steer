from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from cc_transcript.corrections import Correction, CorrectionLog
from cc_transcript.ids import EventUuid, SessionId
from cc_transcript.mining import FEEDBACK_DDL as BASE_FEEDBACK_DDL
from cc_transcript.mining import DedupKey
from cc_transcript.store import FileStateStore

from cc_steer.detectors import detect
from cc_steer.refine import RefinedPair, Refinement
from cc_steer.rendering import has_substantive_content, messages
from cc_steer.store import ACCRUED_EMPTY_REASON, FEEDBACK_DDL, TRIAGE_DDL, FeedbackStore
from cc_steer.triage import JUDGE, Verdict
from tests.builders import assistant_tool_use, denial_result, interrupt_result, parse, user_text

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cc_steer.models import FeedbackCandidate

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"

# The expected TRIAGE_DDL, frozen verbatim. TRIAGE_DDL is composed from the judge
# package's verdicts_ddl() (pinned to cc-steer's column names) plus
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
  is_steering INTEGER NOT NULL,
  what_claude_did TEXT NOT NULL,
  confidence REAL NOT NULL,
  rationale TEXT NOT NULL,
  canonical_key TEXT,
  fidelity TEXT NOT NULL CHECK(fidelity IN ('full','summary')),
  judged_at TEXT NOT NULL,
  UNIQUE(dedup_key, role, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_triage_dedup ON triage(dedup_key);
DROP VIEW IF EXISTS training_pairs;
DROP VIEW IF EXISTS latest_judge;
CREATE VIEW latest_judge AS
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge'
) WHERE rn = 1;
DROP VIEW IF EXISTS latest_auditor;
CREATE VIEW latest_auditor AS
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'auditor'
) WHERE rn = 1;
DROP VIEW IF EXISTS accepted_steering;
CREATE VIEW accepted_steering AS
SELECT
  e.id AS event_id,
  e.dedup_key,
  e.source_kind,
  e.text,
  e.context_json,
  e.payload_json,
  t.category,
  t.what_claude_did,
  e.origin_path
FROM feedback_events e
JOIN latest_judge t ON t.dedup_key = e.dedup_key
WHERE t.is_steering = 1 AND e.quarantined_reason IS NULL;
"""


@pytest.mark.unit
def test_composed_triage_ddl_matches_original_literal() -> None:
    assert TRIAGE_DDL == ORIGINAL_TRIAGE_DDL


@pytest.mark.unit
def test_feedback_ddl_extends_the_platform_table_with_steer_columns() -> None:
    assert "origin_path TEXT" not in BASE_FEEDBACK_DDL
    assert "quarantined_reason TEXT" not in BASE_FEEDBACK_DDL
    assert "origin_path TEXT" in FEEDBACK_DDL  # the .replace() anchor matched
    assert "quarantined_reason TEXT" in FEEDBACK_DDL


async def test_open_extends_an_existing_database_and_sets_busy_timeout(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    legacy_ddl = BASE_FEEDBACK_DDL.replace(
        "  ingested_at TEXT NOT NULL\n",
        "  ingested_at TEXT NOT NULL,\n  origin_path TEXT\n",
    )
    async with await FileStateStore.open(database, extra_schema=legacy_ddl):
        pass
    async with await FeedbackStore.open(database) as upgraded:
        columns = {
            str(row["name"]) async for row in await upgraded.store.conn.execute("PRAGMA table_info(feedback_events)")
        }
        timeout = await (await upgraded.store.conn.execute("PRAGMA busy_timeout")).fetchone()
    assert "quarantined_reason" in columns
    assert timeout is not None and timeout[0] == 2_000


async def seeded_keys(store: FeedbackStore) -> list[DedupKey]:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    rows = await store.unjudged(role=JUDGE, prompt_version=1)
    return [DedupKey(str(row["dedup_key"])) for row in rows]


def verdict(category: str, *, confidence: float = 0.9) -> Verdict:
    return Verdict.model_validate(
        {"category": category, "what_claude_did": "ran a tool", "confidence": confidence, "rationale": "r"}
    )


def refinement(*directions: str) -> Refinement:
    return Refinement(
        pairs=[
            RefinedPair(action="ran a tool", direction_verbatim=text, direction=f"distilled: {text}")
            for text in (directions or ("stop that",))
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
async def test_record_file_scan_quarantines_empty_context_at_accrual(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    substantive = next(c for c in candidates if has_substantive_content(messages(c.window.before)))
    other = next(c for c in candidates if c.dedup_key != substantive.dedup_key)
    empty = replace(other, window=replace(other.window, before=()))
    await store.record_file_scan(FILE, 1.0, [substantive, empty])

    reasons = {
        str(row["dedup_key"]): row["quarantined_reason"]
        async for row in await store.store.conn.execute("SELECT dedup_key, quarantined_reason FROM feedback_events")
    }
    assert reasons[str(empty.dedup_key)] == ACCRUED_EMPTY_REASON
    assert reasons[str(substantive.dedup_key)] is None
    assert await store.quarantined_keys() == {str(empty.dedup_key)}

    active = {str(row["dedup_key"]) for row in await store.unjudged(role=JUDGE, prompt_version=1)}
    assert str(empty.dedup_key) not in active
    assert str(substantive.dedup_key) in active


@pytest.mark.integration
async def test_record_file_scan_does_not_quarantine_an_existing_healthy_duplicate(store: FeedbackStore) -> None:
    substantive = next(
        candidate
        for candidate in sample_candidates()
        if has_substantive_content(messages(candidate.window.before))
    )
    await store.record_file_scan(FILE, 1.0, [substantive])
    existing = await (
        await store.store.conn.execute(
            "SELECT context_json FROM feedback_events WHERE dedup_key = ?", (substantive.dedup_key,)
        )
    ).fetchone()
    empty = replace(substantive, window=replace(substantive.window, before=()))

    assert await store.record_file_scan(FILE, 2.0, [empty]) == 0
    row = await (
        await store.store.conn.execute(
            "SELECT context_json, quarantined_reason FROM feedback_events WHERE dedup_key = ?",
            (substantive.dedup_key,),
        )
    ).fetchone()
    assert existing is not None and row is not None
    assert row["context_json"] == existing["context_json"]
    assert row["quarantined_reason"] is None


@pytest.mark.integration
async def test_record_file_scan_does_not_heal_an_existing_empty_duplicate(store: FeedbackStore) -> None:
    substantive = next(
        candidate
        for candidate in sample_candidates()
        if has_substantive_content(messages(candidate.window.before))
    )
    empty = replace(substantive, window=replace(substantive.window, before=()))
    await store.record_file_scan(FILE, 1.0, [empty])
    existing = await (
        await store.store.conn.execute(
            "SELECT context_json FROM feedback_events WHERE dedup_key = ?", (substantive.dedup_key,)
        )
    ).fetchone()

    assert await store.record_file_scan(FILE, 2.0, [substantive]) == 0
    row = await (
        await store.store.conn.execute(
            "SELECT context_json, quarantined_reason FROM feedback_events WHERE dedup_key = ?",
            (substantive.dedup_key,),
        )
    ).fetchone()
    assert existing is not None and row is not None
    assert row["context_json"] == existing["context_json"]
    assert row["quarantined_reason"] == ACCRUED_EMPTY_REASON


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
async def test_unjudged_honors_version_and_limit(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(
        keys[0], verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    remaining = await store.unjudged(role=JUDGE, prompt_version=1)
    assert keys[0] not in {str(row["dedup_key"]) for row in remaining}
    assert len(remaining) == len(keys) - 1
    assert len(await store.unjudged(role=JUDGE, prompt_version=2)) == len(keys)
    assert len(await store.unjudged(role=JUDGE, prompt_version=1, limit=1)) == 1


@pytest.mark.integration
async def test_unjudged_applies_quarantine_and_limit_in_sql(store: FeedbackStore) -> None:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    first = await (
        await store.store.conn.execute("SELECT dedup_key FROM feedback_events ORDER BY id LIMIT 1")
    ).fetchone()
    assert first is not None
    await store.store.conn.execute(
        "UPDATE feedback_events SET quarantined_reason = ? WHERE dedup_key = ?",
        (ACCRUED_EMPTY_REASON, first["dedup_key"]),
    )
    statements: list[str] = []
    await store.store.conn.set_trace_callback(statements.append)
    rows = await store.unjudged(role=JUDGE, prompt_version=1, limit=1)
    await store.store.conn.set_trace_callback(None)

    assert len(rows) == 1
    assert rows[0]["dedup_key"] != first["dedup_key"]
    [query] = [
        " ".join(statement.split())
        for statement in statements
        if "LEFT JOIN triage" in statement and "FROM feedback_events e" in statement
    ]
    assert "e.quarantined_reason IS NULL" in query
    assert query.endswith("LIMIT 1")


@pytest.mark.integration
async def test_verdict_identity_ignores_model(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    # model is provenance, not identity: a second verdict at the same
    # (dedup_key, role, prompt_version) upserts in place instead of adding a row.
    await store.record_verdict(key, verdict("new_task"), role=JUDGE, prompt_version=1, model="haiku", fidelity="full")
    cur = await store.store.conn.execute("SELECT model, category FROM triage WHERE dedup_key = ?", (key,))
    assert [(row["model"], row["category"]) async for row in cur] == [("sonnet", "wrong_approach")]
    assert key not in {str(row["dedup_key"]) for row in await store.unjudged(role=JUDGE, prompt_version=1)}


@pytest.mark.integration
async def test_accepted_steering_filters_noise_and_latest_judge_wins(store: FeedbackStore) -> None:
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
async def test_auditor_only_event_is_not_accepted_steering(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role="auditor", prompt_version=1, model="opus", fidelity="full"
    )
    assert await store.unrefined(prompt_version=1, model="sonnet") == []
    assert await store.pairs() == []


@pytest.mark.integration
async def test_record_refinement_is_idempotent(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    rows = await store.pairs()
    assert len(rows) == 2
    assert [int(row["pair_index"]) for row in rows] == [0, 1]
    assert {str(row["direction_verbatim"]) for row in rows} == {"use a generator", "stop hardcoding"}


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
    assert accepted["is_steering"] == 1 and accepted["judge_version"] == 2  # latest judge (v2) wins
    assert accepted["pair_count"] == 2
    assert accepted["flipped"] == 1  # noise (v1) -> steering (v2)
    assert accepted["auditor_is_steering"] == 0  # auditor disagreed, called it noise

    noise = rows[keys[1]]
    assert noise["is_steering"] == 0 and noise["pair_count"] is None
    assert noise["flipped"] == 0 and noise["auditor_is_steering"] is None

    for row in (rows[key] for key in keys[2:]):
        assert row["is_steering"] is None and row["pair_count"] is None and row["flipped"] == 0


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
    assert {str(p["direction_verbatim"]) for p in lineage["pairs"]} == {"x", "y"}
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
    assert [str(row["direction_verbatim"]) for row in rows] == ["c"]
    assert rows[0]["prompt_version"] == 2
    assert rows[0]["category"] == "wrong_approach"
    assert rows[0]["action"] == "ran a tool"


@pytest.mark.integration
async def test_refined_pairs_excludes_events_the_latest_judge_now_rejects(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_refinement(key, refinement("a", "b"), prompt_version=1, model="sonnet")
    assert len(await store.pairs()) == 2  # accepted at v1; its v1 refinement is part of the deliverable

    await store.record_verdict(
        key, verdict("status_update"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    assert await store.pairs() == []  # latest judge (v2) flipped it to noise; the stale v1 pairs drop out


async def seeded_refined_pair(store: FeedbackStore) -> DedupKey:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(
        key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    return key


def correction_for(row: Mapping[str, object]) -> Correction:
    return Correction(
        ts_ms=1,
        session_id=SessionId(str(row["session_id"])),
        source="cc-steer",
        anchor_uuid=EventUuid(str(row["event_uuid"])),
        incorrect_digest=None,
        incorrect_file="/a.py",
        incorrect_old="bad",
        incorrect_new="worse",
        correction_origin="session",
        correction_old="worse",
        correction_new="good",
        overlap=1.0,
    )


@pytest.mark.integration
async def test_unenriched_surfaces_refined_pairs_lacking_a_ledger_correction(store: FeedbackStore) -> None:
    await seeded_refined_pair(store)
    rows = await store.unenriched(CorrectionLog.open())
    assert [int(str(row["pair_index"])) for row in rows] == [0, 1]
    assert set(rows[0]) == {
        "dedup_key",
        "refine_version",
        "refine_model",
        "pair_index",
        "action",
        "direction",
        "direction_verbatim",
        "source_kind",
        "session_id",
        "event_uuid",
        "origin_path",
    }
    assert len(await store.unenriched(CorrectionLog.open(), limit=1)) == 1


@pytest.mark.integration
async def test_a_ledger_correction_settles_every_pair_sharing_the_anchor(store: FeedbackStore) -> None:
    await seeded_refined_pair(store)
    log = CorrectionLog.open()
    rows = await store.unenriched(log)
    assert len(rows) == 2  # both pairs share one anchor

    log.append(correction_for(rows[0]))
    # The anchor now carries a correction, so both of its pairs settle together.
    assert await store.unenriched(CorrectionLog.open()) == []


@pytest.mark.integration
async def test_unenriched_excludes_anchorless_pairs(store: FeedbackStore) -> None:
    key = await seeded_refined_pair(store)
    await store.store.conn.execute("UPDATE feedback_events SET session_id = NULL WHERE dedup_key = ?", (key,))
    assert await store.unenriched(CorrectionLog.open()) == []  # no anchor, nothing the extractor can ground


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
