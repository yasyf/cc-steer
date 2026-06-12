from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest
from cc_transcript.context import ContextWindow
from cc_transcript.ids import EventRef, EventUuid, SessionId

from cc_pushback.migrate import migrate_corpus, window_from_snapshot
from cc_pushback.store import FeedbackStore
from cc_pushback.triage import JUDGE, Verdict

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

# The pre-2.0 schema, frozen verbatim from the legacy mining store and triage layer.
LEGACY_SCHEMA = """
CREATE TABLE files (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL
);
CREATE TABLE feedback_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL UNIQUE,
  source_kind TEXT NOT NULL,
  session_id TEXT,
  origin_path TEXT,
  origin_uuid TEXT,
  occurred_at TEXT NOT NULL,
  text TEXT NOT NULL,
  payload_json TEXT,
  context_json TEXT NOT NULL,
  cc_version TEXT,
  ingested_at TEXT NOT NULL
);
CREATE TABLE triage (
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
  judged_at TEXT NOT NULL,
  UNIQUE(dedup_key, role, prompt_version, model)
);
"""

# A real old-shape ContextSnapshot document, tool_inputs included — and one legacy
# turn (k2's) written before tool_inputs existed, exercising the .get() default.
LEGACY_SNAPSHOT = json.dumps(
    {
        "before": [{"role": "user", "text": "please clean the build dir", "tool_calls": [], "tool_inputs": []}],
        "trigger": {
            "role": "assistant",
            "text": "cleaning now",
            "tool_calls": ["Bash", "Edit"],
            "tool_inputs": ["rm -rf build"],
        },
        "after": [{"role": "tool", "text": "exit status 1", "tool_calls": [], "tool_inputs": []}],
    }
)
LEGACY_SNAPSHOT_NO_INPUTS = json.dumps(
    {
        "before": [],
        "trigger": {"role": "assistant", "text": "ran it", "tool_calls": ["Bash"]},
        "after": [],
    }
)


def seed_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    rows = [
        ("k1", "sess-1", "/old/projects/s.jsonl", "uuid-7", LEGACY_SNAPSHOT),
        ("k2", "sess-1", "/old/projects/s.jsonl", None, LEGACY_SNAPSHOT_NO_INPUTS),
    ]
    conn.executemany(
        "INSERT INTO feedback_events (dedup_key, source_kind, session_id, origin_path, origin_uuid, "
        "occurred_at, text, payload_json, context_json, cc_version, ingested_at) "
        "VALUES (?, 'transcript_message', ?, ?, ?, '2026-01-01T00:00:00', 'no, stop', NULL, ?, "
        "'1.2.3', '2026-01-02T00:00:00')",
        rows,
    )
    conn.execute(
        "INSERT INTO triage (dedup_key, role, prompt_version, model, category, is_pushback, "
        "what_claude_did, confidence, rationale, judged_at) "
        "VALUES ('k1', 'judge', 3, 'sonnet', 'wrong_approach', 1, 'did x', 0.9, 'r', '2026-01-03T00:00:00')"
    )
    conn.commit()
    conn.close()


@pytest.mark.unit
def test_window_from_snapshot_builds_a_migrated_summary_window() -> None:
    window = window_from_snapshot(LEGACY_SNAPSHOT, EventRef(SessionId("sess-1"), EventUuid("uuid-7")))
    assert (window.fidelity, window.origin) == ("summary", "migrated")
    assert window.anchor == EventRef(SessionId("sess-1"), EventUuid("uuid-7"))
    assert [ref.preview for ref in window.before] == ["user: please clean the build dir"]
    assert window.trigger is not None
    assert window.trigger.role == "assistant"
    assert window.trigger.preview == "assistant: cleaning now\n  Bash(rm -rf build)\n  Edit()"
    assert window.trigger.refs == () and window.trigger.tool_digests == ()
    assert [ref.role for ref in window.after] == ["assistant"]  # legacy 'tool' role folds into assistant
    assert [ref.preview for ref in window.after] == ["tool: exit status 1"]


@pytest.mark.integration
async def test_migrate_corpus_converts_legacy_rows_once(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    seed_legacy_db(db)
    async with await FeedbackStore.open(db) as store:
        report = await migrate_corpus(store)
        assert (report.migrated, report.skipped) == (2, 0)

        cur = await store.store.conn.execute("SELECT dedup_key, event_uuid, context_json FROM feedback_events")
        rows = {str(row["dedup_key"]): dict(row) async for row in cur}
        anchored = ContextWindow.from_json(str(rows["k1"]["context_json"]))
        assert anchored.anchor == EventRef(SessionId("sess-1"), EventUuid("uuid-7"))
        assert rows["k1"]["event_uuid"] == "uuid-7"  # backfilled from origin_uuid
        anchorless = ContextWindow.from_json(str(rows["k2"]["context_json"]))
        assert anchorless.anchor is None  # origin_uuid was NULL
        assert anchorless.trigger is not None
        assert anchorless.trigger.preview == "assistant: ran it\n  Bash()"
        assert all(
            ContextWindow.from_json(str(row["context_json"])).origin == "migrated" for row in rows.values()
        )

        again = await migrate_corpus(store)
        assert (again.migrated, again.skipped) == (0, 2)


@pytest.mark.integration
async def test_migrate_corpus_makes_the_legacy_db_judgeable(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    seed_legacy_db(db)
    async with await FeedbackStore.open(db) as store:
        await migrate_corpus(store)
        cur = await store.store.conn.execute("SELECT fidelity FROM triage")
        assert [str(row["fidelity"]) async for row in cur] == ["summary"]  # legacy verdicts default to summary

        rows = await store.unjudged(role=JUDGE, prompt_version=3, model="sonnet", refresh_summary=True)
        assert {str(row["dedup_key"]) for row in rows} == {"k1", "k2"}  # k1 re-yields via refresh_summary

        verdict = Verdict(category="wrong_approach", what_claude_did="did x", confidence=0.9, rationale="r")
        await store.record_verdict("k1", verdict, role=JUDGE, prompt_version=3, model="sonnet", fidelity="full")
        cur = await store.store.conn.execute("SELECT fidelity FROM triage WHERE dedup_key = 'k1'")
        assert [str(row["fidelity"]) async for row in cur] == ["full"]  # full replaces the summary verdict


@pytest.mark.integration
async def test_migrate_corpus_is_a_noop_on_a_fresh_db(store: FeedbackStore) -> None:
    report = await migrate_corpus(store)
    assert (report.migrated, report.skipped) == (0, 0)