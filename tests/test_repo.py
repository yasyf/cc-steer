from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from cc_pushback.models import ContextSnapshot, ContextTurn, DedupKey, FeedbackCandidate

if TYPE_CHECKING:
    from cc_pushback.models import SourceKind
    from cc_pushback.repo import Repository

EMPTY_CONTEXT = ContextSnapshot(before=(ContextTurn(role="user", text="ctx"),), trigger=None, after=())


def candidate(key: str, *, source_kind: SourceKind = "transcript_message") -> FeedbackCandidate:
    return FeedbackCandidate(
        dedup_key=DedupKey(key),
        source_kind=source_kind,
        occurred_at=datetime(2026, 6, 1, tzinfo=UTC),
        text=f"text-{key}",
        context=EMPTY_CONTEXT,
    )


def test_record_file_scan_counts_new_rows(feedback_db: Repository) -> None:
    new = feedback_db.record_file_scan("/t.jsonl", 1.0, [candidate("a"), candidate("b")])

    assert new == 2
    assert feedback_db.file_mtimes() == {"/t.jsonl": 1.0}


def test_record_file_scan_is_idempotent(feedback_db: Repository) -> None:
    cands = [candidate("a"), candidate("b")]
    feedback_db.record_file_scan("/t.jsonl", 1.0, cands)

    assert feedback_db.record_file_scan("/t.jsonl", 2.0, cands) == 0
    assert feedback_db.file_mtimes() == {"/t.jsonl": 2.0}


def test_record_file_scan_inserts_only_new(feedback_db: Repository) -> None:
    feedback_db.record_file_scan("/t.jsonl", 1.0, [candidate("a")])

    assert feedback_db.record_file_scan("/t.jsonl", 2.0, [candidate("a"), candidate("b")]) == 1


def test_failed_write_rolls_back_inserts_and_file(
    feedback_db: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: str, mtime: float) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(feedback_db.store, "record_file", boom)

    with pytest.raises(sqlite3.OperationalError):
        feedback_db.record_file_scan("/t.jsonl", 1.0, [candidate("a")])

    assert feedback_db.file_mtimes() == {}
    assert feedback_db.recent() == []


def test_advance_github_cursor_upserts_and_inserts(feedback_db: Repository) -> None:
    new = feedback_db.advance_github_cursor("github:o/r", "2026-06-01", [candidate("g", source_kind="github_review")])

    assert new == 1
    assert feedback_db.cursor_for("github:o/r") == "2026-06-01"


def test_advance_github_cursor_idempotent(feedback_db: Repository) -> None:
    cands = [candidate("g", source_kind="github_review")]
    feedback_db.advance_github_cursor("github:o/r", "2026-06-01", cands)

    assert feedback_db.advance_github_cursor("github:o/r", "2026-06-02", cands) == 0
    assert feedback_db.cursor_for("github:o/r") == "2026-06-02"


def test_cursor_for_unseen_returns_none(feedback_db: Repository) -> None:
    assert feedback_db.cursor_for("github:nope") is None


def test_stats_aggregates_by_source(feedback_db: Repository) -> None:
    feedback_db.record_file_scan(
        "/t.jsonl",
        1.0,
        [candidate("a"), candidate("b"), candidate("p", source_kind="plan_review")],
    )

    stats = feedback_db.stats()

    assert stats.total == 3
    assert stats.files == 1
    assert stats.by_source == {"plan_review": 1, "transcript_message": 2}


def test_recent_filters_and_limits(feedback_db: Repository) -> None:
    feedback_db.record_file_scan(
        "/t.jsonl", 1.0, [candidate("a"), candidate("p", source_kind="plan_review")]
    )

    rows = feedback_db.recent(source_kind="plan_review")

    assert [row["text"] for row in rows] == ["text-p"]


def test_recent_limit_caps_rows(feedback_db: Repository) -> None:
    feedback_db.record_file_scan("/t.jsonl", 1.0, [candidate(str(i)) for i in range(5)])

    assert len(feedback_db.recent(limit=3)) == 3
