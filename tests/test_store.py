from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cc_pushback.detectors import detect
from tests.builders import assistant_tool_use, denial_result, interrupt_result, parse, user_text

if TYPE_CHECKING:
    from cc_pushback.models import FeedbackCandidate
    from cc_pushback.store import FeedbackStore

FILE = "/repo/projects/session.jsonl"


def sample_candidates() -> list[FeedbackCandidate]:
    events = parse(
        [
            assistant_tool_use("t1", "Write", {"file_path": "/a.py"}),
            denial_result("t1", said="don't do that"),
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
            user_text("run the tests instead, not the build"),
        ]
    )
    return detect(Path(FILE), events)


@pytest.mark.integration
def test_record_file_scan_is_idempotent(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    assert len(candidates) >= 2
    first = store.record_file_scan(FILE, 1.0, candidates)
    second = store.record_file_scan(FILE, 2.0, candidates)
    assert first == len(candidates)
    assert second == 0
    assert store.stats().total == len(candidates)


@pytest.mark.integration
def test_record_file_scan_records_mtime(store: FeedbackStore) -> None:
    store.record_file_scan(FILE, 7.0, sample_candidates())
    assert store.file_mtimes() == {FILE: 7.0}


@pytest.mark.integration
def test_record_file_scan_is_atomic_on_failure(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path: str, mtime: float) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store.store, "record_file", boom)
    with pytest.raises(RuntimeError):
        store.record_file_scan(FILE, 1.0, sample_candidates())
    assert store.stats().total == 0
    assert store.file_mtimes() == {}


@pytest.mark.integration
def test_stats_counts_by_source_kind(store: FeedbackStore) -> None:
    store.record_file_scan(FILE, 1.0, sample_candidates())
    by_source = store.stats().by_source
    assert by_source.get("interrupt_rejection", 0) >= 2
    assert by_source.get("transcript_message", 0) >= 1


@pytest.mark.integration
def test_events_returns_full_rows_newest_first(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    store.record_file_scan(FILE, 1.0, candidates)
    rows = store.events()
    assert len(rows) == len(candidates)
    assert set(rows[0]) == {
        "id",
        "source_kind",
        "occurred_at",
        "text",
        "payload_json",
        "context_json",
        "origin_path",
        "session_id",
    }
    assert all(row["context_json"] for row in rows)
    assert [str(row["occurred_at"]) for row in rows] == sorted(
        (str(row["occurred_at"]) for row in rows), reverse=True
    )
