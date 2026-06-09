from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from cc_pushback.scan import scan
from tests.builders import assistant_tool_use, denial_result, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path

    from cc_pushback.store import FeedbackStore


def good_entries() -> list[dict[str, Any]]:
    return [
        assistant_tool_use("t1", "Write", {"file_path": "/a.py"}),
        denial_result("t1", said="don't do that"),
        user_text("use a frozen dataclass here instead of a dict"),
    ]


@pytest.mark.integration
def test_scan_inserts_then_is_incremental(store: FeedbackStore, tmp_path: Path) -> None:
    write_transcript(tmp_path / "proj" / "s.jsonl", good_entries())
    first = scan(store, [tmp_path])
    assert first.scanned == 1
    assert first.inserted >= 2
    assert first.skipped == ()

    second = scan(store, [tmp_path])
    assert second.scanned == 0
    assert second.inserted == 0


@pytest.mark.integration
def test_full_rescan_reparses_but_stays_idempotent(store: FeedbackStore, tmp_path: Path) -> None:
    write_transcript(tmp_path / "proj" / "s.jsonl", good_entries())
    scan(store, [tmp_path])
    total = store.stats().total
    again = scan(store, [tmp_path], full=True)
    assert again.scanned == 1
    assert again.inserted == 0
    assert store.stats().total == total


@pytest.mark.integration
def test_unparseable_transcript_is_skipped_and_left_unrecorded(store: FeedbackStore, tmp_path: Path) -> None:
    good = write_transcript(tmp_path / "proj" / "good.jsonl", good_entries())
    bad = tmp_path / "proj" / "bad.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"type": "user", "message": {"role": "user", "content": "hi"}}\n')  # valid JSON, missing uuid
    report = scan(store, [tmp_path])
    assert report.scanned == 1
    assert bad in report.skipped
    assert str(good) in store.file_mtimes()
    assert str(bad) not in store.file_mtimes()
