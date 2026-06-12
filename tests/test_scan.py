from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from cc_pushback.scan import scan
from tests.builders import assistant_tool_use, denial_result, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path

    from cc_pushback.store import FeedbackStore

pytestmark = pytest.mark.anyio


def good_entries() -> list[dict[str, Any]]:
    return [
        assistant_tool_use("t1", "Write", {"file_path": "/a.py", "content": "x = 1"}),
        denial_result("t1", said="don't do that"),
        user_text("use a frozen dataclass here instead of a dict"),
    ]


@pytest.mark.integration
async def test_scan_inserts_then_is_incremental(store: FeedbackStore, tmp_path: Path) -> None:
    write_transcript(tmp_path / "proj" / "s.jsonl", good_entries())
    first = await scan(store, [tmp_path])
    assert first.scanned == 1
    assert first.inserted >= 2

    second = await scan(store, [tmp_path])
    assert second.scanned == 0
    assert second.inserted == 0


@pytest.mark.integration
async def test_full_rescan_reparses_but_stays_idempotent(store: FeedbackStore, tmp_path: Path) -> None:
    write_transcript(tmp_path / "proj" / "s.jsonl", good_entries())
    await scan(store, [tmp_path])
    total = (await store.stats()).total
    again = await scan(store, [tmp_path], full=True)
    assert again.scanned == 1
    assert again.inserted == 0
    assert (await store.stats()).total == total


@pytest.mark.integration
async def test_unparseable_transcript_is_skipped_and_left_unrecorded(store: FeedbackStore, tmp_path: Path) -> None:
    good = write_transcript(tmp_path / "proj" / "good.jsonl", good_entries())
    bad = tmp_path / "proj" / "bad.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"type": "user", "message": {"role": "user", "content": "hi"}}\n')  # valid JSON, missing uuid
    report = await scan(store, [tmp_path])
    assert report.scanned == 1
    mtimes = await store.file_mtimes()
    assert str(good) in mtimes
    assert str(bad) not in mtimes
