from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from cc_steer.retrain import mine_deep
from cc_steer.retrain.mine_deep import STATUS_NAME, deep_mine, sweep_detectors
from tests.builders import assistant_text, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path

    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio


def steering_transcript(session: str) -> list[dict[str, object]]:
    # An assistant turn then a clear steering correction — one transcript_message candidate.
    return [
        assistant_text("I will store sessions server-side", sessionId=session, uuid=f"{session}-a0"),
        user_text("no, use JWT, sessions are wrong here", sessionId=session, uuid=f"{session}-u0"),
        assistant_text("switching to JWT now", sessionId=session, uuid=f"{session}-a1"),
        user_text("ok", sessionId=session, uuid=f"{session}-u1"),
    ]


def quiet_transcript(session: str) -> list[dict[str, object]]:
    # Benign acks the detector ignores, giving the negative sampler quiet stretches to anchor on.
    entries: list[dict[str, object]] = []
    for i in range(16):
        entries.append(user_text("ok", sessionId=session, uuid=f"{session}-u{i}"))
        entries.append(assistant_text(f"did step {i}", sessionId=session, uuid=f"{session}-a{i}"))
    return entries


async def events_count(store: FeedbackStore) -> int:
    return int((await store.sql("SELECT COUNT(*) AS n FROM feedback_events"))[0]["n"])


async def test_deep_mine_persists_detectors_and_negatives(store: FeedbackStore, tmp_path: Path) -> None:
    steer = "a0000000-0000-0000-0000-000000000000"
    quiet = "b0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{steer}.jsonl", steering_transcript(steer))
    write_transcript(tmp_path / "proj" / f"{quiet}.jsonl", quiet_transcript(quiet))
    out = tmp_path / "out"
    report = await deep_mine(store, [tmp_path], seed=1, per_session=4, min_bytes=0, out=out)

    assert report.detector_files == 2  # both transcripts scanned
    assert report.detector_inserted >= 1  # the steering correction is captured
    assert await events_count(store) == report.detector_inserted
    assert report.negative_sessions == 2
    assert report.inserted["random_negative"] >= 1  # real-silence turns land as negatives
    assert report.inserted["positive_window"] == 0  # no triage ran, so no judged event samples
    status = json.loads((out / STATUS_NAME).read_text())
    assert status["status"] == "done" and status["phase"] == "done"
    assert (out / "mine_deep.log").exists()


async def test_second_pass_is_idempotent(store: FeedbackStore, tmp_path: Path) -> None:
    steer = "b0000000-0000-0000-0000-000000000000"
    quiet = "c0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{steer}.jsonl", steering_transcript(steer))
    write_transcript(tmp_path / "proj" / f"{quiet}.jsonl", quiet_transcript(quiet))
    first = await deep_mine(store, [tmp_path], seed=1, per_session=4, min_bytes=0)
    again = await deep_mine(store, [tmp_path], seed=1, per_session=4, min_bytes=0)
    assert first.detector_inserted >= 1
    assert first.inserted["random_negative"] >= 1
    assert again.detector_files == 0  # the recorded mtime is the checkpoint; nothing re-detected
    assert again.detector_inserted == 0
    assert again.negative_sessions == 0
    assert again.inserted["random_negative"] == 0


async def test_a_file_landing_mid_sweep_is_caught_next_pass(store: FeedbackStore, tmp_path: Path) -> None:
    first_session = "c0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{first_session}.jsonl", steering_transcript(first_session))
    files, _ = await sweep_detectors(store, [tmp_path])
    assert files == 1
    # A concurrent rsync lands a second transcript after the first sweep; the re-glob catches it.
    late = "d0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{late}.jsonl", steering_transcript(late))
    more, _ = await sweep_detectors(store, [tmp_path])
    assert more == 1


async def test_corrupt_transcript_is_skipped_not_recorded(store: FeedbackStore, tmp_path: Path) -> None:
    good = "e0000000-0000-0000-0000-000000000000"
    corrupt = "f0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{good}.jsonl", steering_transcript(good))
    (tmp_path / "proj" / f"{corrupt}.jsonl").write_text("{ not valid json\n")
    files, inserted = await sweep_detectors(store, [tmp_path])
    assert files == 1  # only the good transcript is recorded; the half-written one is skipped
    assert inserted >= 1
    mtimes = await store.file_mtimes()
    assert any(good in path for path in mtimes)
    assert not any(corrupt in path for path in mtimes)  # unrecorded, so a completed rsync retries it


async def test_negatives_only_skips_detectors(store: FeedbackStore, tmp_path: Path) -> None:
    quiet = "10000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{quiet}.jsonl", quiet_transcript(quiet))
    report = await deep_mine(store, [tmp_path], seed=1, per_session=4, min_bytes=0, detectors=False)
    assert report.detector_files == 0
    assert await events_count(store) == 0  # detectors skipped -> no candidates
    assert report.inserted["random_negative"] >= 1


async def test_no_roots_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mine_deep, "default_roots", lambda: ())
    with pytest.raises(SystemExit):
        await mine_deep._run(mine_deep._parse_args(["--out", str(tmp_path / "out")]))
