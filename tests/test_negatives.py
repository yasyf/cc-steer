from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from cc_transcript.context import ContextWindow

from cc_steer.negatives import event_samples, sample_negatives
from tests.builders import assistant_text, user_text, write_transcript
from tests.test_exemplars import TRAIN_SESSION, seed_steering, window_json

if TYPE_CHECKING:
    from pathlib import Path

    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio


def event_row(key: str, *, steering: bool) -> dict[str, object]:
    return {
        "dedup_key": key,
        "session_id": TRAIN_SESSION,
        "event_uuid": f"u-{key}",
        "occurred_at": "2026-01-01T00:00:00",
        "context_json": window_json(TRAIN_SESSION, f"u-{key}"),
        "is_steering": steering,
    }


def test_event_samples_rewinds_positives_until_nothing_remains() -> None:
    samples = event_samples([event_row("k1", steering=True)], offsets=6)
    # The fixture window has two before turns: offsets 0 and -1 exist; -2 would
    # leave nothing, so rewinding stops there.
    assert [sample.offset_turns for sample in samples] == [0, -1]
    assert [sample.sample_key for sample in samples] == ["pos:k1:0", "pos:k1:1"]
    assert {sample.kind for sample in samples} == {"positive_window"}
    window = ContextWindow.from_json(samples[0].window_json)
    assert [turn.preview for turn in window.before] == ["please fix the bug", "I rewrote the module"]
    rewound = ContextWindow.from_json(samples[1].window_json)
    assert [turn.preview for turn in rewound.before] == ["please fix the bug"]


def test_event_samples_emits_one_hard_negative_per_rejected_event() -> None:
    samples = event_samples([event_row("k2", steering=False)])
    assert [sample.kind for sample in samples] == ["hard_negative"]
    assert samples[0].sample_key == "hard:k2"
    assert samples[0].offset_turns == 0


def test_event_samples_skips_malformed_windows() -> None:
    assert event_samples([event_row("k3", steering=True) | {"context_json": "{}"}]) == []


def session_entries(session: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for i in range(16):
        entries.append(user_text(f"please do step {i}", sessionId=session, uuid=f"{session}-u{i}"))
        entries.append(assistant_text(f"did step {i}", sessionId=session, uuid=f"{session}-a{i}"))
    return entries


async def test_sample_negatives_end_to_end(store: FeedbackStore, tmp_path: Path) -> None:
    await seed_steering(store, "k-train", TRAIN_SESSION, "u1")
    quiet = "b0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{quiet}.jsonl", session_entries(quiet))
    write_transcript(tmp_path / "proj" / f"agent-{quiet}.jsonl", session_entries("side"))
    report = await sample_negatives(store, [tmp_path], seed=1, sessions=10, per_session=2, min_bytes=0)
    assert report.inserted["positive_window"] == 2
    assert report.inserted["hard_negative"] == 0
    assert report.inserted["random_negative"] == 2
    assert report.sessions_sampled == 1
    randoms = await store.gate_samples(kind="random_negative")
    assert {str(row["session_id"]) for row in randoms} == {quiet}
    window = ContextWindow.from_json(str(randoms[0]["window_json"]))
    assert window.trigger is None
    assert window.before

    again = await sample_negatives(store, [tmp_path], seed=1, sessions=10, per_session=2, min_bytes=0)
    assert again.inserted == {"positive_window": 0, "hard_negative": 0, "random_negative": 0}
    assert again.sessions_sampled == 0

    stats = await store.gate_sample_stats()
    assert stats == {"positive_window": 2, "random_negative": 2}


async def test_sample_negatives_excludes_turns_near_detected_events(store: FeedbackStore, tmp_path: Path) -> None:
    session = "c0000000-0000-0000-0000-000000000000"
    entries = session_entries(session)
    write_transcript(tmp_path / "proj" / f"{session}.jsonl", entries)
    await store.store.conn.execute(
        "INSERT INTO feedback_events (dedup_key, source_kind, session_id, event_uuid, "
        "occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "k-anchor",
            "transcript_message",
            session,
            f"{session}-u4",
            "2026-01-01T00:00:00",
            "steer",
            json.dumps({"signal": {}}),
            window_json(session, f"{session}-u4"),
            "2.0.1",
            "2026-01-01T00:00:00",
            "/h-proj/s.jsonl",
        ),
    )
    await sample_negatives(store, [tmp_path], seed=3, sessions=10, per_session=50, min_bytes=0)
    randoms = await store.gate_samples(kind="random_negative")
    sampled_uuids = {str(row["anchor_uuid"]) for row in randoms if str(row["session_id"]) == session}
    # u4 opens turn 5; the backward-only radius excludes turns 0-5, and the
    # final turn (16) is never sampled — anchors u5-u14 remain eligible.
    assert sampled_uuids
    assert sampled_uuids <= {f"{session}-u{i}" for i in range(5, 15)}
    assert not sampled_uuids & {f"{session}-u{i}" for i in range(5)}
