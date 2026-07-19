from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId

import cc_steer.negatives as negatives
from cc_steer.negatives import GateSample, event_samples, sample_negatives
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


def leading_empty_window(session: str, uuid: str) -> str:
    # A zero-length leading assistant turn, content only in the trailing user turn.
    return ContextWindow(
        anchor=EventRef(SessionId(session), EventUuid(uuid)),
        before=(
            TurnRef(role="assistant", refs=(), preview="", tool_digests=()),
            TurnRef(role="user", refs=(), preview="drop the vendored copy", tool_digests=()),
        ),
        trigger=TurnRef(role="user", refs=(), preview="no, do it differently", tool_digests=()),
        after=(),
        fidelity="full",
        preview_chars=200,
    ).to_json()


def test_event_samples_stops_rewinding_before_the_window_goes_empty() -> None:
    row = event_row("k-empty", steering=True) | {"context_json": leading_empty_window(TRAIN_SESSION, "u-empty")}
    samples = event_samples([row], offsets=6)
    # Offset -1 rewinds to the bare leading assistant turn, so rewinding stops at 0.
    assert [sample.offset_turns for sample in samples] == [0]
    assert samples[0].sample_key == "pos:k-empty:0"


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


async def test_sample_negatives_marks_a_dropped_only_session(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = "d0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{session}.jsonl", session_entries(session))
    empty = ContextWindow(
        anchor=EventRef(SessionId(session), EventUuid("x")),
        before=(TurnRef(role="assistant", refs=(), preview="", tool_digests=()),),
        trigger=None,
        after=(),
        fidelity="full",
        preview_chars=200,
    ).to_json()
    monkeypatch.setattr(
        negatives,
        "random_samples",
        lambda activity, exclude, *, per_session, seed: [
            GateSample(
                sample_key=f"rand:{session}:x",
                kind="random_negative",
                dedup_key=None,
                session_id=session,
                anchor_uuid="x",
                occurred_at=None,
                offset_turns=0,
                window_json=empty,
                seed=seed,
            )
        ],
    )
    report = await sample_negatives(store, [tmp_path], seed=1, sessions=10, per_session=2, min_bytes=0)
    # Every sample is empty and dropped, yet the parsed session is marked done.
    assert report.inserted["random_negative"] == 0
    assert report.sessions_sampled == 1
    assert session in await store.negative_sessions()

    again = await sample_negatives(store, [tmp_path], seed=1, sessions=10, per_session=2, min_bytes=0)
    assert again.sessions_sampled == 0


async def test_sample_negatives_marks_zero_turn_sessions_so_budget_advances(
    store: FeedbackStore, tmp_path: Path
) -> None:
    sessions = {"a0000000-0000-0000-0000-000000000000", "b0000000-0000-0000-0000-000000000000"}
    for session in sessions:
        write_transcript(tmp_path / "proj" / f"{session}.jsonl", [user_text("x", sessionId=session, uuid=session)])
    # A single-prompt transcript has no completed turn to anchor a negative on, so
    # it yields nothing yet is still marked, advancing a fixed sessions=1 budget.
    first = await sample_negatives(store, [tmp_path], seed=1, sessions=1, min_bytes=0)
    # A zero-turn transcript produces no samples but is still marked, so a fixed
    # sessions=1 budget advances to the other candidate instead of re-parsing it.
    assert (first.sessions_sampled, first.inserted["random_negative"]) == (1, 0)
    assert len(await store.negative_sessions()) == 1
    second = await sample_negatives(store, [tmp_path], seed=1, sessions=1, min_bytes=0)
    assert second.sessions_sampled == 1
    assert await store.negative_sessions() == sessions


async def test_sample_negatives_isolates_a_corrupt_transcript(store: FeedbackStore, tmp_path: Path) -> None:
    good = "a0000000-0000-0000-0000-000000000000"
    corrupt = "e0000000-0000-0000-0000-000000000000"
    write_transcript(tmp_path / "proj" / f"{good}.jsonl", session_entries(good))
    (tmp_path / "proj" / f"{corrupt}.jsonl").write_text("{ not valid json\n")
    report = await sample_negatives(store, [tmp_path], seed=1, sessions=10, per_session=2, min_bytes=0)
    marked = await store.negative_sessions()
    assert good in marked  # the good session is processed and marked
    assert corrupt not in marked  # the corrupt file is skipped, not marked
    assert report.sessions_sampled == 1  # and the pass completes rather than aborting on the corrupt file


async def test_sample_negatives_excludes_turns_near_detected_events(store: FeedbackStore, tmp_path: Path) -> None:
    session = "c0000000-0000-0000-0000-000000000000"
    entries = session_entries(session)
    write_transcript(tmp_path / "proj" / f"{session}.jsonl", entries)
    await store.execute(
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
