from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import DedupKey

from cc_steer.exemplars import (
    VOYAGE_BATCH_CHARS,
    VOYAGE_BATCH_TEXTS,
    VoyageQueryEncoder,
    build_index,
    exemplars_for,
    load_index,
    mmr_select,
    query_encoder,
    voyage_batches,
)
from cc_steer.rendering import split_of
from cc_steer.triage import JUDGE, PROMPT_VERSION, Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio

TRAIN_SESSION = "sess-0"
TEST_SESSION = "sess-14"


class CountingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([np.full(4, float(len(text)), dtype=np.float32) for text in texts])


def window_json(session: str, uuid: str) -> str:
    return ContextWindow(
        anchor=EventRef(SessionId(session), EventUuid(uuid)),
        before=(
            TurnRef(role="user", refs=(), preview="please fix the bug", tool_digests=()),
            TurnRef(role="assistant", refs=(), preview="I rewrote the module", tool_digests=()),
        ),
        trigger=TurnRef(role="user", refs=(), preview="too big, make a surgical fix", tool_digests=()),
        after=(),
        fidelity="full",
        preview_chars=200,
    ).to_json()


async def seed_steering(store: FeedbackStore, key: str, session: str, uuid: str) -> None:
    await store.execute(
        "INSERT INTO feedback_events (dedup_key, source_kind, session_id, event_uuid, "
        "occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            key,
            "transcript_message",
            session,
            uuid,
            "2026-01-01T00:00:00",
            "too big, make a surgical fix",
            json.dumps({"signal": {}}),
            window_json(session, uuid),
            "2.0.1",
            "2026-01-01T00:00:00",
            "/h-proj/s.jsonl",
        ),
    )
    verdict = Verdict.model_validate(
        {"category": "wrong_approach", "what_claude_did": "rewrote the module", "confidence": 0.9, "rationale": "r"}
    )
    await store.record_verdict(
        DedupKey(key), verdict, role=JUDGE, prompt_version=PROMPT_VERSION, model="opus", fidelity="full"
    )


def test_mmr_prefers_diverse_picks() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.9, 0.436], [0.85, -0.527], [0.895, 0.446]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    diverse = [index for index, _ in mmr_select(query, matrix, k=2, diversity=0.5)]
    assert diverse == [0, 1]
    greedy = [index for index, _ in mmr_select(query, matrix, k=2, diversity=0.0)]
    assert greedy == [0, 2]
    assert mmr_select(query, np.zeros((0, 0), dtype=np.float32), k=2) == []


def test_mmr_scores_are_query_similarities() -> None:
    query = np.array([2.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hits = mmr_select(query, matrix, k=2, diversity=0.0)
    assert [index for index, _ in hits] == [0, 1]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(0.0)


async def test_build_index_embeds_train_split_only_and_is_incremental(store: FeedbackStore) -> None:
    assert split_of(TRAIN_SESSION) == "train"
    assert split_of(TEST_SESSION) == "test"
    await seed_steering(store, "k-train", TRAIN_SESSION, "u1")
    await seed_steering(store, "k-test", TEST_SESSION, "u2")
    encoder = CountingEncoder()
    report = await build_index(store, encoder=encoder)
    assert (report.embedded, report.current, report.total) == (1, 0, 1)
    again = await build_index(store, encoder=encoder)
    assert (again.embedded, again.current, again.total) == (0, 1, 1)
    assert len(encoder.calls) == 1


async def test_load_index_round_trips_normalized_vectors(store: FeedbackStore) -> None:
    await seed_steering(store, "k-train", TRAIN_SESSION, "u1")
    await build_index(store, encoder=CountingEncoder())
    keys, matrix = await load_index(store)
    assert keys == ["k-train"]
    assert matrix.shape == (1, 4)
    assert float(np.linalg.norm(matrix[0])) == pytest.approx(1.0)
    empty_keys, empty = await load_index(store, model="other-model")
    assert empty_keys == []
    assert empty.size == 0


async def test_exemplars_for_hydrates_in_hit_order(store: FeedbackStore) -> None:
    await seed_steering(store, "k-train", TRAIN_SESSION, "u1")
    exemplars = await exemplars_for(store, [("k-train", 0.87), ("missing", 0.5)])
    assert [exemplar.dedup_key for exemplar in exemplars] == ["k-train"]
    assert exemplars[0].verbatim == "too big, make a surgical fix"
    assert exemplars[0].score == 0.87
    assert exemplars[0].category == "wrong_approach"
    assert "too big, make a surgical fix" not in exemplars[0].context_text
    assert await exemplars_for(store, []) == []


def test_voyage_batches_respect_text_and_char_budgets() -> None:
    texts = ["x" * 100] * (VOYAGE_BATCH_TEXTS + 10)
    batches = voyage_batches(texts)
    assert [index for batch in batches for index in batch] == list(range(len(texts)))
    assert max(len(batch) for batch in batches) <= VOYAGE_BATCH_TEXTS

    big = ["y" * (VOYAGE_BATCH_CHARS // 2 + 1)] * 3
    assert [len(batch) for batch in voyage_batches(big)] == [1, 1, 1]
    assert voyage_batches([]) == []


def test_query_encoder_dispatches_on_model_name() -> None:
    encoder = query_encoder("voyage-4-large")
    assert isinstance(encoder, VoyageQueryEncoder)
    assert encoder.model == "voyage-4-large"
