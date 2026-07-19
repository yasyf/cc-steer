from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import anyio
import numpy as np
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import DedupKey
from click.testing import CliRunner

import cc_steer.exemplars
from cc_steer.cli import main
from cc_steer.exemplars import (
    EMBED_MODEL,
    Exemplar,
    build_index,
    exemplar_text,
    exemplars_for,
    load_index,
    mmr_select,
)
from cc_steer.rendering import split_of
from cc_steer.store import FeedbackStore
from cc_steer.triage import JUDGE, PROMPT_VERSION, Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio

TRAIN_SESSION = "sess-0"
VERBATIM = "too big, make a surgical fix"


class BasisEncoder:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.slots: dict[str, int] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.basis(text) for text in texts])

    def basis(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[self.slots.setdefault(text, len(self.slots))] = 1.0
        return vector


def window_json(uuid: str, *, fix: str, rewrite: str) -> str:
    return ContextWindow(
        anchor=EventRef(SessionId(TRAIN_SESSION), EventUuid(uuid)),
        before=(
            TurnRef(role="user", refs=(), preview=fix, tool_digests=()),
            TurnRef(role="assistant", refs=(), preview=rewrite, tool_digests=()),
        ),
        trigger=TurnRef(role="user", refs=(), preview=VERBATIM, tool_digests=()),
        after=(),
        fidelity="full",
        preview_chars=200,
    ).to_json()


async def seed_steering(store: FeedbackStore, key: str, uuid: str, *, fix: str, rewrite: str) -> str:
    context_json = window_json(uuid, fix=fix, rewrite=rewrite)
    await store.execute(
        "INSERT INTO feedback_events (dedup_key, source_kind, session_id, event_uuid, "
        "occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            key,
            "transcript_message",
            TRAIN_SESSION,
            uuid,
            "2026-01-01T00:00:00",
            VERBATIM,
            json.dumps({"signal": {}}),
            context_json,
            "2.0.1",
            "2026-01-01T00:00:00",
            "/h-proj/s.jsonl",
        ),
    )
    await store.record_verdict(
        DedupKey(key),
        Verdict.model_validate(
            {"category": "wrong_approach", "what_claude_did": rewrite, "confidence": 0.9, "rationale": "r"}
        ),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        model="opus",
        fidelity="full",
    )
    return context_json


async def lookup(store: FeedbackStore, encoder: BasisEncoder, context: str, *, k: int) -> list[Exemplar]:
    keys, matrix = await load_index(store, model=EMBED_MODEL)
    hits = mmr_select(encoder.encode([context])[0], matrix, k=k)
    return await exemplars_for(store, [(keys[index], score) for index, score in hits])


async def test_lookup_ranks_the_matching_exemplar_first(store: FeedbackStore) -> None:
    assert split_of(TRAIN_SESSION) == "train"
    ctx_auth = await seed_steering(store, "k-auth", "u1", fix="fix the auth bug", rewrite="rewrote the auth module")
    await seed_steering(store, "k-test", "u2", fix="add a test", rewrite="added five tests")
    encoder = BasisEncoder()
    report = await build_index(store, encoder=encoder)
    assert (report.embedded, report.total) == (2, 2)

    query = exemplar_text(ctx_auth)
    assert query is not None
    exemplars = await lookup(store, encoder, query, k=5)

    assert [exemplar.dedup_key for exemplar in exemplars] == ["k-auth", "k-test"]
    assert exemplars[0].score == pytest.approx(1.0)
    assert exemplars[1].score == pytest.approx(0.0)
    assert exemplars[0].verbatim == VERBATIM
    assert exemplars[0].category == "wrong_approach"
    assert exemplars[0].direction == ""
    assert VERBATIM not in exemplars[0].context_text


async def test_lookup_honors_k(store: FeedbackStore) -> None:
    await seed_steering(store, "k-auth", "u1", fix="fix the auth bug", rewrite="rewrote the auth module")
    await seed_steering(store, "k-test", "u2", fix="add a test", rewrite="added five tests")
    encoder = BasisEncoder()
    await build_index(store, encoder=encoder)
    assert len(await lookup(store, encoder, "anything", k=1)) == 1


async def test_lookup_json_output_carries_every_exemplar_field(store: FeedbackStore) -> None:
    ctx_auth = await seed_steering(store, "k-auth", "u1", fix="fix the auth bug", rewrite="rewrote the auth module")
    encoder = BasisEncoder()
    await build_index(store, encoder=encoder)
    query = exemplar_text(ctx_auth)
    assert query is not None
    exemplars = await lookup(store, encoder, query, k=5)

    payload = json.loads(json.dumps([dataclasses.asdict(exemplar) for exemplar in exemplars]))
    assert set(payload[0]) == {"dedup_key", "context_text", "direction", "verbatim", "category", "score"}
    assert payload[0]["dedup_key"] == "k-auth"
    assert payload[0]["verbatim"] == VERBATIM
    assert payload[0]["category"] == "wrong_approach"
    assert payload[0]["score"] == pytest.approx(1.0)


async def test_empty_index_yields_no_keys(store: FeedbackStore) -> None:
    keys, matrix = await load_index(store, model=EMBED_MODEL)
    assert keys == []
    assert matrix.size == 0


def seeded_db(tmp_path: Path, encoder: BasisEncoder) -> tuple[Path, str]:
    query: list[str] = []

    async def build() -> None:
        async with await FeedbackStore.open(tmp_path / "feedback.db") as store:
            context = await seed_steering(store, "k-auth", "u1", fix="fix the auth bug", rewrite="rewrote the auth")
            hit = exemplar_text(context)
            assert hit is not None
            query.append(hit)
            await build_index(store, encoder=encoder)

    anyio.run(build)
    return tmp_path / "feedback.db", query[0]


def empty_db(tmp_path: Path) -> Path:
    async def build() -> None:
        async with await FeedbackStore.open(tmp_path / "feedback.db"):
            pass

    anyio.run(build)
    return tmp_path / "feedback.db"


def test_exemplars_command_errors_on_empty_index(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["exemplars", "anything", "--db", str(empty_db(tmp_path))])
    assert result.exit_code != 0
    assert "the exemplar index is empty" in result.output


def test_exemplars_command_prints_the_score_category_verbatim_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = BasisEncoder()
    db, query = seeded_db(tmp_path, encoder)
    monkeypatch.setattr(cc_steer.exemplars, "query_encoder", lambda model: encoder)
    result = CliRunner().invoke(main, ["exemplars", query, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"[1.000] wrong_approach: {VERBATIM}"


def test_exemplars_command_json_emits_the_dataclass_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = BasisEncoder()
    db, query = seeded_db(tmp_path, encoder)
    monkeypatch.setattr(cc_steer.exemplars, "query_encoder", lambda model: encoder)
    result = CliRunner().invoke(main, ["exemplars", query, "--json", "--db", str(db)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list) and len(payload) == 1
    assert set(payload[0]) == {"dedup_key", "context_text", "direction", "verbatim", "category", "score"}
    assert (payload[0]["dedup_key"], payload[0]["category"], payload[0]["verbatim"]) == (
        "k-auth",
        "wrong_approach",
        VERBATIM,
    )
