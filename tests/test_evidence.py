from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from cc_transcript.corrections import Correction, CorrectionLog
from cc_transcript.ids import EventUuid, SessionId
from cc_transcript.mining import DedupKey

import cc_steer.exemplars
from cc_steer.enrich import SOURCE
from cc_steer.evidence import EvidenceHit, search
from cc_steer.refine import RefinedPair, Refinement
from cc_steer.store import FeedbackStore
from cc_steer.triage import JUDGE, Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio

TS = "2026-06-01T00:00:00+00:00"
YCLAW = "/Users/yasyf/.claude/projects/-Users-yasyf-Code-yclaw/session.jsonl"
STEER = "/Users/yasyf/.claude/projects/-Users-yasyf-Code-cc-steer/session.jsonl"


def verdict(category: str) -> Verdict:
    return Verdict.model_validate(
        {"category": category, "what_claude_did": "ran a tool", "confidence": 0.9, "rationale": "r"}
    )


async def seed(
    store: FeedbackStore,
    *,
    key: str,
    verbatim: str,
    direction: str,
    session_id: str = "sess-1",
    event_uuid: str = "evt-1",
    origin_path: str = YCLAW,
    category: str = "wrong_approach",
    source_kind: str = "transcript_message",
    action: str = "ran a tool",
) -> None:
    await store.execute(
        "INSERT INTO feedback_events "
        "(dedup_key, source_kind, session_id, event_uuid, occurred_at, text, "
        "payload_json, context_json, cc_version, ingested_at, origin_path) "
        "VALUES (?, ?, ?, ?, ?, ?, '{}', '{}', '1.0', ?, ?)",
        (key, source_kind, session_id, event_uuid, TS, verbatim, TS, origin_path),
    )
    await store.record_verdict(
        DedupKey(key), verdict(category), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_refinement(
        DedupKey(key),
        Refinement(pairs=[RefinedPair(action=action, direction_verbatim=verbatim, direction=direction)]),
        prompt_version=1,
        model="sonnet",
    )


async def append_correction(*, session_id: str, event_uuid: str, incorrect_file: str, source: str = SOURCE) -> None:
    log = await CorrectionLog.open()
    await log.append(
        Correction(
            ts_ms=1_000,
            session_id=SessionId(session_id),
            source=source,
            anchor_uuid=EventUuid(event_uuid),
            incorrect_digest=None,
            incorrect_file=incorrect_file,
            incorrect_old="def broken(): ...",
            incorrect_new="def still_broken(): ...",
            correction_origin="session",
            correction_file=incorrect_file,
            correction_old="def still_broken(): ...",
            correction_new="def fixed(): ...",
        )
    )
    await log.close()


class FakeEncoder:
    def __init__(self, axis_by_token: dict[str, int], dims: int) -> None:
        self.axis_by_token = axis_by_token
        self.dims = dims

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dims, dtype=np.float32)
            for token, axis in self.axis_by_token.items():
                if token in text:
                    vector[axis] = 1.0
            vectors.append(vector)
        return np.stack(vectors)


async def test_search_matches_verbatim_and_returns_full_hit(store: FeedbackStore, tmp_path: Path) -> None:
    await seed(store, key="k1", verbatim="stop hardcoding the retry ceiling", direction="Use a configurable retry cap")
    hits = await search("hardcoding retry", db=tmp_path / "feedback.db")
    assert hits == [
        EvidenceHit(
            pair_id=1,
            repo="yclaw",
            category="wrong_approach",
            verbatim="stop hardcoding the retry ceiling",
            direction="Use a configurable retry cap",
            score=hits[0].score,
            source="transcript_message",
        )
    ]
    assert hits[0].score > 0


async def test_search_matches_direction_text(store: FeedbackStore, tmp_path: Path) -> None:
    await seed(store, key="k1", verbatim="no", direction="Prefer a dataclass over a bare dict for config")
    assert [hit.verbatim for hit in await search("dataclass config", db=tmp_path / "feedback.db")] == ["no"]


async def test_search_matches_correction_evidence_via_anchor_join(store: FeedbackStore, tmp_path: Path) -> None:
    await seed(
        store,
        key="k1",
        verbatim="that is wrong",
        direction="Fix it",
        session_id="sess-9",
        event_uuid="evt-9",
    )
    await append_correction(session_id="sess-9", event_uuid="evt-9", incorrect_file="scripts/collect-secrets.sh")
    hits = await search("collect secrets", db=tmp_path / "feedback.db")
    assert [hit.verbatim for hit in hits] == ["that is wrong"]


async def test_correction_from_other_source_is_not_indexed(store: FeedbackStore, tmp_path: Path) -> None:
    await seed(store, key="k1", verbatim="wrong", direction="right", session_id="sess-9", event_uuid="evt-9")
    await append_correction(
        session_id="sess-9", event_uuid="evt-9", incorrect_file="scripts/collect-secrets.sh", source="captain-hook"
    )
    assert await search("collect secrets", db=tmp_path / "feedback.db") == []


async def test_repo_filter_restricts_hits(store: FeedbackStore, tmp_path: Path) -> None:
    await seed(store, key="k1", verbatim="run the tests not the build", direction="Run tests", origin_path=YCLAW)
    await seed(
        store,
        key="k2",
        verbatim="run the tests before pushing",
        direction="Run tests",
        origin_path=STEER,
        session_id="sess-2",
        event_uuid="evt-2",
    )
    hits = await search("run tests", repo="cc-steer", db=tmp_path / "feedback.db")
    assert [(hit.repo, hit.verbatim) for hit in hits] == [("cc-steer", "run the tests before pushing")]


async def test_limit_truncates_shortlist(store: FeedbackStore, tmp_path: Path) -> None:
    for i in range(3):
        await seed(
            store,
            key=f"k{i}",
            verbatim=f"always run lint step {i}",
            direction="Run lint",
            session_id=f"sess-{i}",
            event_uuid=f"evt-{i}",
        )
    assert len(await search("run lint", limit=2, db=tmp_path / "feedback.db")) == 2


async def test_blank_query_returns_no_hits(store: FeedbackStore, tmp_path: Path) -> None:
    await seed(store, key="k1", verbatim="anything", direction="Do it")
    assert await search("   !!!   ", db=tmp_path / "feedback.db") == []


async def test_lazy_rebuild_picks_up_new_pairs(tmp_path: Path) -> None:
    # Close the native store before searching: it and search's stdlib sqlite3 must
    # not hold feedback.db open at once (standalone `cc-steer evidence` never does).
    db = tmp_path / "feedback.db"
    async with await FeedbackStore.open(db) as store:
        await seed(store, key="k1", verbatim="prefer async over threads here", direction="Use async")
    assert len(await search("prefer async", db=db)) == 1
    async with await FeedbackStore.open(db) as store:
        await seed(
            store,
            key="k2",
            verbatim="prefer async everywhere in this module",
            direction="Use async",
            session_id="sess-2",
            event_uuid="evt-2",
        )
    assert len(await search("prefer async", db=db)) == 2


async def test_rerank_orders_by_mmr_cosine(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed(store, key="k1", verbatim="review the auth flow", direction="d", session_id="s1", event_uuid="e1")
    await seed(store, key="k2", verbatim="review the auth logic", direction="d", session_id="s2", event_uuid="e2")
    await seed(store, key="k3", verbatim="review the parser output", direction="d", session_id="s3", event_uuid="e3")
    monkeypatch.setattr(
        cc_steer.exemplars,
        "query_encoder",
        lambda model: FakeEncoder({"parser": 0, "auth": 1, "review": 0}, dims=2),
    )
    hits = await search("review", rerank=True, db=tmp_path / "feedback.db")
    assert len(hits) == 3
    assert hits[0].verbatim == "review the parser output"
    assert hits[0].score == pytest.approx(1.0)
