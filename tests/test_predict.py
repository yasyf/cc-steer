from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from cc_steer.predict import DecisionPrediction, DecisionQuery, predict

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class StubHit:
    pair_id: int
    repo: str | None
    category: str
    verbatim: str
    direction: str
    score: float
    source: str


@pytest.fixture
def evidence_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    import cc_steer

    module = ModuleType("cc_steer.evidence")
    module.EvidenceHit = StubHit

    async def search(query, *, repo=None, limit=10, rerank=False, db=None):
        return []

    module.search = search
    monkeypatch.setitem(sys.modules, "cc_steer.evidence", module)
    monkeypatch.setattr(cc_steer, "evidence", module, raising=False)
    return module


def record_search(hits: list[StubHit], calls: list[dict[str, object]]):
    async def search(
        query: str, *, repo: str | None = None, limit: int = 10, rerank: bool = False, db: Path | None = None
    ):
        calls.append({"query": query, "repo": repo, "limit": limit, "rerank": rerank, "db": db})
        return hits

    return search


@pytest.mark.anyio
async def test_predict_is_evidence_only_and_passes_through_hits(
    evidence_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = [
        StubHit(
            pair_id=7,
            repo="acme",
            category="architecture",
            verbatim="prefer composition over inheritance",
            direction="do",
            score=0.91,
            source="triage",
        )
    ]
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(evidence_module, "search", record_search(hits, calls))

    result = await predict(
        DecisionQuery(
            question="Compose or inherit?", options=("compose", "inherit"), header="H", repo="acme", context="ctx"
        )
    )

    assert result == DecisionPrediction(status="evidence_only", choice=None, confidence=0.0, evidence=hits)
    assert result.evidence is hits
    assert calls == [{"query": "Compose or inherit?", "repo": "acme", "limit": 10, "rerank": False, "db": None}]


@pytest.mark.anyio
async def test_predict_forwards_repo_and_db(
    evidence_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(evidence_module, "search", record_search([], calls))
    db_path = tmp_path / "feedback.db"

    await predict(
        DecisionQuery(question="q", options=(), header=None, repo=None, context=""),
        db=db_path,
    )

    assert calls == [{"query": "q", "repo": None, "limit": 10, "rerank": False, "db": db_path}]


@pytest.mark.anyio
async def test_predict_returns_empty_evidence_when_no_hits(
    evidence_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evidence_module, "search", record_search([], []))

    result = await predict(DecisionQuery(question="q", options=(), header=None, repo=None, context=""))

    assert result.evidence == []
    assert result.status == "evidence_only"


def test_decision_query_is_frozen() -> None:
    query = DecisionQuery(question="q", options=(), header=None, repo=None, context="")
    with pytest.raises(FrozenInstanceError):
        query.question = "mutated"  # type: ignore[misc]


def test_decision_prediction_is_frozen() -> None:
    prediction = DecisionPrediction(status="evidence_only", choice=None, confidence=0.0, evidence=[])
    with pytest.raises(FrozenInstanceError):
        prediction.confidence = 1.0  # type: ignore[misc]
