"""Predict a steering decision from recorded corrections.

The v0 contract is evidence-only: :func:`predict` surfaces the corrections that
match a decision without choosing an option, so callers integrate against the
final shape while the ranking model lands in a later experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from cc_steer.evidence import EvidenceHit


@dataclass(frozen=True, slots=True)
class DecisionQuery:
    """A steering decision to resolve against recorded corrections.

    Example:
        >>> DecisionQuery(
        ...     question="Compose or inherit?",
        ...     options=("compose", "inherit"),
        ...     header=None,
        ...     repo="acme",
        ...     context="",
        ... )
    """

    question: str
    options: tuple[str, ...]
    header: str | None
    repo: str | None
    context: str


@dataclass(frozen=True, slots=True)
class DecisionPrediction:
    """The predicted resolution of a :class:`DecisionQuery` with its evidence.

    In v0 ``status`` is ``"evidence_only"``, ``choice`` is ``None`` and
    ``confidence`` is ``0.0``; ``evidence`` carries the real matching corrections.
    """

    status: str
    choice: str | None
    confidence: float
    evidence: list[EvidenceHit]


async def predict(query: DecisionQuery, *, db: Path | None = None) -> DecisionPrediction:
    """Predict a steering decision, evidence-only in v0.

    Args:
        query: The decision to resolve.
        db: Database to draw evidence from. Defaults to the standard feedback DB.

    Returns:
        A prediction that abstains from choosing (``status="evidence_only"``,
        ``choice=None``, ``confidence=0.0``) and carries the corrections matching
        ``query.question``.
    """
    from cc_steer import evidence

    return DecisionPrediction(
        status="evidence_only",
        choice=None,
        confidence=0.0,
        evidence=await evidence.search(query.question, repo=query.repo, db=db),
    )
