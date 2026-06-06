from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NewType

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    from cc_transcript.models import SessionId

__all__ = [
    "ContextSnapshot",
    "ContextTurn",
    "DedupKey",
    "FeedbackCandidate",
    "PrRef",
    "SourceKind",
]

DedupKey = NewType("DedupKey", str)
PrRef = NewType("PrRef", str)

SourceKind = Literal[
    "transcript_message",
    "plan_review",
    "interrupt_rejection",
    "review_comment",
    "github_review",
    "superset_issue",
]


@dataclass(frozen=True, slots=True)
class ContextTurn:
    """One conversational turn surrounding a piece of feedback.

    Attributes:
        role: Whether the turn came from the user, the assistant, or a tool.
        text: The turn's text content.
        tool_calls: The names of the tools the turn invoked, in order.
    """

    role: Literal["user", "assistant", "tool"]
    text: str
    tool_calls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """The conversational window around a piece of feedback.

    Attributes:
        before: The turns leading up to the trigger.
        trigger: The assistant action the feedback responds to, when known.
        after: The turns following the trigger.
    """

    before: tuple[ContextTurn, ...]
    trigger: ContextTurn | None
    after: tuple[ContextTurn, ...]

    def to_json(self) -> str:
        """Serializes the snapshot to the JSON stored in ``context_json``."""
        return json.dumps(
            {
                "before": [turn_to_dict(turn) for turn in self.before],
                "trigger": turn_to_dict(self.trigger) if self.trigger else None,
                "after": [turn_to_dict(turn) for turn in self.after],
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> ContextSnapshot:
        """Deserializes a snapshot from a ``context_json`` string."""
        data = json.loads(raw)
        return cls(
            before=tuple(turn_from_dict(turn) for turn in data["before"]),
            trigger=turn_from_dict(data["trigger"]) if data["trigger"] else None,
            after=tuple(turn_from_dict(turn) for turn in data["after"]),
        )


@dataclass(frozen=True, slots=True)
class FeedbackCandidate:
    """A single piece of developer pushback extracted from a source.

    Attributes:
        dedup_key: The content-derived key that makes ingestion idempotent.
        source_kind: Which detector produced the candidate.
        occurred_at: When the feedback was given.
        text: The verbatim pushback text.
        context: The conversational window around the feedback.
        session_id: The transcript session, when sourced from a transcript.
        pr_ref: The pull-request reference, when sourced from GitHub.
        origin_path: The file the candidate was extracted from.
        origin_uuid: The originating transcript entry's uuid.
        cc_version: The Claude Code version recorded for the origin.
        payload: Detector-specific metadata preserved verbatim.
    """

    dedup_key: DedupKey
    source_kind: SourceKind
    occurred_at: datetime
    text: str
    context: ContextSnapshot
    session_id: SessionId | None = None
    pr_ref: PrRef | None = None
    origin_path: Path | None = None
    origin_uuid: str | None = None
    cc_version: str | None = None
    payload: Mapping[str, Any] | None = None


def turn_to_dict(turn: ContextTurn) -> dict[str, Any]:
    return {"role": turn.role, "text": turn.text, "tool_calls": list(turn.tool_calls)}


def turn_from_dict(data: Mapping[str, Any]) -> ContextTurn:
    return ContextTurn(role=data["role"], text=data["text"], tool_calls=tuple(data["tool_calls"]))
