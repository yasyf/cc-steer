from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from cc_pushback.llm import PromptMessage, classify_batch
from cc_pushback.matchers import matches
from cc_pushback.patterns import PATTERNS, TAXONOMY_VERSION, render_taxonomy
from cc_pushback.repo import MatchRow, now

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from cc_pushback.llm import LlmBackend, TModel
    from cc_pushback.matchers import PatternName
    from cc_pushback.models import ContextSnapshot, SourceKind
    from cc_pushback.repo import Repository

__all__ = [
    "MATCHER_BACKEND",
    "NO_MATCH",
    "PROMPT_VERSION",
    "ClassifyResponse",
    "FeedbackEvent",
    "cheap_pass",
    "llm_pass",
    "unclassified_events",
]

PROMPT_VERSION = "v1"
MATCHER_BACKEND = "matcher"
NO_MATCH = "none"
TEXT_CAP = 4096

SELECT_UNCLASSIFIED = """
SELECT id, source_kind, text, payload_json, context_json
FROM feedback_events e
WHERE NOT EXISTS (
  SELECT 1 FROM pattern_matches m
  WHERE m.feedback_id = e.id
    AND m.taxonomy_version = ?
    AND m.prompt_version = ?
    AND m.backend = ?
)
ORDER BY e.id
"""


class ClassifyResponse(BaseModel):
    """The structured classification a backend returns for one feedback event.

    Attributes:
        pattern_names: Every taxonomy pattern that fits, possibly empty.
        novel_pattern: A proposed kebab-case name when nothing in the taxonomy fits.
        severity: How strongly the developer pushed back.
        what_claude_did: One sentence describing the behavior that drew the pushback.
        rule: The corrective rule, phrased as the developer would.
    """

    model_config = ConfigDict(frozen=True)

    pattern_names: tuple[str, ...] = Field(description="Every taxonomy name that fits; may be empty.")
    novel_pattern: str | None = Field(
        default=None, description="A short kebab-case name when nothing in the taxonomy fits, else null."
    )
    severity: Literal["nit", "minor", "major", "blocking"] = Field(
        description="How strongly the developer pushes back."
    )
    what_claude_did: str = Field(description="One sentence on the behavior that drew the pushback.")
    rule: str = Field(description="The corrective rule in one imperative sentence.")


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """A persisted feedback event read back for classification.

    Attributes:
        id: The ``feedback_events`` primary key.
        source_kind: Which detector produced the event.
        text: The verbatim pushback text.
        context: The conversational window around the feedback.
        payload: The detector-specific metadata, when present.
    """

    id: int
    source_kind: SourceKind
    text: str
    context: ContextSnapshot
    payload: Mapping[str, Any] | None

    def render(self) -> str:
        return "\n\n".join(
            section
            for section in (
                f"What Claude did: {self.context.trigger.text}" if self.context.trigger else None,
                f"Developer said: {self.text[:TEXT_CAP]}",
            )
            if section
        )


def feedback_event(row: Mapping[str, Any]) -> FeedbackEvent:
    from cc_pushback.models import ContextSnapshot

    return FeedbackEvent(
        id=row["id"],
        source_kind=row["source_kind"],
        text=row["text"],
        context=ContextSnapshot.from_json(row["context_json"]),
        payload=json.loads(row["payload_json"]) if row["payload_json"] else None,
    )


def matcher_row(event: FeedbackEvent, name: PatternName) -> MatchRow:
    return MatchRow(
        feedback_id=event.id,
        pattern_name=name,
        backend=MATCHER_BACKEND,
        taxonomy_version=TAXONOMY_VERSION,
        prompt_version=PROMPT_VERSION,
        severity=None,
        what_claude_did=None,
        rule=None,
        novel=0,
        model=None,
        created_at=now(),
    )


def model_row(
    event: FeedbackEvent, response: ClassifyResponse, name: str, *, backend: str, model: str, novel: int
) -> MatchRow:
    return MatchRow(
        feedback_id=event.id,
        pattern_name=name,
        backend=backend,
        taxonomy_version=TAXONOMY_VERSION,
        prompt_version=PROMPT_VERSION,
        severity=response.severity,
        what_claude_did=response.what_claude_did,
        rule=response.rule,
        novel=novel,
        model=model,
        created_at=now(),
    )


def model_rows(event: FeedbackEvent, response: ClassifyResponse, *, backend: str, model: str) -> list[MatchRow]:
    named = [model_row(event, response, name, backend=backend, model=model, novel=0) for name in response.pattern_names]
    novel = (
        [model_row(event, response, response.novel_pattern, backend=backend, model=model, novel=1)]
        if response.novel_pattern
        else []
    )
    return named + novel or [model_row(event, response, NO_MATCH, backend=backend, model=model, novel=0)]


def event_prompt(event: FeedbackEvent) -> PromptMessage:
    return PromptMessage.load("classify").context("taxonomy", render_taxonomy()).context("event", event.render())


def unclassified_events(
    repo: Repository, *, taxonomy_version: str, prompt_version: str, backend: str, limit: int | None
) -> list[FeedbackEvent]:
    """Loads events with no language-model classification for the given version tuple.

    Args:
        repo: The repository to query.
        taxonomy_version: The taxonomy version the classification must cover.
        prompt_version: The prompt version the classification must cover.
        backend: The language-model backend whose rows count as classified.
        limit: The maximum number of events to load, or ``None`` for all.

    Returns:
        The unclassified events, oldest first.
    """
    query = SELECT_UNCLASSIFIED + ("\nLIMIT ?" if limit is not None else "")
    params = (taxonomy_version, prompt_version, backend, *((limit,) if limit is not None else ()))
    return [feedback_event(row) for row in repo.store.conn.execute(query, params)]


def cheap_pass(events: Sequence[FeedbackEvent]) -> list[MatchRow]:
    """Runs the LLM-free matcher pass, returning a match row per pattern hit.

    Matcher rows stamp the real :data:`TAXONOMY_VERSION` and :data:`PROMPT_VERSION`
    so they share the ``pattern_matches`` primary key uniformly with model rows;
    they carry no prompt semantics of their own.

    Args:
        events: The events to test against every taxonomy matcher.

    Returns:
        One ``backend='matcher'`` row per (event, matching pattern) pair.
    """
    return [
        matcher_row(event, pattern.name) for event in events for pattern in PATTERNS if matches(pattern.matcher, event)
    ]


async def llm_pass(
    events: Sequence[FeedbackEvent], *, backend: LlmBackend, backend_name: str, model: TModel
) -> list[MatchRow]:
    """Classifies every event with the language model, returning match rows.

    Args:
        events: The events to classify, one invocation each.
        backend: The CLI backend that runs the classifications.
        backend_name: The backend's persisted name, e.g. ``claude`` or ``codex``.
        model: The abstract model size to invoke.

    Returns:
        Named-pattern and novel-pattern rows across all events. An event the
        model matches to nothing gets a single ``pattern_name=NO_MATCH`` row so
        it counts as classified and is not re-sent on the next run.
    """
    responses = await classify_batch(backend, [event_prompt(event) for event in events], ClassifyResponse, model=model)
    return [
        row
        for event, response in zip(events, responses, strict=True)
        for row in model_rows(
            event, cast(ClassifyResponse, response), backend=backend_name, model=backend.models[model]
        )
    ]
