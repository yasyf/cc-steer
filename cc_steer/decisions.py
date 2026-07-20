"""Mine ``AskUserQuestion`` decision rounds from Claude Code transcript trees.

Every ``AskUserQuestion`` tool call opens one or more decision rounds — a
question, its short header, the offered options, and the user's pick. This module
walks a transcript tree, lifts each round from cc-transcript's typed
:class:`~cc_transcript.tools.AskUserQuestionResult`, and records one
:class:`DecisionRow` per round.

The typed result carries the rounds (:attr:`~cc_transcript.tools.AskUserQuestionResult.questions`)
and a question-text→answer map. A single-select answer is one option label or a
free-typed custom string; a multi-select answer is the picked labels joined with
``", "``, with any custom text appended. :func:`selection` recovers the chosen
option indices by matching the answer against the offered labels and flags a round
:attr:`~DecisionRow.is_custom` when the user went off-menu. The raw answer string
is preserved verbatim on :attr:`~DecisionRow.answer`, so no pick is ever lossy.

A round whose result predates the typed lift — an older transcript, a rejected or
never-answered call — surfaces not as an
:class:`~cc_transcript.tools.AskUserQuestionResult` but a
:class:`~cc_transcript.tools.TextResult`, an
:class:`~cc_transcript.tools.OtherResult`, or nothing. Those uses are counted and
quarantined loudly as :class:`Quarantine` records, never silently skipped.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
from cc_transcript.activity import SessionActivity
from cc_transcript.discovery import discover
from cc_transcript.ids import SessionId
from cc_transcript.parser import stream
from cc_transcript.tools import AskUserQuestionResult

from cc_steer.rendering import split_of
from cc_steer.retrain.data import DATASET_DIR, DatasetDigest, dataset_digest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from cc_transcript.activity import ToolUse
    from cc_transcript.ids import EventRef
    from cc_transcript.models import TranscriptEvent

ASK_TOOL = "AskUserQuestion"
DEFAULT_DECISIONS_PATH = DATASET_DIR / "decisions.parquet"
ID_CHARS = 16
DIGEST_KEY = b"dataset_digest"
QUARANTINED_KEY = b"quarantined"

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("session_id", pa.string()),
        ("occurred_at", pa.string()),
        ("turn_index", pa.int64()),
        ("event_uuid", pa.string()),
        ("tool_use_id", pa.string()),
        ("question", pa.string()),
        ("header", pa.string()),
        ("options", pa.list_(pa.string())),
        ("multi_select", pa.bool_()),
        ("answer", pa.string()),
        ("chosen_index", pa.list_(pa.int64())),
        ("is_custom", pa.bool_()),
        ("split", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One ``AskUserQuestion`` round: the question, its options, and the user's pick.

    Attributes:
        id: The round's stable content id (session, event, tool use, and round
            index hashed), the parquet dedup key.
        session_id: The session the round was mined from.
        occurred_at: ISO-8601 timestamp of the assistant turn that asked.
        turn_index: The turn the round fired in — the anchor into the session.
        event_uuid: The uuid of the asking assistant event, the precise anchor.
        tool_use_id: The ``AskUserQuestion`` tool-use block id, or None.
        question: The prompt text shown to the user.
        header: The round's short header, or None when the ask omitted one.
        options: The offered option labels, in presentation order, verbatim
            (a trailing ``" (Recommended)"`` is kept).
        multi_select: Whether the round accepted more than one selection.
        answer: The user's pick, verbatim — the raw answer string the platform
            recorded (joined labels and/or custom text), or None when unanswered.
        chosen_index: The ``options`` indices the user selected, in pick order;
            empty when the pick was entirely off-menu or the round was unanswered.
        is_custom: Whether the answer carried free-typed text beyond the offered
            options — the user went off-menu.
        split: The deterministic session-hash split (``train``/``test``).
    """

    id: str
    session_id: str
    occurred_at: str
    turn_index: int
    event_uuid: str
    tool_use_id: str | None
    question: str
    header: str | None
    options: tuple[str, ...]
    multi_select: bool
    answer: str | None
    chosen_index: tuple[int, ...]
    is_custom: bool
    split: str

    def to_record(self) -> dict[str, object]:
        """The row as an Arrow-writable mapping, tuples lowered to lists."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "occurred_at": self.occurred_at,
            "turn_index": self.turn_index,
            "event_uuid": self.event_uuid,
            "tool_use_id": self.tool_use_id,
            "question": self.question,
            "header": self.header,
            "options": list(self.options),
            "multi_select": self.multi_select,
            "answer": self.answer,
            "chosen_index": list(self.chosen_index),
            "is_custom": self.is_custom,
            "split": self.split,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DecisionRow:
        """Rebuild a row from one decisions-parquet record."""
        return cls(
            id=str(record["id"]),
            session_id=str(record["session_id"]),
            occurred_at=str(record["occurred_at"]),
            turn_index=int(record["turn_index"]),
            event_uuid=str(record["event_uuid"]),
            tool_use_id=None if record["tool_use_id"] is None else str(record["tool_use_id"]),
            question=str(record["question"]),
            header=None if record["header"] is None else str(record["header"]),
            options=tuple(str(option) for option in record["options"]),
            multi_select=bool(record["multi_select"]),
            answer=None if record["answer"] is None else str(record["answer"]),
            chosen_index=tuple(int(index) for index in record["chosen_index"]),
            is_custom=bool(record["is_custom"]),
            split=str(record["split"]),
        )


@dataclass(frozen=True, slots=True)
class Quarantine:
    """An ``AskUserQuestion`` use whose result carries no typed rounds.

    Attributes:
        session_id: The session the use was mined from.
        event_uuid: The uuid of the asking assistant event.
        tool_use_id: The ``AskUserQuestion`` tool-use block id, or None.
        result_type: The result type that displaced the typed round, e.g.
            ``TextResult`` (a rejection), ``OtherResult``, or ``NoneType`` (never
            answered).
    """

    session_id: str
    event_uuid: str
    tool_use_id: str | None
    result_type: str


@dataclass(frozen=True, slots=True)
class MineResult:
    """The mined decision rows and the quarantined, un-typed uses.

    Attributes:
        rows: One :class:`DecisionRow` per typed decision round.
        quarantined: One :class:`Quarantine` per ``AskUserQuestion`` use whose
            result predates the typed lift.
    """

    rows: tuple[DecisionRow, ...]
    quarantined: tuple[Quarantine, ...]


@dataclass(frozen=True, slots=True)
class DecisionStats:
    """Aggregate counts over a decisions dataset.

    Attributes:
        total: The number of mined decision rounds.
        by_split: Round counts keyed by split, descending.
        multi_select: Rounds that accepted more than one selection.
        custom: Rounds whose answer went off-menu.
        quarantined: ``AskUserQuestion`` uses that carried no typed rounds.
    """

    total: int
    by_split: Mapping[str, int]
    multi_select: int
    custom: int
    quarantined: int

    def render(self) -> str:
        """The human-readable summary the ``decisions stats`` command prints."""
        return "\n".join(
            [
                f"total: {self.total}",
                *(f"  {split}: {count}" for split, count in self.by_split.items()),
                f"multi-select: {self.multi_select} ({self._share(self.multi_select)})",
                f"custom: {self.custom} ({self._share(self.custom)})",
                f"quarantined: {self.quarantined}",
            ]
        )

    def to_dict(self) -> dict[str, object]:
        """Serializes the stats to a JSON-ready dictionary."""
        return {
            "total": self.total,
            "by_split": dict(self.by_split),
            "multi_select": self.multi_select,
            "custom": self.custom,
            "quarantined": self.quarantined,
        }

    def _share(self, count: int) -> str:
        return f"{count / self.total:.0%}" if self.total else "0%"


def selection(answer: str | None, options: Sequence[str], *, multi_select: bool) -> tuple[tuple[int, ...], bool]:
    """The chosen option indices and whether the pick went off-menu.

    A single-select answer is matched whole against the offered labels. A
    multi-select answer is the picked labels joined with ``", "`` with any custom
    text appended, so it is recovered by greedily consuming the longest matching
    label at each position — a label that itself contains ``", "`` stays intact.
    The pick is custom only when a residue remains that is not a clean
    concatenation of known labels. A None answer (unanswered) and an empty
    multi-select answer (no selection) both yield no indices and are not custom.
    """
    if answer is None:
        return (), False
    index = {label: position for position, label in enumerate(options)}
    if not multi_select:
        return ((index[answer],), False) if answer in index else ((), True)
    ordered = sorted(index, key=len, reverse=True)
    picks: list[int] = []
    remaining = answer
    while remaining:
        if (label := next((opt for opt in ordered if remaining == opt or remaining.startswith(f"{opt}, ")), None)) is None:
            return tuple(picks), True
        picks.append(index[label])
        remaining = remaining[len(label) :].removeprefix(", ")
    return tuple(picks), False


def round_id(ref: EventRef, round_index: int) -> str:
    """A stable content id for one round of an ``AskUserQuestion`` use."""
    key = f"{ref.session_id}:{ref.event_uuid}:{ref.tool_use_id}:{round_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:ID_CHARS]


def rounds_of(use: ToolUse, result: AskUserQuestionResult) -> Iterator[DecisionRow]:
    """Every decision round of one typed ``AskUserQuestion`` use."""
    session_id = str(use.ref.session_id)
    split = split_of(session_id)
    for round_index, question in enumerate(result.questions):
        answer = result.answers.get(question.question)
        chosen_index, is_custom = selection(answer, question.labels, multi_select=question.multi_select)
        yield DecisionRow(
            id=round_id(use.ref, round_index),
            session_id=session_id,
            occurred_at=use.ts.isoformat(),
            turn_index=use.turn_index,
            event_uuid=str(use.ref.event_uuid),
            tool_use_id=None if use.ref.tool_use_id is None else str(use.ref.tool_use_id),
            question=question.question,
            header=question.header,
            options=tuple(question.labels),
            multi_select=question.multi_select,
            answer=answer,
            chosen_index=chosen_index,
            is_custom=is_custom,
            split=split,
        )


def mine_events(session_id: SessionId, events: Sequence[TranscriptEvent]) -> MineResult:
    """Mine one session's events into decision rows and quarantined uses."""
    rows: list[DecisionRow] = []
    quarantined: list[Quarantine] = []
    for turn in SessionActivity.from_events(session_id, events).turns:
        for use in turn.tool_uses:
            if use.call.name != ASK_TOOL:
                continue
            match use.typed_result:
                case AskUserQuestionResult() as result:
                    rows.extend(rounds_of(use, result))
                case other:
                    quarantined.append(
                        Quarantine(
                            session_id=str(session_id),
                            event_uuid=str(use.ref.event_uuid),
                            tool_use_id=None if use.ref.tool_use_id is None else str(use.ref.tool_use_id),
                            result_type=type(other).__name__,
                        )
                    )
    return MineResult(rows=tuple(rows), quarantined=tuple(quarantined))


def mine(root: Path) -> MineResult:
    """Mine every ``AskUserQuestion`` decision round under ``root``.

    Walks the transcript tree at ``root`` — one ``*.jsonl`` per session — and lifts
    every typed round into a :class:`DecisionRow`, keyed by the transcript's stem as
    the session id. Uses whose result predates the typed lift are quarantined, never
    dropped.

    Args:
        root: The transcript directory to mine, for example a corpus mirror or
            ``~/.claude/projects``.

    Returns:
        The mined rows and the quarantined uses.
    """
    rows: list[DecisionRow] = []
    quarantined: list[Quarantine] = []
    for transcript in stream(discover(root)):
        if transcript.path is None:
            continue
        result = mine_events(SessionId(transcript.path.stem), transcript.events)
        rows.extend(result.rows)
        quarantined.extend(result.quarantined)
    return MineResult(rows=tuple(rows), quarantined=tuple(quarantined))


def write_decisions(result: MineResult, out: Path) -> DatasetDigest:
    """Write the mined rows to ``out`` as parquet, stamping the digest and quarantine count.

    The order-invariant :func:`~cc_steer.retrain.data.dataset_digest` and the
    quarantine count ride in the parquet schema metadata, so a later
    :func:`read_decisions` reads the receipt back off the file itself.

    Returns:
        The dataset digest stamped into the file.
    """
    records = [row.to_record() for row in result.rows]
    digest = dataset_digest(records)
    table = pa.Table.from_pylist(records, schema=SCHEMA).replace_schema_metadata(
        {DIGEST_KEY: digest, QUARANTINED_KEY: str(len(result.quarantined))}
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return digest


def read_decisions(path: Path) -> tuple[list[DecisionRow], DatasetDigest, int]:
    """Read a decisions parquet back into rows, its digest, and its quarantine count."""
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    return (
        [DecisionRow.from_record(record) for record in table.to_pylist()],
        DatasetDigest(metadata[DIGEST_KEY].decode()),
        int(metadata[QUARANTINED_KEY].decode()),
    )


def stats_of(rows: Sequence[DecisionRow], *, quarantined: int) -> DecisionStats:
    """Aggregate mined rows into the counts the ``decisions stats`` command prints."""
    return DecisionStats(
        total=len(rows),
        by_split=dict(Counter(row.split for row in rows).most_common()),
        multi_select=sum(row.multi_select for row in rows),
        custom=sum(row.is_custom for row in rows),
        quarantined=quarantined,
    )
