"""Typed queue and label records for the cc-present human-labeling tool.

The tool routes two row populations through one board: the blind golden watcher-fires
packet (whose ``labels.json`` gates the judged promotion panel) and a disagreement
adjudication queue (rows where the medium judge, the fable relabel, and the opus
auditor split). Every row renders as one :class:`LabelItem` carrying the fire window,
its provenance, and the controls that decide it; a filled control reduces to one
:class:`PersistedLabel`. Both are frozen so a queue is a value the block emitter and
the ingest reducer can share without either mutating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from cc_steer.triage import Category

if TYPE_CHECKING:
    from collections.abc import Mapping

LabelSource = Literal["golden", "adjudication"]

CATEGORIES: tuple[str, ...] = get_args(Category)
WARRANT_YES = "yes"
WARRANT_NO = "no"
FLAG_HARD = "hard"
FLAG_AMBIGUOUS = "ambiguous"
FLAGS: tuple[str, ...] = (FLAG_HARD, FLAG_AMBIGUOUS)

CONTROL_WINDOW = "window"
CONTROL_DISPUTE = "dispute"
CONTROL_WARRANT = "warrant"
CONTROL_CATEGORY = "category"
CONTROL_FLAG = "flag"
CONTROL_NOTE = "note"


def block_id(item_id: str, control: str) -> str:
    return f"{item_id}-{control}"


def split_block_id(raw: str) -> tuple[str, str]:
    item_id, _, control = raw.rpartition("-")
    return item_id, control


@dataclass(frozen=True, slots=True)
class FireWindow:
    """The rendered fire window a labeler judges, plus what provenance survives to it.

    Attributes:
        text: The context window exactly as the watcher saw it at the moment.
        detector: The detector kind that fired, when known; the blind golden packet withholds it.
        session_id: The originating session id, when known.
        turn: The moment's turn offset within the session, when known.
    """

    text: str
    detector: str | None = None
    session_id: str | None = None
    turn: int | None = None


@dataclass(frozen=True, slots=True)
class LabelItem:
    """One queue row to label: the fire window plus the controls that decide it.

    Attributes:
        item_id: The globally unique, kebab-case id seeding this row's block ids.
        source: Which population the row came from — ``golden`` or ``adjudication``.
        row_id: The opaque provenance key the persisted label is keyed on.
        window: The rendered fire window and its provenance.
        row_number: The golden packet's 1-based row number; ``None`` for adjudication rows.
        stratum: The golden sampling stratum; ``None`` for adjudication rows.
        disputed_category: The category the judges split on; ``None`` for golden rows.
        candidate_labels: The disagreeing warrant judgments, source name to ``yes``/``no``.
    """

    item_id: str
    source: LabelSource
    row_id: str
    window: FireWindow
    row_number: int | None = None
    stratum: str | None = None
    disputed_category: str | None = None
    candidate_labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LabelQueue:
    """The full labeling queue: golden warrant rows followed by adjudication rows.

    Attributes:
        items: The queue rows in presentation order.
    """

    items: tuple[LabelItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def golden(self) -> tuple[LabelItem, ...]:
        return tuple(item for item in self.items if item.source == "golden")

    @property
    def adjudication(self) -> tuple[LabelItem, ...]:
        return tuple(item for item in self.items if item.source == "adjudication")


@dataclass(frozen=True, slots=True)
class PersistedLabel:
    """One row's reduced human label — the fixpoint of every interaction event on it.

    Attributes:
        item_id: The queue row's block-id stem.
        source: The population the row came from.
        row_id: The provenance key the label is keyed on.
        row_number: The golden packet row number, carried through to ``labels.json``.
        warrant: Whether the human judged steering warranted; ``None`` until answered or once cleared.
        category: The human's steer-type category pick; ``None`` when unset.
        flags: The hard/ambiguous flags the human raised.
        note: The human's free-text note; ``None`` when unset.
    """

    item_id: str
    source: LabelSource
    row_id: str
    row_number: int | None = None
    warrant: bool | None = None
    category: str | None = None
    flags: tuple[str, ...] = ()
    note: str | None = None

    @property
    def complete(self) -> bool:
        return self.warrant is not None
