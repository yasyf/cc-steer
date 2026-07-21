"""Emit the cc-present block payloads the main session upserts for a label queue.

This module is data-only: it turns a :class:`~cc_steer.labeling.models.LabelQueue`
into the JSON document envelope and per-row blocks the cc-present board renders, and
makes zero MCP or network calls — the live channel is driven from the main session.
Each row becomes one ``card``: the fire window as a ``code`` block, an ``approval``
for the warrant verdict, a single-select ``choice`` over the eleven steer-type
categories, a multi-select ``choice`` for the hard/ambiguous flags, and an ``input``
for a free-text note. Every control's block id is ``{item_id}-{control}``, so the
ingest reducer recovers the row and the field from the event's ``blockId`` alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cc_steer.labeling.models import (
    CATEGORIES,
    CONTROL_CATEGORY,
    CONTROL_DISPUTE,
    CONTROL_FLAG,
    CONTROL_NOTE,
    CONTROL_WARRANT,
    CONTROL_WINDOW,
    FLAG_AMBIGUOUS,
    FLAG_HARD,
    block_id,
)

if TYPE_CHECKING:
    from cc_steer.labeling.models import LabelItem, LabelQueue

DOCUMENT_TITLE = "Watcher-fires human labeling"
DOCUMENT_INTRO = (
    "Label each row from the situation alone — the window is exactly what the watcher saw at one "
    "decision point. Approve means steering **was** warranted there; reject means it was not."
)
SUBMIT_LABEL = "Submit labels"
SUBMIT_NOTE = "Submitting records every answered row; the golden gate reads only the warrant verdicts."

WARRANT_PROMPT = "Was steering the assistant warranted at this point? (Approve = yes, Reject = no)"
CATEGORY_PROMPT = "Which steer-type category fits? (adjudication rows: pick the correct one)"
FLAG_PROMPT = "Flag this row for review?"
NOTE_LABEL = "Note (optional)"

GOLDEN_SECTION_ID = "sec-golden"
GOLDEN_SECTION_TITLE = "Golden watcher-fires — warrant labels"
GOLDEN_SECTION_MD = "Blind rows. Judge the merit of intervening here, never the outcome."
ADJUDICATION_SECTION_ID = "sec-adjudication"
ADJUDICATION_SECTION_TITLE = "Disagreement adjudication"
ADJUDICATION_SECTION_MD = (
    "Rows where the judges split. Set the warrant and the category the panel should have agreed on."
)


def render_document(queue: LabelQueue, *, title: str = DOCUMENT_TITLE) -> dict:
    """The full cc-present document envelope for a queue: two sections, one card per row.

    The returned dict is a data-only board the main session pushes; it carries a stats
    row, a submit bar, and — per populated population — a section header followed by
    every row's card.
    """
    return {
        "version": 1,
        "title": title,
        "intro": DOCUMENT_INTRO,
        "stats": [
            {"label": "golden rows", "value": str(len(queue.golden))},
            {"label": "adjudication rows", "value": str(len(queue.adjudication))},
        ],
        "submit": {"label": SUBMIT_LABEL, "note": SUBMIT_NOTE},
        "blocks": [
            *_population(GOLDEN_SECTION_ID, GOLDEN_SECTION_TITLE, GOLDEN_SECTION_MD, queue.golden),
            *_population(
                ADJUDICATION_SECTION_ID, ADJUDICATION_SECTION_TITLE, ADJUDICATION_SECTION_MD, queue.adjudication
            ),
        ],
    }


def item_card(item: LabelItem) -> dict:
    """The ``card`` block for one queue row — the fire window and every label control.

    A card nests, in order: the window (plus the judge dispute for adjudication rows),
    the warrant ``approval``, the category ``choice``, the flag ``choice``, and the
    note ``input``. The card decides by its own controls, so several decidables coexist.
    """
    return {
        "id": item.item_id,
        "type": "card",
        "title": _card_title(item),
        "summary": _card_summary(item),
        "chips": _chips(item),
        "flagged": item.source == "adjudication",
        "status": "open",
        "children": [
            _window_block(item),
            *([_dispute_block(item)] if item.source == "adjudication" else []),
            _warrant_block(item),
            _category_block(item),
            _flag_block(item),
            _note_block(item),
        ],
    }


def _population(section_id: str, title: str, md: str, items: tuple[LabelItem, ...]) -> list[dict]:
    if not items:
        return []
    return [{"id": section_id, "type": "section", "title": title, "md": md}, *(item_card(item) for item in items)]


def _card_title(item: LabelItem) -> str:
    match item.source:
        case "golden":
            return f"Row {item.row_number} · {item.stratum}"
        case "adjudication":
            return f"Adjudicate · disputed {item.disputed_category}"


def _card_summary(item: LabelItem) -> str:
    match item.source:
        case "golden":
            return "Blind warrant label."
        case "adjudication":
            return "Judges split: " + ", ".join(f"{source}={label}" for source, label in item.candidate_labels.items())


def _chips(item: LabelItem) -> list[dict]:
    return [
        {"label": item.source},
        *([{"label": item.stratum}] if item.stratum else []),
        *([{"label": "split", "tone": "flag"}] if item.source == "adjudication" else []),
    ]


def _window_block(item: LabelItem) -> dict:
    return {
        "id": block_id(item.item_id, CONTROL_WINDOW),
        "type": "code",
        "lang": "text",
        "title": _window_title(item),
        "code": item.window.text,
    }


def _window_title(item: LabelItem) -> str:
    parts = [
        *([item.window.detector] if item.window.detector else []),
        *([item.window.session_id] if item.window.session_id else []),
        *([f"turn {item.window.turn}"] if item.window.turn is not None else []),
    ]
    return " · ".join(parts) if parts else "Fire window"


def _dispute_block(item: LabelItem) -> dict:
    return {
        "id": block_id(item.item_id, CONTROL_DISPUTE),
        "type": "record",
        "title": "Judge disagreement",
        "facts": [
            {"label": "disputed category", "value": item.disputed_category or "—"},
            *({"label": source, "value": label} for source, label in item.candidate_labels.items()),
        ],
    }


def _warrant_block(item: LabelItem) -> dict:
    return {"id": block_id(item.item_id, CONTROL_WARRANT), "type": "approval", "prompt": WARRANT_PROMPT}


def _category_block(item: LabelItem) -> dict:
    return {
        "id": block_id(item.item_id, CONTROL_CATEGORY),
        "type": "choice",
        "prompt": CATEGORY_PROMPT,
        "multi": False,
        "options": [{"id": category, "label": category.replace("_", " ")} for category in CATEGORIES],
    }


def _flag_block(item: LabelItem) -> dict:
    return {
        "id": block_id(item.item_id, CONTROL_FLAG),
        "type": "choice",
        "prompt": FLAG_PROMPT,
        "multi": True,
        "options": [
            {"id": FLAG_HARD, "label": "Hard", "hint": "genuinely difficult to call"},
            {"id": FLAG_AMBIGUOUS, "label": "Ambiguous", "hint": "under-determined by the window"},
        ],
    }


def _note_block(item: LabelItem) -> dict:
    return {
        "id": block_id(item.item_id, CONTROL_NOTE),
        "type": "input",
        "label": NOTE_LABEL,
        "placeholder": "Why this call, or what the window is missing",
        "multiline": True,
    }
