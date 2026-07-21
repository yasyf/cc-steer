"""The cc-present human-labeling tool: build a label queue, emit its board, ingest verdicts.

The main session drives the live cc-present channel; this package is everything else.
:func:`build_queue` assembles the golden watcher-fires rows and the E44 adjudication
rows into a typed :class:`LabelQueue`; :func:`render_document` turns that queue into
the board payload the session upserts; and :func:`record_event` folds each streamed
interaction event into the persisted labels — the golden ``labels.json`` the judged
gate reads and the richer ``human-gold-v1`` record.
"""

from __future__ import annotations

from cc_steer.labeling.blocks import item_card, render_document
from cc_steer.labeling.ingest import (
    LabelingError,
    human_gold_dir,
    read_journal,
    rebuild,
    record_event,
    reduce_journal,
)
from cc_steer.labeling.models import (
    CATEGORIES,
    FLAGS,
    FireWindow,
    LabelItem,
    LabelQueue,
    LabelSource,
    PersistedLabel,
)
from cc_steer.labeling.queue import build_queue, read_disagreements, read_golden_items

__all__ = [
    "CATEGORIES",
    "FLAGS",
    "FireWindow",
    "LabelItem",
    "LabelQueue",
    "LabelSource",
    "LabelingError",
    "PersistedLabel",
    "build_queue",
    "human_gold_dir",
    "item_card",
    "read_disagreements",
    "read_golden_items",
    "read_journal",
    "rebuild",
    "record_event",
    "reduce_journal",
    "render_document",
]
