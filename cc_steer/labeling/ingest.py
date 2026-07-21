"""Persist human labels from cc-present interaction events, crash-safe and idempotent.

Every event the board streams back is appended verbatim to an append-only journal,
then the whole journal is reduced into the persisted artifacts — so a re-delivered or
replayed event is harmless and rebuilding from the same journal is a fixpoint. The
reduction is last-write-wins per control (a re-pick or a cleared verdict overrides),
recovering the row and the field from each event's ``blockId``.

Two artifact sets land under the frozen-eval root. The rich record — every touched
row with its warrant, category, flags, and note — is ``human-gold-v1/labels.jsonl``
beside a sha ``MANIFEST.json``, mirroring :mod:`cc_steer.retrain.evalset`'s freeze
manifest. The golden warrant verdicts additionally fill the packet's ``labels.json``
in the exact ``[{"row", "row_id", "label"}]`` shape
:func:`~cc_steer.retrain.judged.load_golden` binds against — written only once every
golden row is answered (and removed if a verdict is later cleared), so the judged gate
never sees a partial label set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

from cc_steer.labeling import models
from cc_steer.labeling.models import WARRANT_NO, WARRANT_YES, LabelItem, PersistedLabel, split_block_id
from cc_steer.retrain import judged
from cc_steer.retrain.evalset import eval_root

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cc_steer.labeling.models import LabelQueue

HUMAN_GOLD_LEAF: tuple[str, ...] = ("human-gold-v1",)
JOURNAL_NAME = "events.jsonl"
HUMAN_LABELS_NAME = "labels.jsonl"
HUMAN_MANIFEST_NAME = "MANIFEST.json"


class LabelingError(RuntimeError):
    """An interaction event carried a verdict or selection outside its control's contract."""


def human_gold_dir(*, root: Path | None = None) -> Path:
    """The ``human-gold-v1`` directory under the frozen-eval root."""
    return eval_root(root).joinpath(*HUMAN_GOLD_LEAF)


def record_event(
    event: Mapping[str, object],
    queue: LabelQueue,
    *,
    golden_dir: Path | None = None,
    root: Path | None = None,
) -> dict[str, PersistedLabel]:
    """Append one interaction event to the journal, then rebuild every artifact from it.

    Returns the freshly reduced ``item_id -> PersistedLabel`` map. Appending the same
    event twice is harmless: the reduction is last-write-wins, so the artifacts are
    unchanged.
    """
    directory = human_gold_dir(root=root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / JOURNAL_NAME).open("a") as journal:
        journal.write(json.dumps(event) + "\n")
    return rebuild(queue, golden_dir=golden_dir, root=root)


def rebuild(
    queue: LabelQueue, *, golden_dir: Path | None = None, root: Path | None = None
) -> dict[str, PersistedLabel]:
    """Reduce the whole journal against the queue and rewrite both artifact sets."""
    directory = human_gold_dir(root=root)
    reduced = reduce_journal(queue, read_journal(directory / JOURNAL_NAME))
    _write_human_gold(directory, queue, reduced)
    _write_golden_labels(golden_dir or judged.golden_dir(root=root), queue, reduced)
    return reduced


def read_journal(path: Path) -> tuple[dict, ...]:
    """The append-only event journal parsed into records, empty when it does not yet exist."""
    if not path.exists():
        return ()
    return tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def reduce_journal(queue: LabelQueue, events: Sequence[Mapping[str, object]]) -> dict[str, PersistedLabel]:
    """Fold the event stream into one persisted label per labeled row, last-write-wins.

    Only the four control events (warrant, category, flag, note) advance a row; a
    feedback thread, a submit, or any event for a block outside the queue is skipped,
    so a row appears exactly when a labeler acted on one of its controls.
    """
    by_id = {item.item_id: item for item in queue.items}
    labels: dict[str, PersistedLabel] = {}
    for event in events:
        item_id, control = split_block_id(str(event.get("blockId", "")))
        if (item := by_id.get(item_id)) is not None and (
            updated := _apply(labels.get(item_id, _seed(item)), control, event)
        ):
            labels[item_id] = updated
    return labels


def _seed(item: LabelItem) -> PersistedLabel:
    return PersistedLabel(item_id=item.item_id, source=item.source, row_id=item.row_id, row_number=item.row_number)


def _apply(current: PersistedLabel, control: str, event: Mapping[str, object]) -> PersistedLabel | None:
    match (control, event.get("type")):
        case (models.CONTROL_WARRANT, "decision.created"):
            return replace(current, warrant=_verdict_warrant(event.get("verdict")))
        case (models.CONTROL_CATEGORY, "choice.selected"):
            return replace(current, category=_single_option(event.get("optionIds")))
        case (models.CONTROL_FLAG, "choice.selected"):
            return replace(current, flags=tuple(str(option) for option in _options(event.get("optionIds"))))
        case (models.CONTROL_NOTE, "input.submitted"):
            return replace(current, note=str(event.get("text", "")) or None)
        case _:
            return None


def _verdict_warrant(verdict: object) -> bool | None:
    match verdict:
        case "approved":
            return True
        case "rejected":
            return False
        case "cleared":
            return None
        case other:
            raise LabelingError(f"warrant approval verdict {other!r} is not approved/rejected/cleared")


def _single_option(option_ids: object) -> str | None:
    match _options(option_ids):
        case [option]:
            return str(option)
        case []:
            return None
        case many:
            raise LabelingError(f"category is single-select but the selection was {many!r}")


def _options(option_ids: object) -> list[object]:
    match option_ids:
        case list() as options:
            return options
        case other:
            raise LabelingError(f"choice event optionIds must be a list, got {other!r}")


def _write_human_gold(directory: Path, queue: LabelQueue, reduced: Mapping[str, PersistedLabel]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(_human_line(reduced[item.item_id])) + "\n" for item in queue.items if item.item_id in reduced
    )
    (directory / HUMAN_LABELS_NAME).write_text(payload)
    manifest = {HUMAN_LABELS_NAME: _sha256(payload.encode()), "n": payload.count("\n")}
    (directory / HUMAN_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _human_line(label: PersistedLabel) -> dict:
    return {
        "item_id": label.item_id,
        "row_id": label.row_id,
        "source": label.source,
        "row_number": label.row_number,
        "warrant": label.warrant,
        "category": label.category,
        "flags": list(label.flags),
        "note": label.note,
    }


def _write_golden_labels(golden_dir: Path, queue: LabelQueue, reduced: Mapping[str, PersistedLabel]) -> None:
    path = golden_dir / judged.LABELS_NAME
    if (entries := _golden_entries(queue, reduced)) is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps(entries, indent=2) + "\n")


def _golden_entries(queue: LabelQueue, reduced: Mapping[str, PersistedLabel]) -> list[dict] | None:
    rows = sorted(queue.golden, key=lambda item: item.row_number or 0)
    if not rows or any((label := reduced.get(item.item_id)) is None or label.warrant is None for item in rows):
        return None
    return [
        {
            "row": item.row_number,
            "row_id": item.row_id,
            "label": WARRANT_YES if reduced[item.item_id].warrant else WARRANT_NO,
        }
        for item in rows
    ]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
