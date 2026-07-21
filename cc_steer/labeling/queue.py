"""Build the typed label queue from the golden packet and the adjudication file.

The golden rows come from the authored blind packet under the frozen-eval root
(:func:`~cc_steer.retrain.judged.golden_dir`): its ``manifest.json`` supplies each
row's number, provenance key, and stratum, and its ``fires.jsonl`` sidecar supplies
the window text. The adjudication rows come from an ``out/disagreements.jsonl`` the
E44 lane produces; each line is one disagreement:

    {"row_id": str, "context": str, "category": str,
     "labels": {"<source>": "yes"|"no", ...},
     "detector"?: str, "session_id"?: str, "turn"?: int}

``row_id``, ``context`` (the rendered moment window), ``category`` (the disputed
steer-type), and ``labels`` (the splitting warrant judgments) are required; the
provenance keys are optional. The path is a parameter because the file lands after
this tool: a missing path yields a golden-only queue, and a given-but-absent path
raises rather than fabricating rows.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from athome.research.golden import MANIFEST_NAME

from cc_steer.labeling.models import FireWindow, LabelItem, LabelQueue
from cc_steer.retrain import judged

if TYPE_CHECKING:
    from pathlib import Path


def read_golden_items(golden_dir: Path) -> tuple[LabelItem, ...]:
    """The blind golden packet rendered as warrant-labeling queue rows, in row-number order."""
    contexts = _read_fires((golden_dir / judged.FIRES_NAME).read_text())
    return tuple(
        LabelItem(
            item_id=f"golden-{int(row['row']):04d}",
            source="golden",
            row_id=str(row["row_id"]),
            window=FireWindow(text=contexts[str(row["row_id"])]),
            row_number=int(row["row"]),
            stratum=str(row["stratum"]),
        )
        for row in sorted(json.loads((golden_dir / MANIFEST_NAME).read_text())["rows"], key=lambda r: int(r["row"]))
    )


def read_disagreements(path: Path) -> tuple[LabelItem, ...]:
    """The E44 disagreement rows rendered as adjudication queue rows, in file order."""
    return tuple(
        _adjudication_item(index, json.loads(line))
        for index, line in enumerate(line for line in path.read_text().splitlines() if line.strip())
    )


def build_queue(
    *, golden_dir: Path | None = None, disagreements_path: Path | None = None, root: Path | None = None
) -> LabelQueue:
    """Assemble the label queue: golden warrant rows first, then any adjudication rows.

    Args:
        golden_dir: The authored golden packet directory; defaults to the frozen-eval golden dir.
        disagreements_path: The E44 ``disagreements.jsonl``; omitted yields a golden-only queue.
        root: The frozen-eval root override the default golden dir resolves under.
    """
    return LabelQueue(
        items=read_golden_items(golden_dir or judged.golden_dir(root=root))
        + (read_disagreements(disagreements_path) if disagreements_path is not None else ())
    )


def _adjudication_item(index: int, row: dict) -> LabelItem:
    return LabelItem(
        item_id=f"adj-{index:04d}",
        source="adjudication",
        row_id=str(row["row_id"]),
        window=FireWindow(
            text=str(row["context"]),
            detector=row.get("detector"),
            session_id=row.get("session_id"),
            turn=row.get("turn"),
        ),
        disputed_category=str(row["category"]),
        candidate_labels=dict(row["labels"]),
    )


def _read_fires(text: str) -> dict[str, str]:
    return {
        record["row_id"]: record["context"]
        for record in (json.loads(line) for line in text.splitlines() if line.strip())
    }
