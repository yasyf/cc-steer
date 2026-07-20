"""The frozen promotion eval and its incumbent probability store.

Promotion compares a candidate against the incumbent on ONE eval that never moves.
:func:`freeze_eval` copies the exported watcher ``test.parquet`` into
``~/.cc-steer/eval/`` (``root`` overrides) as ``watcher_eval.parquet`` beside a
sha256 ``MANIFEST.json`` and refuses to overwrite changed content — frozen means
frozen. :class:`EvalFrame` loads it into the arrays the promotion gate reads: the
row ids, the fire labels, the corrective mask (a true steer that is not option
picking), the prose mask (not a QA event), and the render-v2 context tails the
watcher scores.

Each trained version's per-row ``P(NO_STEER)`` lands once in
``probs/<version>.json`` via :func:`write_probs` — the single writer. :func:`load_probs`
verifies the stored digest and row coverage against the frame and fails loud on any
mismatch: the incumbent is never rescored, so a stale or partial file is a bug, not
a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import numpy as np

from cc_steer.rendering import ask_block, gate_text_is_substantive, has_substantive_content
from cc_steer.retrain.data import DIRECTION, WatcherRow, dataset_digest
from cc_steer.triage import Category

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import pyarrow as pa

    from cc_steer.retrain.data import DatasetDigest

EVAL_DIR: Path = Path.home() / ".cc-steer" / "eval"
MANIFEST_NAME = "MANIFEST.json"
WATCHER_EVAL_NAME = "watcher_eval.parquet"
GATE_EVAL_NAME = "gate_eval.parquet"
STEER_TYPE_EVAL_NAME = "steer_type_eval.parquet"
PICK_EVAL_NAME = "pick_eval.parquet"
EVAL_NAMES: dict[str, str] = {
    "gate": GATE_EVAL_NAME,
    "watcher": WATCHER_EVAL_NAME,
    "steer_type": STEER_TYPE_EVAL_NAME,
    "pick": PICK_EVAL_NAME,
}
PROBS_DIRNAME = "probs"
QA_SOURCE_KIND = "question_answer"
RENDER_VERSION = 2
STEER_TYPE_CATEGORIES: tuple[str, ...] = get_args(Category)

VIEW_COLUMNS: dict[str, tuple[str, ...]] = {
    "gate": ("id", "text", "label", "kind", "offset_turns", "source_kind", "category", "session_id", "split"),
    "watcher": ("prompt", "completion", "verbatim", "label", "id", "category", "source_kind", "split"),
    "steer_type": ("id", "text", "category", "is_steering", "source_kind", "session_id", "split"),
    "pick": ("id", "text", "question", "options", "chosen_index", "n_options", "session_id", "split"),
}


class FrozenViolationError(RuntimeError):
    """A frozen eval file would change, is missing from the manifest, or fails verification."""


class SchemaError(ValueError):
    """A frozen eval table is missing required columns or carries duplicate row ids."""


class ProbsStoreError(RuntimeError):
    """A stored incumbent probability file is missing, stale, or does not cover the frame."""


class EmptyEvalContext(ValueError):
    """Eval rows whose rendered context has no substantive content — invalid to freeze."""

    def __init__(self, view: str, ids: Sequence[str]) -> None:
        self.view = view
        self.ids = tuple(ids)
        super().__init__(f"{view} eval has {len(self.ids)} rows with empty rendered context: {list(self.ids)}")


@dataclass(frozen=True, slots=True)
class EvalFrame:
    """The frozen watcher eval as the arrays the promotion gate reads.

    Attributes:
        ids: The row ids, in file order (probability arrays align to this order).
        labels: ``True`` for true-steer rows (should fire).
        corrective: ``label & category != "direction"`` — a true corrective steer.
        prose: ``source_kind != "question_answer"`` — not an option-picking QA event.
        tails: The render-v2 flattened context tail per row (the watcher's input).
        digest: The eval's order-invariant content digest, stamped into the probs store.
    """

    ids: tuple[str, ...]
    labels: np.ndarray
    corrective: np.ndarray
    prose: np.ndarray
    tails: tuple[str, ...]
    digest: DatasetDigest

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, *, root: Path | None = None) -> EvalFrame:
        """Build the frame from the frozen ``watcher_eval.parquet`` under ``root``."""
        table = load_frozen(root=root)
        rows = [WatcherRow.from_record(record) for record in table.to_pylist()]
        if duplicates := sorted(rid for rid, count in Counter(row.id for row in rows).items() if count > 1):
            raise SchemaError(f"frozen eval has duplicate row ids {duplicates}; ids must be unique")
        return cls(
            ids=tuple(row.id for row in rows),
            labels=np.array([row.label for row in rows], dtype=bool),
            corrective=np.array([row.label and row.category != DIRECTION for row in rows], dtype=bool),
            prose=np.array([row.source_kind != QA_SOURCE_KIND for row in rows], dtype=bool),
            tails=tuple(row.draft_text() for row in rows),
            digest=dataset_digest(table.to_pylist()),
        )


def eval_root(root: Path | None = None) -> Path:
    """The frozen-eval root: the parameter, env ``CC_STEER_EVAL``, or ``~/.cc-steer/eval``."""
    if root is not None:
        return root
    override = os.environ.get("CC_STEER_EVAL")
    return Path(override) if override else EVAL_DIR


def freeze_eval(view: str = "watcher", *, dataset_dir: Path | None = None, root: Path | None = None) -> str:
    """Copy the exported ``<view>/test.parquet`` into the eval root and merge its sha256 manifest.

    Freezes either the ``gate`` or the ``watcher`` eval into ``<view>_eval.parquet``,
    keeping the sibling view's manifest entry intact. Idempotent for identical bytes;
    raises :class:`FrozenViolationError` before writing anything when the frozen file
    exists with different content, and :class:`EmptyEvalContext` — naming the offending
    row ids — when any row's rendered context has no substantive content, so an invalid
    eval can never be frozen. Returns the frozen file's sha256.
    """
    import pyarrow.parquet as pq

    from cc_steer.retrain.data import DATASET_DIR

    source = (dataset_dir or DATASET_DIR) / view / "test.parquet"
    if not source.exists():
        raise FileNotFoundError(f"no {view} test parquet at {source}")
    _validate_columns(view, pq.read_schema(source).names)
    if empty := _empty_context_ids(view, pq.read_table(source)):
        raise EmptyEvalContext(view, empty)
    payload = source.read_bytes()
    sha = _sha256(payload)
    frozen = eval_root(root)
    destination = frozen / EVAL_NAMES[view]
    manifest = _manifest(frozen)
    if destination.exists() and _sha256(destination.read_bytes()) != sha:
        raise FrozenViolationError(f"{destination} is frozen with different content; refusing to overwrite it")
    if (frozen_sha := manifest.get(EVAL_NAMES[view])) is not None and frozen_sha != sha:
        raise FrozenViolationError(
            f"{EVAL_NAMES[view]} is frozen at {frozen_sha} in {frozen / MANIFEST_NAME} but the source hashes to {sha}; "
            "the frozen file is gone yet its manifest entry survives — refusing to refreeze drifted content"
        )
    frozen.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(payload)
    (frozen / MANIFEST_NAME).write_text(json.dumps(manifest | {EVAL_NAMES[view]: sha}, indent=2, sort_keys=True) + "\n")
    return sha


def load_frozen(view: str = "watcher", *, root: Path | None = None) -> pa.Table:
    """Load the frozen ``<view>`` eval, verifying its sha256 against the manifest."""
    import pyarrow.parquet as pq

    frozen = eval_root(root)
    name = EVAL_NAMES[view]
    expected = _manifest(frozen).get(name)
    if expected is None:
        raise FrozenViolationError(f"{name} is not in {frozen / MANIFEST_NAME}; run freeze_eval first")
    path = frozen / name
    if not path.exists():
        raise FrozenViolationError(f"{path} is in the manifest but missing on disk")
    actual = _sha256(path.read_bytes())
    if actual != expected:
        raise FrozenViolationError(f"{path} sha256 mismatch: manifest {expected}, file {actual}")
    _validate_columns(view, (table := pq.read_table(path)).column_names)
    return table


def probs_path(version: str, *, root: Path | None = None) -> Path:
    """The incumbent probability file for one registry version: ``probs/<version>.json``."""
    return eval_root(root) / PROBS_DIRNAME / f"{version}.json"


def write_probs(
    frame: EvalFrame,
    version: str,
    probs: Mapping[str, float],
    *,
    auc: float,
    render: int = RENDER_VERSION,
    root: Path | None = None,
) -> Path:
    """Write one version's per-row ``P(NO_STEER)`` — the single codepath that creates these files.

    ``probs`` must cover every frame row; the meta stamps the frame digest and the render
    the version was scored under (its own contract — a migrated incumbent may predate the
    frame's render) so :func:`load_probs` can refuse a stale file.
    """
    if missing := [row_id for row_id in frame.ids if row_id not in probs]:
        raise ProbsStoreError(f"probs for {version} miss {len(missing)} frame rows; refusing to write a partial file")
    if invalid := {row_id: probs[row_id] for row_id in frame.ids if not 0.0 <= float(probs[row_id]) <= 1.0}:
        raise ProbsStoreError(f"probs for {version} must be finite in [0, 1]; got {invalid}")
    path = probs_path(version, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"dataset_digest": frame.digest, "render": render, "auc": auc},
        "probs": {row_id: float(probs[row_id]) for row_id in frame.ids},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_probs(frame: EvalFrame, version: str, *, expected_render: int, root: Path | None = None) -> np.ndarray:
    """Load one version's stored probabilities aligned to ``frame.ids``; fail loud on any drift.

    Verifies the file exists, its render stamp matches ``expected_render`` (the render
    version stamped in that model's registry metadata — a migrated render-1 file stays
    loadable by passing 1), the stored digest matches the frame, and every frame row is
    present. Never rescores — a mismatch is a bug.
    """
    path = probs_path(version, root=root)
    if not path.exists():
        raise ProbsStoreError(f"no stored probs for {version} at {path}; score and write them first")
    payload = json.loads(path.read_text())
    if (stored := payload["meta"]["dataset_digest"]) != frame.digest:
        raise ProbsStoreError(f"{path} digest {stored} != frame digest {frame.digest}; the frozen eval moved")
    if (render := payload["meta"]["render"]) != expected_render:
        raise ProbsStoreError(f"{path} render {render} != expected {expected_render}; probs are from another render")
    probs = payload["probs"]
    if missing := [row_id for row_id in frame.ids if row_id not in probs]:
        raise ProbsStoreError(f"{path} misses {len(missing)} frame rows; it does not cover the eval")
    return np.array([float(probs[row_id]) for row_id in frame.ids], dtype=np.float64)


def _empty_context_ids(view: str, table: pa.Table) -> list[str]:
    match view:
        case "watcher":
            return sorted(
                row.id
                for record in table.to_pylist()
                if not has_substantive_content((row := WatcherRow.from_record(record)).prompt)
            )
        case "gate":
            return sorted(
                str(record["id"]) for record in table.to_pylist() if not gate_text_is_substantive(str(record["text"]))
            )
        case "steer_type":
            return sorted(
                str(record["id"]) for record in table.to_pylist() if not gate_text_is_substantive(str(record["text"]))
            )
        case "pick":
            return sorted(str(record["id"]) for record in table.to_pylist() if not str(record["text"]).strip())
    return []


def _validate_columns(view: str, columns: Sequence[str]) -> None:
    if missing := [col for col in VIEW_COLUMNS[view] if col not in columns]:
        raise SchemaError(f"{view} eval is missing columns {missing}; has {list(columns)}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(frozen: Path) -> dict[str, str]:
    path = frozen / MANIFEST_NAME
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()))


def _refuse_duplicate_ids(records: Sequence[Mapping[str, Any]]) -> None:
    if duplicates := sorted(rid for rid, count in Counter(str(r["id"]) for r in records).items() if count > 1):
        raise SchemaError(f"frozen eval has duplicate row ids {duplicates}; ids must be unique")


def steer_type_text(record: Mapping[str, Any]) -> str:
    """The steer-type classifier input: context, the agent action, and the user's steer, role-blocked.

    Unlike the watcher's window — which must never see the steer it predicts — the
    category classifier is told a steer happened and names its kind, so the user's
    message is part of the input, not a held-out label.
    """
    parts = [f"<{message['role']}>\n{message['content']}" for message in record["context"]]
    if action := record["agent_action"]:
        parts.append(f"<assistant>\n{action}")
    parts.append(f"<user>\n{record['user_message']}")
    return "\n\n".join(parts)


def build_steer_type_eval(*, dataset_dir: Path | None = None, seed: int = 1729) -> Path:
    """Build the frozen steer-type eval source from the exported traces test split.

    One row per judged feedback moment: :func:`steer_type_text` as ``text`` and the
    judge's eleven-way ``category`` as the label. Near-duplicate inputs collapse to
    one seeded representative each (:func:`~cc_steer.retrain.data.near_dup_indices`),
    then the test-split parquet ``freeze_eval("steer_type")`` freezes is written under
    ``<dataset_dir>/steer_type/test.parquet``. Returns the written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cc_steer.retrain.data import DATASET_DIR, near_dup_indices

    source = (dataset_dir or DATASET_DIR) / "traces" / "test.parquet"
    if not source.exists():
        raise FileNotFoundError(f"no traces test parquet at {source}")
    rows = [
        {
            "id": str(record["id"]),
            "text": steer_type_text(record),
            "category": str(record["category"]),
            "is_steering": bool(record["is_steering"]),
            "source_kind": str(record["source_kind"]),
            "session_id": str(record["session_id"]),
            "split": str(record["split"]),
        }
        for record in pq.read_table(source).to_pylist()
    ]
    kept, _ = near_dup_indices([row["text"] for row in rows], seed=seed)
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("category", pa.string()),
            ("is_steering", pa.bool_()),
            ("source_kind", pa.string()),
            ("session_id", pa.string()),
            ("split", pa.string()),
        ]
    )
    out = (dataset_dir or DATASET_DIR) / "steer_type" / "test.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([rows[i] for i in kept], schema=schema), out)
    return out


def build_pick_eval(*, decisions_path: Path | None = None, dataset_dir: Path | None = None, seed: int = 1729) -> Path:
    """Build the frozen pick-prediction eval source from the mined decisions dataset.

    One row per single-select, on-menu ``AskUserQuestion`` round in the test split:
    the rendered question and options (:func:`~cc_steer.rendering.ask_block`) as
    ``text`` and the user's chosen option index as ``chosen_index``. Multi-select and
    off-menu rounds are dropped — the label is a single option index, unrepresentable
    for them. Near-duplicate asks collapse to one seeded representative each, then the
    test-split parquet ``freeze_eval("pick")`` freezes is written under
    ``<dataset_dir>/pick/test.parquet``. Returns the written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cc_steer.decisions import DEFAULT_DECISIONS_PATH, read_decisions
    from cc_steer.retrain.data import DATASET_DIR, near_dup_indices

    rows_in, _digest, _quarantined = read_decisions(decisions_path or DEFAULT_DECISIONS_PATH)
    rows = [
        {
            "id": row.id,
            "text": ask_block(row.question, header=row.header or "", options=row.options),
            "question": row.question,
            "options": list(row.options),
            "chosen_index": row.chosen_index[0],
            "n_options": len(row.options),
            "session_id": row.session_id,
            "split": row.split,
        }
        for row in rows_in
        if row.split == "test"
        and not row.multi_select
        and not row.is_custom
        and len(row.chosen_index) == 1
        and len(row.options) >= 2
    ]
    kept, _ = near_dup_indices([row["text"] for row in rows], seed=seed)
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("question", pa.string()),
            ("options", pa.list_(pa.string())),
            ("chosen_index", pa.int64()),
            ("n_options", pa.int64()),
            ("session_id", pa.string()),
            ("split", pa.string()),
        ]
    )
    out = (dataset_dir or DATASET_DIR) / "pick" / "test.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([rows[i] for i in kept], schema=schema), out)
    return out


def freeze_steer_type(*, dataset_dir: Path | None = None, root: Path | None = None, seed: int = 1729) -> str:
    """Build then freeze the steer-type eval; returns the frozen file's sha256."""
    build_steer_type_eval(dataset_dir=dataset_dir, seed=seed)
    return freeze_eval("steer_type", dataset_dir=dataset_dir, root=root)


def freeze_pick(
    *, decisions_path: Path | None = None, dataset_dir: Path | None = None, root: Path | None = None, seed: int = 1729
) -> str:
    """Build then freeze the pick-prediction eval; returns the frozen file's sha256."""
    build_pick_eval(decisions_path=decisions_path, dataset_dir=dataset_dir, seed=seed)
    return freeze_eval("pick", dataset_dir=dataset_dir, root=root)


@dataclass(frozen=True, slots=True)
class SteerTypeFrame:
    """The frozen steer-type eval as the arrays a category classifier is scored on.

    Attributes:
        ids: The row ids, in file order.
        texts: The role-blocked classifier input per row.
        categories: The judge's eleven-way category label per row (one of
            :data:`STEER_TYPE_CATEGORIES`).
        digest: The eval's order-invariant content digest.
    """

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    categories: tuple[str, ...]
    digest: DatasetDigest

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, *, root: Path | None = None) -> SteerTypeFrame:
        """Build the frame from the frozen ``steer_type_eval.parquet`` under ``root``."""
        records = load_frozen("steer_type", root=root).to_pylist()
        _refuse_duplicate_ids(records)
        return cls(
            ids=tuple(str(record["id"]) for record in records),
            texts=tuple(str(record["text"]) for record in records),
            categories=tuple(str(record["category"]) for record in records),
            digest=dataset_digest(records),
        )


@dataclass(frozen=True, slots=True)
class PickFrame:
    """The frozen pick-prediction eval as the arrays an option classifier is scored on.

    Attributes:
        ids: The row ids, in file order.
        texts: The rendered ask — question and options — per row.
        chosen: The user's chosen option index per row.
        n_options: The number of options offered per row (the prediction's valid range).
        digest: The eval's order-invariant content digest.
    """

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    chosen: np.ndarray
    n_options: np.ndarray
    digest: DatasetDigest

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, *, root: Path | None = None) -> PickFrame:
        """Build the frame from the frozen ``pick_eval.parquet`` under ``root``."""
        records = load_frozen("pick", root=root).to_pylist()
        _refuse_duplicate_ids(records)
        return cls(
            ids=tuple(str(record["id"]) for record in records),
            texts=tuple(str(record["text"]) for record in records),
            chosen=np.array([int(record["chosen_index"]) for record in records], dtype=np.int64),
            n_options=np.array([int(record["n_options"]) for record in records], dtype=np.int64),
            digest=dataset_digest(records),
        )
