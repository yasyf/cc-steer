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
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, get_args

import numpy as np

from cc_steer.instrument import PairedDeLong, actionable, mde, paired_delong
from cc_steer.rendering import ask_block, gate_text_is_substantive, has_substantive_content
from cc_steer.retrain.data import DIRECTION, SEMANTIC_THRESHOLD, WatcherRow, dataset_digest, exact_text_overlap
from cc_steer.triage import Category

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal

    import pyarrow as pa

    from cc_steer.retrain.data import DatasetDigest, DedupStats, Embedder

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
COMPARISONS_DIRNAME = "comparisons"
QA_SOURCE_KIND = "question_answer"
RENDER_VERSION = 2
STEER_TYPE_CATEGORIES: tuple[str, ...] = get_args(Category)
DEDUP_SIDECAR = "dedup.json"
META_SUFFIX = ".meta"
# Views where an exact cross-split text overlap is a hard freeze refusal (the E43 leak locus). Only
# the watcher's long, unique context windows: short gate/steer_type/pick texts can collide benignly.
DISJOINT_ENFORCED_VIEWS: tuple[str, ...] = ("watcher",)

# Rebuild-frame label provenance, most authoritative first. A ``medium_judge`` label is guidance
# only — a tie-breaker/feature — and never the sole label a candidate row is admitted on.
PROVENANCE_PRECEDENCE: tuple[str, ...] = ("human", "fable", "medium_judge")
GUIDANCE_PROVENANCE = "medium_judge"
TARGET_MDE = 0.02
# Projection defaults: the incumbent watcher's sanity-gate AUC and a paired correlation in the
# instrument card's measured band (paired MDE ~0.017-0.024), so the projection matches the card.
PROJECTION_AUC = 0.93
PROJECTION_RHO = 0.8

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


class SplitLeakError(RuntimeError):
    """A test-split eval row whose exact text also appears in the train split — the E43 leak."""

    def __init__(self, view: str, ids: Sequence[str]) -> None:
        self.view = view
        self.ids = tuple(ids)
        super().__init__(
            f"{view} eval leaks {len(self.ids)} row(s) whose exact text also appears in the train split: "
            f"{list(self.ids)}; freeze refuses a train/eval overlap"
        )


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


class ProbsFrame(Protocol):
    """The frame surface :func:`write_probs` persists: the row ids and the order-invariant digest.

    Both :class:`EvalFrame` and :class:`~cc_steer.retrain.encoder.EncoderFrame` satisfy it, so an
    encoder arm's per-row probs land through the same single writer as the lexical gate's, keyed
    under the same digest and comparable on the same frame.
    """

    @property
    def ids(self) -> tuple[str, ...]: ...

    @property
    def digest(self) -> DatasetDigest: ...


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
    exists with different content, :class:`EmptyEvalContext` — naming the offending
    row ids — when any row's rendered context has no substantive content, and
    :class:`SplitLeakError` when a sibling ``train.parquet`` shares any row's exact text
    with the eval (the E43 train/eval leak), so an invalid eval can never be frozen.
    Any dedup sidecar the eval builder wrote and the disjointness check's counts merge
    into the manifest under ``<name>.meta``. Returns the frozen file's sha256.
    """
    import pyarrow.parquet as pq

    from cc_steer.retrain.data import DATASET_DIR

    source = (dataset_dir or DATASET_DIR) / view / "test.parquet"
    if not source.exists():
        raise FileNotFoundError(f"no {view} test parquet at {source}")
    _validate_columns(view, pq.read_schema(source).names)
    if empty := _empty_context_ids(view, pq.read_table(source)):
        raise EmptyEvalContext(view, empty)
    meta = _disjointness_meta(view, source)
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
    entries = {EVAL_NAMES[view]: sha} | ({f"{EVAL_NAMES[view]}{META_SUFFIX}": meta} if meta else {})
    (frozen / MANIFEST_NAME).write_text(json.dumps(manifest | entries, indent=2, sort_keys=True) + "\n")
    return sha


def _disjointness_meta(view: str, source: Path) -> dict[str, Any]:
    """The exact cross-split disjointness check plus any dedup sidecar, for the freeze manifest.

    When a sibling ``train.parquet`` exists, computes the test rows whose exact text also appears in
    train. For an enforced view (:data:`DISJOINT_ENFORCED_VIEWS` — the watcher frame, the E43 leak
    locus) any overlap raises :class:`SplitLeakError`; for the others it is recorded as an overlap
    count so a real leak stays visible without refusing a benign short-text collision. Also folds in
    any dedup sidecar the builder wrote. Returns the ``.meta`` payload, empty when nothing applies.
    """
    import pyarrow.parquet as pq

    meta: dict[str, Any] = {}
    if (sidecar := source.parent / DEDUP_SIDECAR).exists():
        meta["dedup"] = json.loads(sidecar.read_text())
    train_source = source.parent / "train.parquet"
    if train_source.exists():
        test_texts = _view_row_texts(view, pq.read_table(source))
        train_texts = _view_row_texts(view, pq.read_table(train_source))
        leaked = set(exact_text_overlap(train_texts.values(), test_texts.values()))
        leaked_ids = sorted(rid for rid, text in test_texts.items() if text.strip() in leaked)
        if leaked_ids and view in DISJOINT_ENFORCED_VIEWS:
            raise SplitLeakError(view, leaked_ids)
        meta["disjoint_train_rows_checked"] = len(train_texts)
        meta["train_eval_overlap"] = len(leaked_ids)
    return meta


def _view_row_texts(view: str, table: pa.Table) -> dict[str, str]:
    """Each row's exact-disjointness text keyed by id: the flattened window for watcher, ``text`` else."""
    match view:
        case "watcher":
            return {(row := WatcherRow.from_record(record)).id: row.gate_text for record in table.to_pylist()}
        case _:
            return {str(record["id"]): str(record["text"]) for record in table.to_pylist()}


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
    frame: ProbsFrame,
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


def build_steer_type_eval(
    *,
    dataset_dir: Path | None = None,
    seed: int = 1729,
    embed: Embedder | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> Path:
    """Build the frozen steer-type eval source from the exported traces test split.

    One row per judged feedback moment: :func:`steer_type_text` as ``text`` and the
    judge's eleven-way ``category`` as the label. Near-duplicate inputs collapse to
    one seeded representative each (:func:`~cc_steer.retrain.data.near_dup_indices`),
    with the embedding-cosine pass enabled when ``embed`` is supplied; the collapse
    counts land in a ``dedup.json`` sidecar the freeze folds into the manifest, so the
    prune is reported rather than silent. The test-split parquet ``freeze_eval`` reads
    is written under ``<dataset_dir>/steer_type/test.parquet``. Returns the written path.
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
    kept, stats = near_dup_indices(
        [str(row["text"]) for row in rows], seed=seed, embed=embed, semantic_threshold=semantic_threshold
    )
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
    _write_dedup_sidecar(out.parent, stats)
    return out


def build_pick_eval(
    *,
    decisions_path: Path | None = None,
    dataset_dir: Path | None = None,
    seed: int = 1729,
    embed: Embedder | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> Path:
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
    kept, stats = near_dup_indices(
        [str(row["text"]) for row in rows], seed=seed, embed=embed, semantic_threshold=semantic_threshold
    )
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
    _write_dedup_sidecar(out.parent, stats)
    return out


def _write_dedup_sidecar(view_dir: Path, stats: DedupStats) -> None:
    (view_dir / DEDUP_SIDECAR).write_text(json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n")


def freeze_steer_type(
    *,
    dataset_dir: Path | None = None,
    root: Path | None = None,
    seed: int = 1729,
    embed: Embedder | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> str:
    """Build then freeze the steer-type eval; returns the frozen file's sha256."""
    build_steer_type_eval(dataset_dir=dataset_dir, seed=seed, embed=embed, semantic_threshold=semantic_threshold)
    return freeze_eval("steer_type", dataset_dir=dataset_dir, root=root)


def freeze_pick(
    *,
    decisions_path: Path | None = None,
    dataset_dir: Path | None = None,
    root: Path | None = None,
    seed: int = 1729,
    embed: Embedder | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> str:
    """Build then freeze the pick-prediction eval; returns the frozen file's sha256."""
    build_pick_eval(
        decisions_path=decisions_path,
        dataset_dir=dataset_dir,
        seed=seed,
        embed=embed,
        semantic_threshold=semantic_threshold,
    )
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


@dataclass(frozen=True, slots=True)
class ArmComparison:
    """A paired fast-DeLong comparison of two watchers on the frozen frame.

    Both arms are compared on the fire scale (``fire = 1 - P(NO_STEER)``), so a higher score means
    fire on a true-steer row. ``paired`` carries both AUCs, their delta, the covariance-aware
    standard error, and — the point of persisting both arms — the pairing correlation ``rho``.
    ``is_actionable`` applies the instrument card's two-part rule at ``frame_mde = mde(se_delta)``.

    Attributes:
        incumbent: The incumbent arm's version label.
        candidate: The candidate arm's version label.
        paired: The paired fast-DeLong result (``auc_a`` = incumbent, ``auc_b`` = candidate).
        frame_mde: The frame's minimum detectable effect for this pairing, ``mde(se_delta)``.
        is_actionable: Whether the delta clears the card's CI-excludes-zero-and-beats-MDE bar.
    """

    incumbent: str
    candidate: str
    paired: PairedDeLong
    frame_mde: float
    is_actionable: bool

    def as_metrics(self) -> dict[str, float]:
        """The comparison as flat journal metrics, keys prefixed ``paired_``."""
        return {
            "paired_incumbent_auc": self.paired.auc_a,
            "paired_candidate_auc": self.paired.auc_b,
            "paired_delta_auc": self.paired.delta,
            "paired_se_delta": self.paired.se_delta,
            "paired_rho": self.paired.rho,
            "paired_ci_lo": self.paired.ci95[0],
            "paired_ci_hi": self.paired.ci95[1],
            "paired_frame_mde": self.frame_mde,
            "paired_actionable": float(self.is_actionable),
        }


def compare_arms(
    frame: EvalFrame, incumbent_nosteer: np.ndarray, candidate_nosteer: np.ndarray, *, incumbent: str, candidate: str
) -> ArmComparison:
    """Pair two arms' per-row ``P(NO_STEER)`` vectors on ``frame`` with fast-DeLong and the card rule.

    Both vectors align to ``frame.ids``; they are flipped to the fire scale before the paired
    test so the AUCs read as fire-vs-true-steer. Returns the :class:`ArmComparison` — both AUCs,
    the delta, the measured ``rho``, and whether the delta is actionable at ``mde(se_delta)``.
    """
    paired = paired_delong(
        frame.labels.astype(int),
        1.0 - np.asarray(incumbent_nosteer, dtype=np.float64),
        1.0 - np.asarray(candidate_nosteer, dtype=np.float64),
    )
    frame_mde = mde(paired.se_delta)
    return ArmComparison(incumbent, candidate, paired, frame_mde, actionable(paired.delta, paired.se_delta, frame_mde))


def comparison_path(incumbent: str, candidate: str, *, root: Path | None = None) -> Path:
    """The paired-comparison artifact path: ``comparisons/<incumbent>__<candidate>.json``."""
    return eval_root(root) / COMPARISONS_DIRNAME / f"{incumbent}__{candidate}.json"


def write_comparison(
    frame: EvalFrame,
    comparison: ArmComparison,
    incumbent_nosteer: np.ndarray,
    candidate_nosteer: np.ndarray,
    *,
    root: Path | None = None,
) -> Path:
    """Persist both arms' per-row ``P(NO_STEER)`` and the paired stats — the single comparison writer.

    Writes both vectors aligned to ``frame.ids`` so a rejected candidate's paired comparison is
    reconstructable later, stamped with the frame digest. Returns the written path.
    """
    inc = np.asarray(incumbent_nosteer, dtype=np.float64)
    cand = np.asarray(candidate_nosteer, dtype=np.float64)
    path = comparison_path(comparison.incumbent, comparison.candidate, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "dataset_digest": frame.digest,
                    "incumbent": comparison.incumbent,
                    "candidate": comparison.candidate,
                },
                "paired": comparison.as_metrics(),
                "probs": {
                    row_id: {"incumbent": float(inc[i]), "candidate": float(cand[i])}
                    for i, row_id in enumerate(frame.ids)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One provenance-tagged label for a mined candidate row.

    Attributes:
        id: The candidate's stable id.
        is_steering: The label — a true steer (``True``) or NO_STEER (``False``).
        category: The steer category (empty for a negative).
        provenance: The label's source, one of :data:`PROVENANCE_PRECEDENCE`.
    """

    id: str
    is_steering: bool
    category: str
    provenance: Literal["human", "fable", "medium_judge"]


@dataclass(frozen=True, slots=True)
class ResolvedLabel:
    """A candidate's authoritative label after applying provenance precedence.

    ``provenance`` is the winning source and is never ``medium_judge`` — a medium-judge label is
    guidance only, so a row labelled only by the medium judge has no :class:`ResolvedLabel` and is
    dropped. ``guidance`` records the medium judge's own call when present, and
    ``agrees_with_guidance`` whether that call matched the authoritative label.
    """

    id: str
    is_steering: bool
    category: str
    provenance: Literal["human", "fable"]
    guidance: bool | None
    agrees_with_guidance: bool | None


def resolve_labels(records: Sequence[LabelRecord]) -> tuple[list[ResolvedLabel], list[str]]:
    """Resolve per-id labels by provenance precedence; return ``(resolved, medium_only_dropped)``.

    Each id's authoritative label is its highest-precedence non-guidance record (human over fable);
    the medium judge attaches as guidance but never labels alone, so an id whose only source is the
    medium judge is returned in ``medium_only_dropped``, never in ``resolved``.
    """
    by_id: dict[str, list[LabelRecord]] = defaultdict(list)
    for record in records:
        by_id[record.id].append(record)
    resolved: list[ResolvedLabel] = []
    medium_only: list[str] = []
    for rid, recs in by_id.items():
        authoritative = min(
            (r for r in recs if r.provenance != GUIDANCE_PROVENANCE),
            key=lambda r: PROVENANCE_PRECEDENCE.index(r.provenance),
            default=None,
        )
        guidance = next((r for r in recs if r.provenance == GUIDANCE_PROVENANCE), None)
        if authoritative is None:
            medium_only.append(rid)
            continue
        match authoritative.provenance:
            case "human" | "fable" as provenance:
                resolved.append(
                    ResolvedLabel(
                        id=rid,
                        is_steering=authoritative.is_steering,
                        category=authoritative.category,
                        provenance=provenance,
                        guidance=guidance.is_steering if guidance else None,
                        agrees_with_guidance=guidance.is_steering == authoritative.is_steering if guidance else None,
                    )
                )
            case escaped:
                raise AssertionError(f"guidance provenance {escaped!r} escaped the authoritative filter")
    return sorted(resolved, key=lambda r: r.id), sorted(medium_only)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """The Hanley & McNeil (1982) analytic standard error of an AUC at ``(n_pos, n_neg)``.

    Projects a frame's DeLong-scale SE before any scoring exists, from an assumed operating ``auc``
    and the class counts — the sizing knob for a candidate frame.
    """
    if n_pos < 1 or n_neg < 1:
        return float("inf")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return math.sqrt(max(variance, 0.0))


def projected_frame_mde(n_pos: int, n_neg: int, *, auc: float = PROJECTION_AUC, rho: float = PROJECTION_RHO) -> float:
    """The projected paired minimum detectable effect for a candidate frame of ``(n_pos, n_neg)``.

    Turns the Hanley-McNeil single-arm SE into a paired ``se_delta = se * sqrt(2(1 - rho))`` for two
    equally-precise arms correlated at ``rho``, then feeds :func:`~cc_steer.instrument.mde`. This is
    the number a candidate frame must drive under :data:`TARGET_MDE` before it is worth freezing.
    """
    return mde(hanley_mcneil_se(auc, n_pos, n_neg) * math.sqrt(2.0 * (1.0 - rho)))


def negatives_for_target_mde(
    n_pos: int,
    *,
    target_mde: float = TARGET_MDE,
    auc: float = PROJECTION_AUC,
    rho: float = PROJECTION_RHO,
    cap: int = 200_000,
) -> int:
    """The fewest negatives that drive the projected paired MDE at or under ``target_mde``.

    Projected MDE falls monotonically as negatives grow toward a floor set by ``n_pos``; when even
    ``cap`` negatives cannot reach ``target_mde`` (the floor is above it), ``cap`` is returned.
    """
    if n_pos < 1:
        return 0
    if projected_frame_mde(n_pos, cap, auc=auc, rho=rho) > target_mde:
        return cap
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if projected_frame_mde(n_pos, mid, auc=auc, rho=rho) <= target_mde:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    """A projected, negative-rich candidate frame — computed, never frozen.

    Attributes:
        positive_ids: The authoritative true-steer rows admitted.
        negative_ids: The chosen NO_STEER rows (authoritative or structural).
        n_pos: ``len(positive_ids)``.
        n_neg: ``len(negative_ids)``.
        projected_mde: The projected paired MDE at this sizing.
        target_mde: The MDE target the frame is sized against.
        meets_target: Whether ``projected_mde <= target_mde``.
        provenance_counts: Admitted-row counts by label source (plus ``structural`` negatives).
        guidance_agree: Admitted rows where the medium-judge guidance matched the label.
        guidance_disagree: Admitted rows where it disagreed.
        medium_only_dropped: Candidates dropped for carrying only a medium-judge label.
    """

    positive_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]
    n_pos: int
    n_neg: int
    projected_mde: float
    target_mde: float
    meets_target: bool
    provenance_counts: Mapping[str, int]
    guidance_agree: int
    guidance_disagree: int
    medium_only_dropped: int

    @property
    def ids(self) -> tuple[str, ...]:
        """Every admitted row id, positives before negatives."""
        return self.positive_ids + self.negative_ids


def plan_rebuild_frame(
    labels: Sequence[LabelRecord],
    negative_pool: Sequence[str],
    *,
    target_mde: float = TARGET_MDE,
    projection_auc: float = PROJECTION_AUC,
    projection_rho: float = PROJECTION_RHO,
    min_negative_ratio: float = 1.0,
    seed: int = 1729,
) -> RebuildPlan:
    """Size a negative-rich candidate frame from mined labels and a negative pool, reporting its MDE.

    Resolves ``labels`` by provenance precedence (human > fable > medium-judge guidance), admits
    every authoritative true-steer, then draws negatives — authoritative negatives first, then the
    structural ``negative_pool`` — until both the ``min_negative_ratio`` floor and the count that
    drives the projected paired MDE under ``target_mde`` are met, capped by what the pool holds. The
    returned :class:`RebuildPlan` reports the projected MDE and provenance mix so the frame can be
    judged before anything is frozen; it never freezes or cuts a frame over.
    """
    resolved, medium_only = resolve_labels(labels)
    positives = [r for r in resolved if r.is_steering]
    positive_ids = {r.id for r in positives}
    resolved_negatives = [r for r in resolved if not r.is_steering]
    negative_provenance = {r.id: r.provenance for r in resolved_negatives}
    authoritative = [r.id for r in resolved_negatives if r.id not in positive_ids]
    structural = [
        nid for nid in dict.fromkeys(negative_pool) if nid not in positive_ids and nid not in set(authoritative)
    ]
    n_pos = len(positives)
    want = max(
        math.ceil(min_negative_ratio * n_pos),
        negatives_for_target_mde(n_pos, target_mde=target_mde, auc=projection_auc, rho=projection_rho),
    )
    n_neg = min(want, len(authoritative) + len(structural))
    rng = np.random.default_rng(seed)
    fill = n_neg - min(n_neg, len(authoritative))
    chosen = (
        sorted(
            [
                *(
                    authoritative
                    if fill
                    else (authoritative[i] for i in rng.choice(len(authoritative), size=n_neg, replace=False))
                ),
                *([structural[i] for i in rng.choice(len(structural), size=fill, replace=False)] if fill else []),
            ]
        )
        if n_neg
        else []
    )
    projected = projected_frame_mde(n_pos, n_neg, auc=projection_auc, rho=projection_rho)
    provenance_counts: Counter[str] = Counter(r.provenance for r in positives)
    for nid in chosen:
        provenance_counts[negative_provenance.get(nid, "structural")] += 1
    frame_ids = positive_ids | set(chosen)
    guided = [r for r in resolved if r.id in frame_ids and r.agrees_with_guidance is not None]
    return RebuildPlan(
        positive_ids=tuple(sorted(positive_ids)),
        negative_ids=tuple(chosen),
        n_pos=n_pos,
        n_neg=n_neg,
        projected_mde=projected,
        target_mde=target_mde,
        meets_target=projected <= target_mde,
        provenance_counts=dict(provenance_counts),
        guidance_agree=sum(1 for r in guided if r.agrees_with_guidance),
        guidance_disagree=sum(1 for r in guided if not r.agrees_with_guidance),
        medium_only_dropped=len(medium_only),
    )
