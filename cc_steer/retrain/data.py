"""The training pool: read the exported watcher train view and curate it deterministically.

The export writes the watcher view as parquet under ``~/.cc-steer/dataset/watcher/``
(``dataset_dir`` overrides). This module loads ``train.parquet`` into
:class:`WatcherRow` records and rebuilds the lab's curation pipeline as composable,
seeded steps: near-duplicate collapse to one representative per cluster
(:func:`near_dup_representatives`), a stratified validation carve
(:func:`carve_val`), NO_STEER oversampling to balance the classes
(:func:`balance_no_steer`), and corrective-positive oversampling past the
direction count (:func:`oversample_corrective_to`). Every step is deterministic in
its seed, so the whole pool — and its :func:`dataset_digest` — is reproducible and
drives the retrain trigger.

Rendering is the production contract: ``gate_text`` flattens the full window,
``draft_text`` flattens the tail under :data:`~cc_steer.rendering.DRAFT_CHAR_CAP`
chars, and both come from :mod:`cc_steer.rendering` / :mod:`cc_steer.watcher.cascade`
verbatim so train and serve stay byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType

import numpy as np
import pyarrow.parquet as pq

from cc_steer.rendering import DRAFT_CHAR_CAP, Message, tail_messages
from cc_steer.watcher.cascade import flattened

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pyarrow as pa

DATASET_DIR: Path = Path.home() / ".cc-steer" / "dataset"
HF_PUSH_NAME = ".hf_push.json"
DIGEST_CHARS = 16
SEED = 1729
VAL_N = 200
DIRECTION = "direction"

NGRAM = 5
JACCARD_THRESHOLD = 0.8
NUM_PERM = 128
BANDS = 32
MINHASH_PRIME = 2_147_483_647  # 2**31 - 1, keeps a*H products inside uint64
SEMANTIC_THRESHOLD = 0.92  # cosine floor for the embedding near-dup pass — conservative to avoid over-pruning

DatasetDigest = NewType("DatasetDigest", str)

type Embedder = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True, slots=True)
class WatcherRow:
    """One watcher-view row: the context window plus its held-out reference steer.

    Attributes:
        id: The row's stable identifier.
        prompt: The context turns as chat messages (the model-visible window).
        reference: The completion — the steering direction or the ``NO_STEER`` sentinel.
        verbatim: The user's raw steering message, empty for negatives.
        label: ``True`` for a true-steer row (should fire).
        category: The steering category; ``"direction"`` marks option-picking.
        source_kind: The capture source; ``"question_answer"`` marks a QA event.
        session_id: The session the row was mined from.
    """

    id: str
    prompt: tuple[Message, ...]
    reference: str
    verbatim: str
    label: bool
    category: str
    source_kind: str = ""
    session_id: str = ""

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> WatcherRow:
        """Build a row from one exported watcher parquet record."""
        return cls(
            id=str(record["id"]),
            prompt=tuple({"role": m["role"], "content": m["content"]} for m in record["prompt"]),
            reference=str(record["completion"][0]["content"]),
            verbatim=str(record["verbatim"]),
            label=bool(record["label"]),
            category=str(record["category"]),
            source_kind=str(record.get("source_kind") or ""),
            session_id=str(record.get("session_id") or ""),
        )

    @property
    def gate_text(self) -> str:
        """The stage-1 input: the full flattened window, byte-identical to rendering.gate_text."""
        return flattened(self.prompt)

    def draft_text(self, cap: int = DRAFT_CHAR_CAP) -> str:
        """The local drafter's user message: the flattened tail under ``cap`` chars."""
        return flattened(tail_messages(self.prompt, cap))


@dataclass(frozen=True, slots=True)
class DedupStats:
    """Near-dup collapse counts, with the semantic pass's marginal removals broken out.

    ``n_semantic_removed`` is how many rows the embedding pass collapsed *beyond* what
    char-5-gram MinHash alone would have removed — the number that must be watched so the
    semantic pass reports rather than silently over-prunes. It is 0 (and
    ``semantic_threshold`` is ``None``) whenever no embedder ran.
    """

    n_in: int
    n_kept: int
    n_removed: int
    n_clusters: int
    n_multi_member_clusters: int
    n_semantic_removed: int = 0
    semantic_threshold: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "dedup_n_in": float(self.n_in),
            "dedup_n_kept": float(self.n_kept),
            "dedup_n_removed": float(self.n_removed),
            "dedup_n_clusters": float(self.n_clusters),
            "dedup_n_multi_member_clusters": float(self.n_multi_member_clusters),
            "dedup_n_semantic_removed": float(self.n_semantic_removed),
            **({"dedup_semantic_threshold": self.semantic_threshold} if self.semantic_threshold is not None else {}),
        }


def canonical_json(payload: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, non-JSON values stringified."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def row_digest(row: Mapping[str, object]) -> str:
    """SHA-256 of one row's canonical JSON."""
    return hashlib.sha256(canonical_json(dict(row)).encode()).hexdigest()


def dataset_digest(rows: Sequence[Mapping[str, object]]) -> DatasetDigest:
    """Order-invariant content digest over rows — the retrain trigger and journal receipt."""
    hasher = hashlib.sha256()
    for digest in sorted(row_digest(row) for row in rows):
        hasher.update(digest.encode())
        hasher.update(b"\n")
    return DatasetDigest(hasher.hexdigest()[:DIGEST_CHARS])


def load_train_table(*, dataset_dir: Path | None = None) -> pa.Table:
    """Read the exported watcher train view from ``<dataset_dir>/watcher/train.parquet``."""
    path = (dataset_dir or DATASET_DIR) / "watcher" / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no watcher train parquet at {path}")
    return pq.read_table(path)


def load_train_rows(*, dataset_dir: Path | None = None) -> list[WatcherRow]:
    """The exported watcher train view as rows, order preserved."""
    return [WatcherRow.from_record(record) for record in load_train_table(dataset_dir=dataset_dir).to_pylist()]


def train_digest(*, dataset_dir: Path | None = None) -> DatasetDigest:
    """The watcher train view's content digest — what :func:`~cc_steer.retrain.promotion.should_retrain` triggers on."""
    return dataset_digest(load_train_table(dataset_dir=dataset_dir).to_pylist())


def hf_revision(*, dataset_dir: Path | None = None) -> str | None:
    """The HuggingFace revision recorded by the latest successful dataset push, if any."""
    path = (dataset_dir or DATASET_DIR) / HF_PUSH_NAME
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    return json.loads(raw)["hf_revision"]


def training_sample(row: WatcherRow, *, system: str, cap: int = DRAFT_CHAR_CAP) -> dict[str, Any]:
    """One mlx-lm chat-format training record: system + flattened context tail -> completion."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": row.draft_text(cap)},
            {"role": "assistant", "content": row.reference},
        ]
    }


def balance_no_steer(rows: Sequence[WatcherRow], *, seed: int = SEED) -> tuple[list[WatcherRow], float]:
    """Oversample NO_STEER rows to match the steer count; return ``(rows, ratio)``.

    Every original row is kept once; negatives are then topped up with seeded draws
    (without replacement per cycle) until the classes balance. ``ratio`` is
    ``positives / negatives`` before balancing.
    """
    positives = [r for r in rows if r.label]
    negatives = [r for r in rows if not r.label]
    if not positives or not negatives:
        return list(rows), 0.0
    ratio = len(positives) / len(negatives)
    rng = np.random.default_rng(seed)
    extra_n = len(positives) - len(negatives)
    extras: list[WatcherRow] = []
    while extra_n > 0:
        take = min(extra_n, len(negatives))
        extras.extend(negatives[i] for i in rng.choice(len(negatives), size=take, replace=False))
        extra_n -= take
    balanced = list(rows) + extras
    rng.shuffle(balanced)  # type: ignore[arg-type]
    return balanced, ratio


def carve_val(
    rows: Sequence[WatcherRow], *, n: int = VAL_N, seed: int = SEED
) -> tuple[list[WatcherRow], list[WatcherRow]]:
    """Seeded label-stratified ``(val, rest)`` carve of the train view."""
    rng = np.random.default_rng(seed)
    val_idx: set[int] = set()
    for label in (True, False):
        idx = [i for i, row in enumerate(rows) if row.label == label]
        take = max(1, round(n * len(idx) / len(rows)))
        val_idx.update(idx[i] for i in rng.choice(len(idx), size=min(take, len(idx)), replace=False))
    val = [rows[i] for i in sorted(val_idx)]
    rest = [rows[i] for i in range(len(rows)) if i not in val_idx]
    return val, rest


def oversample_corrective_to(
    rows: Sequence[WatcherRow], *, factor: float, seed: int = SEED
) -> tuple[list[WatcherRow], int, int]:
    """Duplicate corrective (label, non-direction) positives to ``factor`` x their count — no clamp.

    Lifts the direction-parity clamp so corrective can exceed direction. Seeded draws
    cycle the corrective pool when the target exceeds one full copy. Returns the
    shuffled pool and the ``(before, after)`` corrective counts.
    """
    rows = list(rows)
    corrective = [r for r in rows if r.label and r.category != DIRECTION]
    if not corrective:
        return rows, 0, 0
    target = round(factor * len(corrective))
    rng = np.random.default_rng(seed)
    extras: list[WatcherRow] = []
    remaining = max(0, target - len(corrective))
    while remaining > 0:
        take = min(remaining, len(corrective))
        extras.extend(corrective[i] for i in rng.choice(len(corrective), size=take, replace=False))
        remaining -= take
    out = rows + extras
    rng.shuffle(out)  # type: ignore[arg-type]
    return out, len(corrective), len(corrective) + len(extras)


def shingles(text: str, n: int = NGRAM) -> set[int]:
    """The set of char-``n``-gram rolling hashes of ``text`` (empty if too short)."""
    data = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(np.uint64)
    if data.size < n:
        return set()
    base = np.uint64(257)
    acc = np.zeros(data.size - n + 1, dtype=np.uint64)
    for offset in range(n):
        acc = acc * base + data[offset : offset + acc.size]
    return {int(v) for v in acc.tolist()}


def jaccard(a: set[int], b: set[int]) -> float:
    """Exact Jaccard over two shingle sets (0.0 if both empty)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def near_dup_representatives(
    rows: Sequence[WatcherRow],
    *,
    threshold: float = JACCARD_THRESHOLD,
    seed: int = SEED,
    embed: Embedder | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> tuple[list[int], DedupStats]:
    """Collapse near-duplicate ``draft_text`` rows to one seeded representative each.

    A MinHash/LSH self-join finds candidate pairs, an exact char-5-gram Jaccard
    confirms each, and the rows that clear ``threshold`` union into clusters. One
    seeded member survives per cluster. Returns the kept indices (sorted) and the
    collapse counts; deterministic in ``seed``. Empty-shingle rows (a context tail
    below the 5-char n-gram floor) never union, so they each survive as singletons.
    ``embed`` enables the semantic near-dup pass (see :func:`near_dup_indices`).
    """
    return near_dup_indices(
        [row.draft_text(DRAFT_CHAR_CAP) for row in rows],
        threshold=threshold,
        seed=seed,
        embed=embed,
        semantic_threshold=semantic_threshold,
    )


def near_dup_indices(
    texts: Sequence[str],
    *,
    threshold: float = JACCARD_THRESHOLD,
    seed: int = SEED,
    embed: Embedder | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> tuple[list[int], DedupStats]:
    """Collapse near-duplicate ``texts`` to one seeded representative index each.

    The text-addressed core shared by :func:`near_dup_representatives` and the frozen
    eval frame builders: any view that can render one string per row (a watcher
    tail, a rendered ask, a classification input) deduplicates through the same
    MinHash/LSH self-join, so every frame inherits one collapse contract.

    When ``embed`` is supplied, an embedding-cosine pass runs *on top of* the MinHash
    clusters: any pair whose vectors clear ``semantic_threshold`` unions too, catching
    paraphrases that share no char-5-gram. The returned :class:`DedupStats` breaks out
    ``n_semantic_removed`` — the rows collapsed beyond MinHash alone — so a run can see
    exactly what the semantic pass pruned rather than trusting it blindly.
    """
    n = len(texts)
    if n == 0:
        return [], DedupStats(0, 0, 0, 0, 0)
    sets = [shingles(text) for text in texts]
    sigs = _minhash_signatures(sets, seed=seed)
    candidates = _lsh_candidates(sigs, sigs)
    minhash_pairs = [
        (i, j) for i in range(n) for j in candidates[i] if j > i and jaccard(sets[i], sets[j]) >= threshold
    ]
    if embed is None:
        kept, clusters = _collapse(n, minhash_pairs, seed=seed)
        return kept, _dedup_stats(n, kept, clusters, n_semantic_removed=0, semantic_threshold=None)
    minhash_kept, _ = _collapse(n, minhash_pairs, seed=seed)
    kept, clusters = _collapse(
        n, minhash_pairs + semantic_near_dup_pairs(texts, embed, threshold=semantic_threshold), seed=seed
    )
    return kept, _dedup_stats(
        n, kept, clusters, n_semantic_removed=len(minhash_kept) - len(kept), semantic_threshold=semantic_threshold
    )


def semantic_near_dup_pairs(texts: Sequence[str], embed: Embedder, *, threshold: float) -> list[tuple[int, int]]:
    """The ``(i, j)`` index pairs whose embedding cosine similarity clears ``threshold``.

    Embeds every text once, L2-normalizes, and returns each upper-triangular pair at or
    above ``threshold``. A zero vector never matches (its normalized cosine is 0).
    """
    if len(texts) < 2:
        return []
    vectors = np.asarray(embed(list(texts)), dtype=np.float64)
    unit = vectors / np.where((norms := np.linalg.norm(vectors, axis=1, keepdims=True)) == 0.0, 1.0, norms)
    sims = unit @ unit.T
    return [(int(i), int(j)) for i, j in zip(*np.triu_indices(len(texts), k=1)) if sims[i, j] >= threshold]


def exact_text_overlap(reference: Sequence[str], query: Sequence[str]) -> list[str]:
    """The whitespace-stripped texts present in both ``reference`` and ``query`` (sorted, unique).

    The exact cross-split leak check: a ``query`` (eval) text that also appears in
    ``reference`` (train) is a train/eval overlap, whatever the ids differ to.
    """
    seen = {text.strip() for text in reference if text.strip()}
    return sorted({stripped for text in query if (stripped := text.strip()) in seen})


def _collapse(n: int, pairs: Sequence[tuple[int, int]], *, seed: int) -> tuple[list[int], dict[int, list[int]]]:
    roots = _union_find_roots(n, pairs)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[roots[i]].append(i)
    rng = np.random.default_rng(seed)
    kept = sorted(clusters[root][int(rng.integers(0, len(clusters[root])))] for root in sorted(clusters))
    return kept, clusters


def _dedup_stats(
    n: int,
    kept: Sequence[int],
    clusters: Mapping[int, list[int]],
    *,
    n_semantic_removed: int,
    semantic_threshold: float | None,
) -> DedupStats:
    return DedupStats(
        n_in=n,
        n_kept=len(kept),
        n_removed=n - len(kept),
        n_clusters=len(clusters),
        n_multi_member_clusters=sum(1 for members in clusters.values() if len(members) > 1),
        n_semantic_removed=n_semantic_removed,
        semantic_threshold=semantic_threshold,
    )


def sentence_transformer_embedder(
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2", *, device: str | None = None
) -> Embedder:
    """An :data:`Embedder` backed by a local ``sentence-transformers`` model (the ``embed`` extra).

    Loads ``model_id`` once and returns a callable that embeds a batch of texts to a
    ``(n, dim)`` float matrix. ``sentence-transformers`` is imported lazily so importing
    this module never drags it in.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    return lambda texts: np.asarray(
        model.encode(list(texts), normalize_embeddings=False, show_progress_bar=False), dtype=np.float64
    )


def _minhash_signatures(shingle_sets: Sequence[set[int]], *, seed: int = SEED) -> np.ndarray:
    """A ``(rows x NUM_PERM)`` uint32 MinHash signature matrix; empty rows get MAX."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MINHASH_PRIME, size=NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, MINHASH_PRIME, size=NUM_PERM, dtype=np.uint64)
    prime = np.uint64(MINHASH_PRIME)
    sigs = np.full((len(shingle_sets), NUM_PERM), np.iinfo(np.uint32).max, dtype=np.uint64)
    for row, values in enumerate(shingle_sets):
        if not values:
            continue
        h = np.fromiter((v % MINHASH_PRIME for v in values), dtype=np.uint64, count=len(values))
        sigs[row] = ((a[:, None] * h[None, :] + b[:, None]) % prime).min(axis=1)
    return sigs.astype(np.uint32)


def _lsh_candidates(eval_sigs: np.ndarray, train_sigs: np.ndarray) -> dict[int, set[int]]:
    """eval row index -> set of train row indices sharing any LSH band bucket."""
    rows_per_band = NUM_PERM // BANDS
    candidates: dict[int, set[int]] = {i: set() for i in range(len(eval_sigs))}
    for band in range(BANDS):
        lo = band * rows_per_band
        hi = lo + rows_per_band
        buckets: dict[bytes, list[int]] = {}
        for t_idx in range(len(train_sigs)):
            buckets.setdefault(train_sigs[t_idx, lo:hi].tobytes(), []).append(t_idx)
        for e_idx in range(len(eval_sigs)):
            if hit := buckets.get(eval_sigs[e_idx, lo:hi].tobytes()):
                candidates[e_idx].update(hit)
    return candidates


def _union_find_roots(n: int, pairs: Sequence[tuple[int, int]]) -> list[int]:
    """Roots of a union-find over ``n`` items unioned by ``pairs`` (root = min index)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]
