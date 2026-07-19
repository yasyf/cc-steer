"""The exemplar index: embedded past steers, retrieved for the frontier refiner.

Stage 3 of the live cascade conditions a frontier model on how this user has
steered in similar moments. This module embeds every accepted steering event's
rendered context (train split only, so evaluation retrieval is never
contaminated) into the ``exemplar_embedding`` table, and retrieves neighbors at
runtime by brute-force cosine over the whole index — at a few thousand rows an
ANN index would only add recall loss — re-selected with MMR so the exemplars
shown are similar to the moment yet diverse among themselves.

Embedding runs through the Voyage API by default (``voyage-4-large``, the
``embed`` extra) — the whole index embeds in massively parallel batches — with
a local sentence-transformers path for offline models. Pure retrieval needs
only numpy, so the live daemon can load and query the index without the
embedding stack when vectors are already built.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anyio
import numpy as np
from cc_transcript.context import ContextWindow

from cc_steer.rendering import gate_text, split_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_steer.store import FeedbackStore

EMBED_MODEL = "voyage-4-large"
LOCAL_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_EMBED_CHARS = 16_000
MAX_SEQ_TOKENS = 4_096
VOYAGE_BATCH_TEXTS = 256
VOYAGE_BATCH_CHARS = 240_000
VOYAGE_CONCURRENCY = 16
ENV_FILE = Path.home() / ".cc-steer" / ".env"

EXEMPLAR_EVENTS_QUERY = """
SELECT e.dedup_key, e.session_id, e.event_uuid, e.text, e.context_json, t.category
FROM feedback_events e
JOIN latest_judge t ON t.dedup_key = e.dedup_key
WHERE t.is_steering = 1 AND e.quarantined_reason IS NULL
ORDER BY e.id
"""

EXEMPLAR_DETAIL_QUERY = """
SELECT e.dedup_key, e.text, e.context_json, ap.category,
  (SELECT GROUP_CONCAT(direction, char(10)) FROM latest_refinement r WHERE r.dedup_key = e.dedup_key) AS direction
FROM feedback_events e
JOIN accepted_steering ap ON ap.dedup_key = e.dedup_key
"""


class Encoder(Protocol):
    """Anything that embeds a batch of texts into unit-normalizable vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class Exemplar:
    """One retrieved past steer, ready to show the frontier refiner.

    Attributes:
        dedup_key: The parent event.
        context_text: The rendered moment the user steered at.
        direction: The refined one-sentence direction(s), newline-joined.
        verbatim: The user's raw steering text.
        category: The judge's category.
        score: The retrieval score against the query.
    """

    dedup_key: str
    context_text: str
    direction: str
    verbatim: str
    category: str
    score: float


@dataclass(frozen=True, slots=True)
class IndexReport:
    """The outcome of one indexing pass.

    Attributes:
        embedded: How many exemplars were (re)embedded this pass.
        current: How many were already embedded at the current text digest.
        total: Train-split steering events eligible for the index.
    """

    embedded: int
    current: int
    total: int


def voyage_api_key() -> str:
    """The Voyage key: ``VOYAGE_API_KEY`` in the environment or ``~/.cc-steer/.env``."""
    if key := os.environ.get("VOYAGE_API_KEY"):
        return key
    from dotenv import dotenv_values

    if key := dotenv_values(ENV_FILE).get("VOYAGE_API_KEY"):
        return key
    raise RuntimeError("VOYAGE_API_KEY is not set (environment or ~/.cc-steer/.env)")


def voyage_batches(texts: Sequence[str]) -> list[list[int]]:
    """Splits texts into index batches under the API's per-request budgets."""
    batches: list[list[int]] = [[]]
    chars = 0
    for index, text in enumerate(texts):
        if batches[-1] and (len(batches[-1]) >= VOYAGE_BATCH_TEXTS or chars + len(text) > VOYAGE_BATCH_CHARS):
            batches.append([])
            chars = 0
        batches[-1].append(index)
        chars += len(text)
    return batches if batches[0] else []


async def voyage_embed(
    texts: Sequence[str],
    *,
    model: str = EMBED_MODEL,
    input_type: str = "document",
    concurrency: int = VOYAGE_CONCURRENCY,
) -> np.ndarray:
    """Embeds every text through the Voyage API, batches fired concurrently.

    Batches respect the per-request text and token budgets and run under a
    semaphore; the client retries rate limits and transient server errors
    internally.
    """
    try:
        from voyageai import AsyncClient
    except ImportError as error:
        raise RuntimeError("embedding requires the 'embed' extra: pip install 'cc-steer[embed]'") from error

    client = AsyncClient(api_key=voyage_api_key(), max_retries=3, timeout=120)
    vectors: list[np.ndarray | None] = [None] * len(texts)
    limiter = anyio.Semaphore(concurrency)

    async def embed_batch(indexes: list[int]) -> None:
        async with limiter:
            response = await client.embed([texts[i] for i in indexes], model=model, input_type=input_type)
        for index, vector in zip(indexes, response.embeddings, strict=True):
            vectors[index] = np.asarray(vector, dtype=np.float32)

    async with anyio.create_task_group() as group:
        for indexes in voyage_batches(texts):
            group.start_soon(embed_batch, indexes)
    resolved = [vector for vector in vectors if vector is not None]
    assert len(resolved) == len(texts)
    return np.stack(resolved)


class VoyageQueryEncoder:
    """A sync single-query encoder over the Voyage API, for the live cascade."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        self.model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from voyageai import Client

        client = Client(api_key=voyage_api_key(), max_retries=3, timeout=60)
        response = client.embed(list(texts), model=self.model, input_type="query")
        return np.asarray(response.embeddings, dtype=np.float32)


def query_encoder(model: str = EMBED_MODEL) -> Encoder:
    """The runtime query encoder for ``model``: Voyage API or local, by name."""
    if model.startswith("voyage"):
        return VoyageQueryEncoder(model)
    return sentence_transformer_encoder(model)


def sentence_transformer_encoder(model: str = LOCAL_EMBED_MODEL) -> Encoder:
    """The local encoder for non-API models; requires the ``embed`` extra."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("embedding requires the 'embed' extra: pip install 'cc-steer[embed]'") from error

    transformer = SentenceTransformer(model)
    transformer.max_seq_length = MAX_SEQ_TOKENS

    class _Wrapped:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.asarray(transformer.encode(list(texts), normalize_embeddings=True), dtype=np.float32)

    return _Wrapped()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def exemplar_text(context_json: str) -> str | None:
    """The embedded key: the rendered context's tail, where the recent turns live.

    Tool-heavy turns can balloon a preview far past any encoder's context, so
    the text is capped from the end — the digest is computed over the capped
    text, keeping incrementality stable.
    """
    try:
        text = gate_text(ContextWindow.from_json(context_json))[-MAX_EMBED_CHARS:]
    except (ValueError, KeyError):
        return None
    return text or None


async def build_index(
    store: FeedbackStore, *, model: str = EMBED_MODEL, batch: int = 32, encoder: Encoder | None = None
) -> IndexReport:
    """Embeds every train-split steering event whose rendering changed.

    Incremental by content digest: an exemplar re-embeds only when its rendered
    context differs from what the stored vector was computed over. Test-split
    events are never indexed.

    Args:
        store: The open feedback store.
        model: The embedding model id, recorded per vector.
        batch: Encode batch size.
        encoder: The encoder to use; defaults to sentence-transformers.

    Returns:
        The :class:`IndexReport` for this pass.
    """
    rows = await store.sql(EXEMPLAR_EVENTS_QUERY)
    known = {str(row["dedup_key"]): str(row["text_digest"]) for row in await store.embeddings(model=model)}
    pending: list[tuple[str, str, str]] = []
    current = 0
    total = 0
    for row in rows:
        if split_of(str(row["session_id"])) != "train":
            continue
        if (text := exemplar_text(str(row["context_json"]))) is None:
            continue
        total += 1
        digest = text_digest(text)
        if known.get(str(row["dedup_key"])) == digest:
            current += 1
            continue
        pending.append((str(row["dedup_key"]), digest, text))
    if not pending:
        return IndexReport(embedded=0, current=current, total=total)
    if encoder is None and model.startswith("voyage"):
        vectors = await voyage_embed([text for _, _, text in pending], model=model)
        await store.record_embeddings(
            [
                (key, model, digest, int(vector.shape[-1]), vector.tobytes())
                for (key, digest, _), vector in zip(pending, vectors, strict=True)
            ]
        )
        return IndexReport(embedded=len(pending), current=current, total=total)
    active = encoder or sentence_transformer_encoder(model)
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        vectors = np.asarray(active.encode([text for _, _, text in chunk]), dtype=np.float32)
        await store.record_embeddings(
            [
                (key, model, digest, int(vector.shape[-1]), vector.tobytes())
                for (key, digest, _), vector in zip(chunk, vectors, strict=True)
            ]
        )
    return IndexReport(embedded=len(pending), current=current, total=total)


async def load_index(store: FeedbackStore, *, model: str = EMBED_MODEL) -> tuple[list[str], np.ndarray]:
    """Loads the whole index as ``(keys, unit-normalized matrix)``; empty when unbuilt."""
    rows = await store.embeddings(model=model)
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    keys = [str(row["dedup_key"]) for row in rows]

    def vector(row: Mapping[str, object]) -> np.ndarray:
        raw, dim = row["vector"], row["dim"]
        assert isinstance(raw, bytes) and isinstance(dim, int)
        return np.frombuffer(raw, dtype=np.float32, count=dim)

    matrix = np.stack([vector(row) for row in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return keys, matrix / np.maximum(norms, 1e-12)


def mmr_select(
    query: np.ndarray, matrix: np.ndarray, *, top_n: int = 32, k: int = 8, diversity: float = 0.5
) -> list[tuple[int, float]]:
    """Retrieves ``k`` rows by cosine, re-selected with maximal marginal relevance.

    Takes the ``top_n`` nearest rows, then greedily picks the row maximizing
    ``(1 - diversity) * sim(query, row) - diversity * max sim(row, picked)`` —
    pure top-k is redundant; the refiner wants similar-yet-varied precedents.

    Returns:
        ``(row index, query similarity)`` pairs, best first.
    """
    if matrix.size == 0 or k <= 0:
        return []
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    sims = matrix @ query
    pool = np.argsort(-sims)[: min(top_n, len(sims))].tolist()
    picked: list[int] = []
    while pool and len(picked) < k:
        if not picked:
            best = pool[0]
        else:
            cross = matrix[pool] @ matrix[picked].T
            scores = (1 - diversity) * sims[pool] - diversity * cross.max(axis=1)
            best = pool[int(np.argmax(scores))]
        picked.append(best)
        pool.remove(best)
    return [(index, float(sims[index])) for index in picked]


async def exemplars_for(
    store: FeedbackStore, hits: Sequence[tuple[str, float]], *, max_chars: int = 2000
) -> list[Exemplar]:
    """Hydrates retrieval hits into :class:`Exemplar` rows, preserving order."""
    if not hits:
        return []
    scores: Mapping[str, float] = dict(hits)
    keys = list(scores)
    marks = ",".join("?" for _ in keys)
    by_key = {
        str(row["dedup_key"]): row
        for row in await store.sql(f"{EXEMPLAR_DETAIL_QUERY} WHERE e.dedup_key IN ({marks})", keys)
    }
    exemplars = []
    for key, score in hits:
        if (row := by_key.get(key)) is None:
            continue
        exemplars.append(
            Exemplar(
                dedup_key=key,
                context_text=(exemplar_text(str(row["context_json"])) or "")[:max_chars],
                direction=str(row["direction"] or ""),
                verbatim=str(row["text"]),
                category=str(row["category"]),
                score=score,
            )
        )
    return exemplars
