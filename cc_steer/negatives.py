"""The gate-sampling stage: positives rewound turn by turn, negatives mined at scale.

The gate trains on turn-level decisions, so this stage materializes three sample
kinds into the ``gate_sample`` table, all idempotent by ``sample_key``:

- ``positive_window``: every accepted steering event, rewound 0 through
  ``W_MAX - 1`` turns before the intervention — human interruption timing is
  noisy and late, so training labels come as windows, not exact turns. Rewinding
  truncates the stored context; no transcript is re-read.
- ``hard_negative``: judge-rejected detector hits — moments that looked like
  steering but were noise, the gate's hardest boundary.
- ``random_negative``: assistant turns sampled from stretches the user let run,
  drawn from live and mirror transcripts, away from any detected event. These
  are the cheap bulk that sets the class ratio; sessions already sampled are
  skipped, so repeated passes extend coverage instead of re-parsing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript import TranscriptParser
from cc_transcript.activity import SessionActivity
from cc_transcript.context import ContextWindow
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import sample_windows

from cc_steer.rendering import has_substantive_gate_content, truncated

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_steer.store import FeedbackStore

KINDS = ("positive_window", "hard_negative", "random_negative")
W_MAX = 6
EXCLUSION_RADIUS = 6
MIN_TRANSCRIPT_BYTES = 32_768
PER_SESSION = 20

EVENT_WINDOWS_QUERY = """
SELECT e.dedup_key, e.session_id, e.event_uuid, e.occurred_at, e.context_json, j.is_steering
FROM feedback_events e
JOIN latest_judge j ON j.dedup_key = e.dedup_key
WHERE e.quarantined_reason IS NULL
ORDER BY e.id
"""

ANCHORS_QUERY = "SELECT session_id, event_uuid FROM feedback_events"


@dataclass(frozen=True, slots=True)
class GateSample:
    """One gate training sample: a context window and its provenance.

    Attributes:
        sample_key: The idempotence key (``pos:<dedup>:<k>``, ``hard:<dedup>``,
            or ``rand:<session>:<anchor>``).
        kind: One of :data:`KINDS`.
        dedup_key: The parent feedback event, when one exists.
        session_id: The session the window came from.
        anchor_uuid: The event the window is anchored on.
        occurred_at: The anchor's timestamp, when known.
        offset_turns: How many turns before the intervention the window ends
            (0 or negative); always 0 for negatives.
        window_json: The serialized :class:`ContextWindow`.
        seed: The sampling seed that produced the row.
    """

    sample_key: str
    kind: str
    dedup_key: str | None
    session_id: str
    anchor_uuid: str
    occurred_at: str | None
    offset_turns: int
    window_json: str
    seed: int


@dataclass(frozen=True, slots=True)
class NegativesReport:
    """The outcome of one sampling pass.

    Attributes:
        inserted: Newly inserted rows keyed by kind.
        sessions_sampled: How many transcripts were parsed for random negatives.
    """

    inserted: Mapping[str, int]
    sessions_sampled: int


def event_samples(rows: Sequence[Mapping[str, object]], *, offsets: int = W_MAX, seed: int = 0) -> list[GateSample]:
    """Builds positive windows and hard negatives from judged events' stored context.

    Steering events yield one row per rewind offset (0 through ``offsets - 1``,
    stopping once nothing remains to rewind or the rewound window renders no
    substantive gate content — a deeper rewind can only lose more, so it never
    emits an empty positive); rejected events yield one offset-0 hard negative
    each. The insert choke point (:meth:`FeedbackStore.record_gate_samples`) is
    the backstop that drops any empty sample of any kind.
    """
    samples: list[GateSample] = []
    for row in rows:
        try:
            window = ContextWindow.from_json(str(row["context_json"]))
        except (ValueError, KeyError):
            continue
        base = {
            "dedup_key": str(row["dedup_key"]),
            "session_id": str(row["session_id"]),
            "anchor_uuid": str(row["event_uuid"]),
            "occurred_at": str(row["occurred_at"]) if row["occurred_at"] is not None else None,
            "seed": seed,
        }
        if bool(row["is_steering"]):
            for k in range(offsets):
                if (rewound := truncated(window, k)) is None or not has_substantive_gate_content(rewound):
                    break
                samples.append(
                    GateSample(
                        sample_key=f"pos:{row['dedup_key']}:{k}",
                        kind="positive_window",
                        offset_turns=-k,
                        window_json=rewound.to_json(),
                        **base,
                    )
                )
        else:
            samples.append(
                GateSample(
                    sample_key=f"hard:{row['dedup_key']}",
                    kind="hard_negative",
                    offset_turns=0,
                    window_json=window.to_json(),
                    **base,
                )
            )
    return samples


def random_samples(
    activity: SessionActivity,
    exclude: Sequence[EventRef],
    *,
    per_session: int,
    seed: int,
) -> list[GateSample]:
    """Samples assistant-anchored windows from one session's quiet stretches."""
    windows = sample_windows(activity, n=per_session, exclude=exclude, exclusion_radius=EXCLUSION_RADIUS, seed=seed)
    return [
        GateSample(
            sample_key=f"rand:{window.anchor.session_id}:{window.anchor.event_uuid}",
            kind="random_negative",
            dedup_key=None,
            session_id=str(window.anchor.session_id),
            anchor_uuid=str(window.anchor.event_uuid),
            occurred_at=None,
            offset_turns=0,
            window_json=window.to_json(),
            seed=seed,
        )
        for window in windows
    ]


def transcript_candidates(roots: Sequence[Path], *, min_bytes: int = MIN_TRANSCRIPT_BYTES) -> list[Path]:
    """Main-session transcripts under ``roots`` big enough to hold real work.

    Half the corpus is one-turn stubs with no completed turn to sample — the
    size floor keeps the parse budget on sessions the user actually let run.
    """
    paths: set[Path] = set()
    for root in roots:
        if root.is_dir():
            paths.update(
                path
                for path in root.rglob("*.jsonl")
                if not path.name.startswith("agent-")
                if path.stat().st_size >= min_bytes
            )
    return sorted(paths)


async def sample_negatives(
    store: FeedbackStore,
    roots: Sequence[Path],
    *,
    seed: int = 1,
    sessions: int = 400,
    per_session: int = PER_SESSION,
    offsets: int = W_MAX,
    min_bytes: int = MIN_TRANSCRIPT_BYTES,
    resample: bool = False,
) -> NegativesReport:
    """Runs one sampling pass: event-derived samples plus budgeted random negatives.

    Event-derived samples (positive windows, hard negatives) are recomputed from
    the store every pass and deduped by key, so they track the judged corpus.
    Random negatives parse at most ``sessions`` transcripts this pass, chosen
    deterministically by ``seed`` from the candidates under ``roots`` that carry
    no random sample yet; every detected event's anchor is excluded with a
    ``EXCLUSION_RADIUS``-turn radius so no negative sits near a real intervention.

    Args:
        store: The open feedback store.
        roots: Transcript directories to mine for random negatives.
        seed: Deterministic sampling seed.
        sessions: Maximum transcripts to parse this pass.
        per_session: Random negatives to draw per transcript.
        offsets: Rewind width for positive windows.
        min_bytes: Transcript size floor for random-negative candidates.
        resample: Revisit sessions that already carry random samples (new draws
            still dedupe by key); by default they are skipped.

    Returns:
        The :class:`NegativesReport` with per-kind insert counts.
    """
    cur = await store.store.conn.execute(EVENT_WINDOWS_QUERY)
    event_rows = [dict(row) async for row in cur]
    inserted = {
        "positive_window": 0,
        "hard_negative": 0,
        "random_negative": 0,
    }
    from_events = event_samples(event_rows, offsets=offsets, seed=seed)
    for kind in ("positive_window", "hard_negative"):
        batch = [sample for sample in from_events if sample.kind == kind]
        inserted[kind] = await store.record_gate_samples(batch)

    anchor_cur = await store.store.conn.execute(ANCHORS_QUERY)
    anchors: dict[str, list[EventRef]] = {}
    async for row in anchor_cur:
        session = str(row["session_id"])
        anchors.setdefault(session, []).append(
            EventRef(session_id=SessionId(session), event_uuid=EventUuid(str(row["event_uuid"])), tool_use_id=None)
        )

    done = set() if resample else await store.negative_sessions()
    candidates = [path for path in transcript_candidates(roots, min_bytes=min_bytes) if path.stem not in done]
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(sessions, len(candidates)))
    marked: list[str] = []
    async for parsed in TranscriptParser.stream_transcripts([(path, path.stat().st_mtime) for path in chosen]):
        session_id = SessionId(Path(parsed.path).stem)
        activity = SessionActivity.from_events(session_id, parsed.events)
        if not activity.turns:
            continue
        batch = random_samples(activity, anchors.get(str(session_id), []), per_session=per_session, seed=seed)
        inserted["random_negative"] += await store.record_gate_samples(batch)
        marked.append(str(session_id))
    await store.mark_sessions_sampled(marked)
    return NegativesReport(inserted=inserted, sessions_sampled=len(marked))
