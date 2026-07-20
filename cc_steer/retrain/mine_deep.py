"""The deep mechanical mine: sweep every un-mined session for detector candidates and
negative-enriched samples, persisting through the store with NO LLM triage.

Only ~744 of the ~16.7k mirror sessions were ever mined, and the 293-negative minority
class is the instrument's MDE bottleneck. This driver runs the detector pass
(:func:`cc_steer.detectors.detect`) for candidate positives and the negative sampler
(:func:`cc_steer.negatives.sample_negatives`) biased hard toward real-silence turns, and
persists everything the way the scan pipeline does — :meth:`FeedbackStore.record_file_scan`
and the gate-sample insert — but stops short of judging: the fable labels land later from the
E44 lane.

It is idempotent and resumable, the E30.5 driver contract: the store's recorded file mtimes
and ``sampled_session`` markers are the durable checkpoint, a live status json and an
append-only log ride alongside under ``--out``, and transcripts that land mid-sweep (a
concurrent rsync refresh) are picked up by the next re-glob pass rather than missed — a
half-written file that fails to parse is left unrecorded and retried, never poisoned in.

Run detached (mechanical, no spend)::

    python -m cc_steer.retrain.mine_deep --out ~/.cc-steer/experiments/deep_mine/out
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from cc_transcript import CLAUDE_PROJECTS_DIR
from cc_transcript.discovery import find_in

from cc_steer.detectors import detect
from cc_steer.negatives import MIN_TRANSCRIPT_BYTES, sample_negatives
from cc_steer.store import FeedbackStore
from cc_steer.watcher.live import scrubbed_events

if TYPE_CHECKING:
    from collections.abc import Sequence

MIRRORS_DIR: Path = Path.home() / ".cc-steer" / "mirrors"
STATUS_NAME = "mine_deep_status.json"
LOG_NAME = "mine_deep.log"
DETECTOR_MAX_PASSES = 8
DETECTOR_STATUS_EVERY = 200
NEGATIVE_MAX_PASSES = 1_000
DEEP_NEGATIVE_CHUNK = 500
# Bias hard toward the NO_STEER minority class: 50% more real-silence turns per session than the
# weekly pipeline's PER_SESSION=20, since deep mining exists to grow the negative floor.
DEEP_PER_SESSION = 30
NEGATIVE_KINDS = ("positive_window", "hard_negative", "random_negative")


@dataclass(frozen=True, slots=True)
class DeepMineReport:
    """One deep-mine pass's totals.

    Attributes:
        detector_files: Transcripts detected and recorded this run.
        detector_inserted: New feedback events the detector pass inserted.
        negative_sessions: Transcripts parsed for negatives this run.
        inserted: New gate samples by kind (``positive_window`` / ``hard_negative`` /
            ``random_negative``); the event-derived kinds stay 0 until the E44 lane judges.
    """

    detector_files: int
    detector_inserted: int
    negative_sessions: int
    inserted: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "detector_files": self.detector_files,
            "detector_inserted": self.detector_inserted,
            "negative_sessions": self.negative_sessions,
            "inserted": dict(self.inserted),
        }


def default_roots() -> tuple[Path, ...]:
    """The mirror corpus plus the live projects dir when each is present."""
    return tuple(root for root in (MIRRORS_DIR, CLAUDE_PROJECTS_DIR) if root.is_dir())


async def sweep_detectors(
    store: FeedbackStore, roots: Sequence[Path], *, out: Path | None = None, max_passes: int = DETECTOR_MAX_PASSES
) -> tuple[int, int]:
    """Detect and persist candidates for every un-mined transcript under ``roots``; return ``(files, inserted)``.

    Re-globs the roots against the store's recorded mtimes each pass, so a transcript that
    lands mid-sweep is caught by the following pass; a file that fails to parse (a half-written
    concurrent rsync) is skipped and left unrecorded to retry, and a pass that records nothing
    new ends the loop so unparseable files cannot spin it.
    """
    files = 0
    inserted = 0
    for _ in range(max_passes):
        known = await store.file_mtimes()
        batch = [(path, mtime) for root in roots for path, mtime in find_in(root, known_mtimes=known)]
        if not batch:
            break
        pass_files = 0
        for path, mtime in batch:
            try:
                candidates = detect(scrubbed_events(path))
            except (OSError, KeyError, ValueError, TypeError):
                continue
            inserted += await store.record_file_scan(str(path), mtime, candidates)
            files += 1
            pass_files += 1
            if out is not None and files % DETECTOR_STATUS_EVERY == 0:
                _write_status(
                    out, phase="detectors", status="running", detector_files=files, detector_inserted=inserted
                )
        if pass_files == 0:
            break
    return files, inserted


async def sweep_negatives(
    store: FeedbackStore,
    roots: Sequence[Path],
    *,
    seed: int = 1,
    per_session: int = DEEP_PER_SESSION,
    chunk: int = DEEP_NEGATIVE_CHUNK,
    min_bytes: int = MIN_TRANSCRIPT_BYTES,
    max_passes: int = NEGATIVE_MAX_PASSES,
    out: Path | None = None,
) -> tuple[dict[str, int], int]:
    """Negative-enrich until every candidate transcript is sampled; return ``(inserted, sessions)``.

    Chains :func:`~cc_steer.negatives.sample_negatives` in ``chunk``-session batches, each skipping
    the sessions already sampled, until a batch parses nothing new — so the whole un-mined corpus is
    swept without one unbounded pass. The event-derived kinds recompute from the (as-yet unjudged)
    store every batch and net zero; the random-negative floor is what grows.
    """
    inserted = dict.fromkeys(NEGATIVE_KINDS, 0)
    sessions = 0
    for _ in range(max_passes):
        report = await sample_negatives(
            store, roots, seed=seed, sessions=chunk, per_session=per_session, min_bytes=min_bytes
        )
        for kind, count in report.inserted.items():
            inserted[kind] += count
        sessions += report.sessions_sampled
        if out is not None:
            _write_status(out, phase="negatives", status="running", negative_sessions=sessions, inserted=inserted)
        if report.sessions_sampled == 0:
            break
    return inserted, sessions


async def deep_mine(
    store: FeedbackStore,
    roots: Sequence[Path],
    *,
    seed: int = 1,
    per_session: int = DEEP_PER_SESSION,
    negative_chunk: int = DEEP_NEGATIVE_CHUNK,
    min_bytes: int = MIN_TRANSCRIPT_BYTES,
    detectors: bool = True,
    negatives: bool = True,
    out: Path | None = None,
) -> DeepMineReport:
    """Run the detector pass then the negative-enriched pass over ``roots``, writing the status json.

    Detectors first so candidate positives are persisted, then negatives biased toward silence. No
    triage runs — the E44 lane labels later. Returns the pass totals.
    """
    if out is not None:
        _append_log(out, f"start roots={[str(root) for root in roots]} seed={seed} per_session={per_session}")
    detector_files, detector_inserted = await sweep_detectors(store, roots, out=out) if detectors else (0, 0)
    if out is not None:
        _append_log(out, f"detectors done: {detector_files} files, {detector_inserted} events")
    negative_inserted, negative_sessions = (
        await sweep_negatives(
            store, roots, seed=seed, per_session=per_session, chunk=negative_chunk, min_bytes=min_bytes, out=out
        )
        if negatives
        else (dict.fromkeys(NEGATIVE_KINDS, 0), 0)
    )
    report = DeepMineReport(detector_files, detector_inserted, negative_sessions, negative_inserted)
    if out is not None:
        _append_log(out, f"negatives done: {negative_sessions} sessions, {negative_inserted}")
        _write_status(out, phase="done", status="done", **report.as_dict())
    return report


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(out: Path, *, phase: str, status: str, **fields: object) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / STATUS_NAME).write_text(
        json.dumps({"phase": phase, "status": status, "ts": _timestamp(), **fields}, indent=2, sort_keys=True) + "\n"
    )


def _append_log(out: Path, line: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / LOG_NAME).open("a") as handle:
        handle.write(f"{_timestamp()} {line}\n")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cc_steer.retrain.mine_deep",
        description="Deep mechanical mine of un-mined sessions: detectors + negative enrichment, no triage.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Driver output dir for the status json and log.")
    parser.add_argument("--db", type=Path, default=None, help="Feedback DB path (default ~/.cc-steer/feedback.db).")
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        dest="roots",
        help="Transcript root to sweep (repeatable; default the mirror corpus).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Negative-sampling seed.")
    parser.add_argument("--per-session", type=int, default=DEEP_PER_SESSION, help="Random negatives per session.")
    parser.add_argument("--negative-chunk", type=int, default=DEEP_NEGATIVE_CHUNK, help="Sessions per negative batch.")
    parser.add_argument(
        "--min-bytes", type=int, default=MIN_TRANSCRIPT_BYTES, help="Transcript size floor for negative candidates."
    )
    parser.add_argument("--detectors-only", action="store_true", help="Run only the detector pass.")
    parser.add_argument("--negatives-only", action="store_true", help="Run only the negative-enrichment pass.")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> DeepMineReport:
    roots = tuple(args.roots) if args.roots else default_roots()
    if not roots:
        raise SystemExit(f"no transcript roots to sweep; pass --root or create {MIRRORS_DIR}")
    async with await FeedbackStore.open(args.db or FeedbackStore.default_path()) as store:
        return await deep_mine(
            store,
            roots,
            seed=args.seed,
            per_session=args.per_session,
            negative_chunk=args.negative_chunk,
            min_bytes=args.min_bytes,
            detectors=not args.negatives_only,
            negatives=not args.detectors_only,
            out=args.out,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """The ``python -m cc_steer.retrain.mine_deep`` entrypoint: sweep, write status, print totals."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = anyio.run(_run, args)
    except Exception as error:
        _write_status(args.out, phase="error", status="error", detail=f"{type(error).__name__}: {error}"[:300])
        _append_log(args.out, f"ERROR {type(error).__name__}: {error}")
        raise
    print(json.dumps(report.as_dict()))


if __name__ == "__main__":
    main()
