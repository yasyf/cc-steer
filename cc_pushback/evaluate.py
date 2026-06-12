"""Mechanical evaluation of the triage judge: golden-set gate, audit estimates, flips.

Everything here recomputes from raw verdicts — no derived metrics are stored. The
golden fixture is the one model-independent truth anchor; the auditor estimates
generalization on rows the fixture does not cover. The same seeded sampler that drew
the audit reproduces the uniform core here, so headline estimates stay unbiased by
the low-confidence oversample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript.judge import (
    AuditEstimate,
    Disagreement,
    FlipReport,
    GoldenFailure,
    GoldenResult,
    GoldenRow,
    Metrics,
    exact_upper_bound,
    flip_pairs,
    golden_result,
    sample_audit,
)

from cc_pushback.triage import AUDIT_VERSION, AUDITOR, JUDGE, KIND_QUOTAS, PROMPT_VERSION, REMAINDER_KIND

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_pushback.store import FeedbackStore

__all__ = [
    "GOLDEN_PATH",
    "AuditEstimate",
    "Disagreement",
    "FlipReport",
    "GoldenFailure",
    "GoldenResult",
    "GoldenRow",
    "Metrics",
    "estimate",
    "evaluate",
    "exact_upper_bound",
    "flip_report",
    "golden_result",
    "golden_sha256",
    "load_golden",
]

GOLDEN_PATH = Path(__file__).with_name("golden_triage.json")


def load_golden(path: Path = GOLDEN_PATH) -> tuple[GoldenRow, ...]:
    """Loads the frozen golden fixture, mapping its labels to booleans.

    The on-disk fixture is frozen with ``"pushback"``/``"noise"`` labels; the
    mining domain's :class:`GoldenRow` carries ``expected`` as a bool, so the
    mapping happens here at load time.

    Args:
        path: The fixture file, defaulting to the packaged one.

    Returns:
        The fixture's rows.
    """
    return tuple(GoldenRow(**row | {"expected": row["expected"] == "pushback"}) for row in json.loads(path.read_text()))


def golden_sha256(path: Path = GOLDEN_PATH) -> str:
    """Returns the fixture file's SHA-256, so any edit is visible in eval output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(rows: Sequence[Mapping[str, object]], audits: Mapping[str, Mapping[str, object]]) -> AuditEstimate:
    audited = [audits[key] for row in rows if (key := str(row["dedup_key"])) in audits]
    return AuditEstimate(audited=len(audited), hits=sum(bool(verdict["accepted"]) for verdict in audited))


async def evaluate(
    store: FeedbackStore,
    *,
    prompt_version: int = PROMPT_VERSION,
    seed: int = 1,
    accepts: int = 60,
    rejects: int = 60,
    golden_path: Path = GOLDEN_PATH,
) -> Metrics:
    """Computes the full mechanical evaluation of one prompt version's verdicts.

    Reproduces the audit's uniform core with the same seeded sampler the audit
    used, so the headline precision/contamination estimates exclude the
    low-confidence oversample by construction.

    Args:
        store: The open feedback store.
        prompt_version: The judge prompt version to evaluate.
        seed: The sampling seed the audit ran with.
        accepts: The audit's accept budget.
        rejects: The audit's reject budget.
        golden_path: The golden fixture to gate against.

    Returns:
        The assembled metrics.

    Raises:
        LookupError: If any golden row is missing from the corpus.
    """
    judged = await store.judged(role=JUDGE, prompt_version=prompt_version)
    corpus_keys = await store.dedup_keys()
    judge_by_key = {str(row["dedup_key"]): row for row in judged}
    audits = {str(row["dedup_key"]): row for row in await store.judged(role=AUDITOR, prompt_version=AUDIT_VERSION)}
    core = sample_audit(
        judged, accepts=accepts, rejects=rejects, seed=seed, quotas=KIND_QUOTAS, remainder_kind=REMAINDER_KIND
    ).core
    accepted_all = [row for row in judged if row["accepted"]]
    rejected_all = [row for row in judged if not row["accepted"]]
    by_kind: dict[str, tuple[int, int]] = {}
    for row in judged:
        kind = str(row["source_kind"])
        seen, accepted = by_kind.get(kind, (0, 0))
        by_kind[kind] = (seen + 1, accepted + bool(row["accepted"]))
    return Metrics(
        prompt_version=prompt_version,
        total=len(corpus_keys),
        judged=len(judged),
        accepted=len(accepted_all),
        golden=golden_result(load_golden(golden_path), corpus_keys, judge_by_key, golden_sha256(golden_path)),
        core_accepts=estimate([row for row in core if row["accepted"]], audits),
        core_rejects=estimate([row for row in core if not row["accepted"]], audits),
        pool_accepts=estimate(accepted_all, audits),
        pool_rejects=estimate(rejected_all, audits),
        by_kind=by_kind,
        disagreements=tuple(
            Disagreement(
                dedup_key=key,
                source_kind=str(row["source_kind"]),
                text=str(row["text"]),
                judge_category=str(row["category"]),
                auditor_category=str(audit["category"]),
                judge_rationale=str(row["rationale"]),
                auditor_rationale=str(audit["rationale"]),
            )
            for key, row in judge_by_key.items()
            if (audit := audits.get(key)) is not None and bool(audit["accepted"]) is not bool(row["accepted"])
        ),
    )


async def flip_report(store: FeedbackStore, *, from_version: int, to_version: int) -> FlipReport:
    """Compares two prompt versions' verdicts row by row.

    Args:
        store: The open feedback store.
        from_version: The earlier prompt version.
        to_version: The later prompt version.

    Returns:
        The overlap size and every side-changing row.
    """
    return flip_pairs(
        await store.judged(role=JUDGE, prompt_version=from_version),
        await store.judged(role=JUDGE, prompt_version=to_version),
    )
