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
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cc_pushback.triage import AUDIT_VERSION, AUDITOR, JUDGE, PROMPT_VERSION, sample_audit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_pushback.store import FeedbackStore

GOLDEN_PATH = Path(__file__).with_name("golden_triage.json")


@dataclass(frozen=True, slots=True)
class GoldenRow:
    """One frozen, hand-labeled row of the golden regression set.

    Attributes:
        dedup_key: The content-derived key joining the row to ``feedback_events``.
        source_kind: The detector that produced the row.
        text: The verbatim message, kept for human review of the fixture.
        expected: The frozen label.
        note: One clause recording why the label holds.
    """

    dedup_key: str
    source_kind: str
    text: str
    expected: Literal["pushback", "noise"]
    note: str


@dataclass(frozen=True, slots=True)
class GoldenFailure:
    """One golden row the judge got wrong (or has not judged).

    Attributes:
        dedup_key: The failing row's key.
        expected: The frozen label.
        category: The judge's category, or ``None`` when the row is unjudged.
        text: The verbatim message.
    """

    dedup_key: str
    expected: str
    category: str | None
    text: str


@dataclass(frozen=True, slots=True)
class GoldenResult:
    """The golden-set gate outcome.

    Attributes:
        total: The fixture's row count.
        passed: How many rows the judge labeled to match the fixture.
        sha256: The fixture file's digest, printed so any edit is visible.
        failures: Every mismatched or unjudged row.
    """

    total: int
    passed: int
    sha256: str
    failures: tuple[GoldenFailure, ...]


@dataclass(frozen=True, slots=True)
class AuditEstimate:
    """One binomial estimate from audited rows.

    Attributes:
        audited: How many rows of the population carry an auditor verdict.
        hits: How many of those the auditor called pushback.
    """

    audited: int
    hits: int

    @property
    def rate(self) -> float | None:
        """``hits / audited``, or ``None`` when nothing is audited."""
        return self.hits / self.audited if self.audited else None


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One audited row where the auditor's side differs from the judge's.

    Attributes:
        dedup_key: The row's key.
        source_kind: The detector that produced the row.
        text: The verbatim message.
        judge_category: The judge's category.
        auditor_category: The auditor's category.
    """

    dedup_key: str
    source_kind: str
    text: str
    judge_category: str
    auditor_category: str


@dataclass(frozen=True, slots=True)
class Metrics:
    """The full mechanical evaluation of one prompt version.

    Attributes:
        prompt_version: The judge prompt version evaluated.
        total: The corpus row count.
        judged: How many rows carry a judge verdict at this version.
        accepted: How many of those are pushback.
        golden: The golden-set gate outcome.
        core_accepts: Audited precision numerator/denominator over the uniform core.
        core_rejects: Audited contamination numerator/denominator over the uniform core.
        pool_accepts: The same estimate over every audited accept (cumulative pool).
        pool_rejects: The same estimate over every audited reject (cumulative pool).
        by_kind: ``(judged, accepted)`` counts per source kind, descriptive only.
        disagreements: Every audited row where auditor and judge disagree.
    """

    prompt_version: int
    total: int
    judged: int
    accepted: int
    golden: GoldenResult
    core_accepts: AuditEstimate
    core_rejects: AuditEstimate
    pool_accepts: AuditEstimate
    pool_rejects: AuditEstimate
    by_kind: Mapping[str, tuple[int, int]]
    disagreements: tuple[Disagreement, ...]

    @property
    def precision(self) -> float | None:
        """Audited precision over the uniform core's accepts."""
        return self.core_accepts.rate

    @property
    def contamination(self) -> float | None:
        """Audited genuine-pushback rate over the uniform core's rejects."""
        return self.core_rejects.rate

    @property
    def contamination_upper(self) -> float | None:
        """The exact one-sided 95% upper bound on contamination."""
        est = self.core_rejects
        return exact_upper_bound(est.hits, est.audited) if est.audited else None

    @property
    def recall_hat(self) -> float | None:
        """The derived estimate of the fraction of genuine pushback accepted."""
        match (self.precision, self.contamination):
            case (None, _) | (_, None):
                return None
            case (p, c):
                rejected = self.judged - self.accepted
                genuine = self.accepted * p + rejected * c
                return self.accepted * p / genuine if genuine else None


@dataclass(frozen=True, slots=True)
class Flip:
    """One row whose pushback-vs-noise side changed between prompt versions.

    Attributes:
        dedup_key: The row's key.
        text: The verbatim message.
        from_category: The category at the earlier version.
        to_category: The category at the later version.
    """

    dedup_key: str
    text: str
    from_category: str
    to_category: str


@dataclass(frozen=True, slots=True)
class FlipReport:
    """The verdict churn between two prompt versions.

    Attributes:
        common: How many rows carry judge verdicts at both versions.
        flips: Every row whose side changed.
    """

    common: int
    flips: tuple[Flip, ...]

    @property
    def rate(self) -> float | None:
        """``len(flips) / common``, or ``None`` when no rows overlap."""
        return len(self.flips) / self.common if self.common else None


def exact_upper_bound(hits: int, n: int, alpha: float = 0.05) -> float:
    """Returns the exact (Clopper-Pearson) one-sided upper confidence bound.

    The smallest rate ``p`` such that observing ``hits`` or fewer successes in
    ``n`` trials has probability at most ``alpha`` — the rule of three's exact
    generalization.

    Args:
        hits: The observed success count.
        n: The trial count.
        alpha: The one-sided significance level.

    Returns:
        The upper bound on the true rate.
    """
    if hits >= n:
        return 1.0
    lo, hi = hits / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(comb(n, k) * mid**k * (1 - mid) ** (n - k) for k in range(hits + 1)) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def load_golden(path: Path = GOLDEN_PATH) -> tuple[GoldenRow, ...]:
    """Loads the frozen golden fixture.

    Args:
        path: The fixture file, defaulting to the packaged one.

    Returns:
        The fixture's rows.
    """
    return tuple(GoldenRow(**row) for row in json.loads(path.read_text()))


def golden_sha256(path: Path = GOLDEN_PATH) -> str:
    """Returns the fixture file's SHA-256, so any edit is visible in eval output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def golden_failure(row: GoldenRow, verdict: Mapping[str, object] | None) -> GoldenFailure | None:
    match verdict:
        case None:
            return GoldenFailure(dedup_key=row.dedup_key, expected=row.expected, category=None, text=row.text)
        case v if bool(v["is_pushback"]) is not (row.expected == "pushback"):
            return GoldenFailure(
                dedup_key=row.dedup_key, expected=row.expected, category=str(v["category"]), text=row.text
            )
        case _:
            return None


def golden_result(
    golden: Sequence[GoldenRow], corpus_keys: set[str], judge_by_key: Mapping[str, Mapping[str, object]], sha256: str
) -> GoldenResult:
    if missing := [row.dedup_key for row in golden if row.dedup_key not in corpus_keys]:
        raise LookupError(f"golden rows missing from the corpus (drift): {missing}")
    failures = tuple(
        failure for row in golden if (failure := golden_failure(row, judge_by_key.get(row.dedup_key))) is not None
    )
    return GoldenResult(total=len(golden), passed=len(golden) - len(failures), sha256=sha256, failures=failures)


def estimate(rows: Sequence[Mapping[str, object]], audits: Mapping[str, Mapping[str, object]]) -> AuditEstimate:
    audited = [audits[key] for row in rows if (key := str(row["dedup_key"])) in audits]
    return AuditEstimate(audited=len(audited), hits=sum(bool(verdict["is_pushback"]) for verdict in audited))


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
    audits = {
        str(row["dedup_key"]): row for row in await store.judged(role=AUDITOR, prompt_version=AUDIT_VERSION)
    }
    core = sample_audit(judged, accepts=accepts, rejects=rejects, seed=seed).core
    accepted_all = [row for row in judged if row["is_pushback"]]
    rejected_all = [row for row in judged if not row["is_pushback"]]
    by_kind: dict[str, tuple[int, int]] = {}
    for row in judged:
        kind = str(row["source_kind"])
        seen, accepted = by_kind.get(kind, (0, 0))
        by_kind[kind] = (seen + 1, accepted + bool(row["is_pushback"]))
    return Metrics(
        prompt_version=prompt_version,
        total=len(corpus_keys),
        judged=len(judged),
        accepted=len(accepted_all),
        golden=golden_result(load_golden(golden_path), corpus_keys, judge_by_key, golden_sha256(golden_path)),
        core_accepts=estimate([row for row in core if row["is_pushback"]], audits),
        core_rejects=estimate([row for row in core if not row["is_pushback"]], audits),
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
            )
            for key, row in judge_by_key.items()
            if (audit := audits.get(key)) is not None and bool(audit["is_pushback"]) is not bool(row["is_pushback"])
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
    earlier = {str(row["dedup_key"]): row for row in await store.judged(role=JUDGE, prompt_version=from_version)}
    later = {str(row["dedup_key"]): row for row in await store.judged(role=JUDGE, prompt_version=to_version)}
    common = earlier.keys() & later.keys()
    return FlipReport(
        common=len(common),
        flips=tuple(
            Flip(
                dedup_key=key,
                text=str(later[key]["text"]),
                from_category=str(earlier[key]["category"]),
                to_category=str(later[key]["category"]),
            )
            for key in sorted(common)
            if bool(earlier[key]["is_pushback"]) is not bool(later[key]["is_pushback"])
        ),
    )
