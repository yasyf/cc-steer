"""The E10.B warrant audit: re-label the frame's warrant calls with the human-gated judge panel.

The watcher's frozen eval calls each row "warranted" (a true prose-corrective steer) or not, and the
promotion gate trains and gates on those labels. This module audits them. It re-derives the authored
golden sample, proves its labeled rows came from it, then spends the same enforced judge path the
promotion gate uses (:func:`~cc_steer.retrain.judged.panel_labels` behind the golden gate, then
:func:`~athome.research.judge.judge_candidates` with embedded health controls) to grade whether each
labeled row's steer was warranted — twice, under two position-debias seeds. A row the judge grades
warranted under both seeds is corrected to warranted, not-warranted under both to not, and a split is
left uncertain and flagged (the frame label stands).

The audit reports two things it journals under ``watcher-warrant-audit``: per-stratum Wilson
confidence intervals on the frame's false-positive rate (warranted rows the judge overturns) and
false-negative rate (negatives the judge promotes), and a paired AUC — how well the incumbent's fire
scores rank the frame labels versus the judge-corrected labels over the same rows. A sha-stamped
sidecar under ``<eval>/audit/`` records every row's votes and the judge identity, self-hashed over its
canonical rows so a later reader can prove the record was not edited.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from athome.research.common import canonical_json
from athome.research.golden import MANIFEST_NAME, GoldenProof, agreement, prove_gate
from athome.research.judge import (
    Judge,
    JudgeRow,
    Pairwise,
    Vote,
    VoteCache,
    VoteContext,
    judge_candidates,
)

from cc_steer.retrain import evalset, golden, judged, promotion

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cc_steer.retrain.evalset import EvalFrame

AUDIT_SEEDS: tuple[int, ...] = (1729, 2718)
VOTES_NAME = "votes.json"
AUDIT_DIRNAME = "audit"
SIDECAR_NAME = "warrant-audit-v1.json"
JOURNAL_COMPONENT = "watcher-warrant-audit"
WILSON_Z = 1.96


class WarrantAuditError(RuntimeError):
    """The golden packet's audit provenance drifted from the recomputed sample; the audit cannot proceed."""


@dataclass(frozen=True, slots=True)
class Corrected:
    """One audited golden row: the frame's call, the judge's two votes, and the reconciled label.

    Attributes:
        row_id: The frame row id.
        stratum: The packet stratum the row was drawn from (``warranted`` or ``negative``).
        frame_warranted: The frame's own warrant call for the row (``stratum == "warranted"``).
        human: The human golden label.
        corrected: The judge-reconciled warrant label; the frame label stands when uncertain.
        certain: Whether the judge voted unanimously across the seeds (an uncertain row is flagged).
        votes: The judge's vote per seed, in ``seeds`` order.
    """

    row_id: str
    stratum: str
    frame_warranted: bool
    human: bool
    corrected: bool
    certain: bool
    votes: tuple[Vote, ...]


@dataclass(frozen=True, slots=True)
class AuditResult:
    """The reduced warrant audit: per-stratum error rates with Wilson intervals and the paired AUC."""

    version: str
    threshold: float
    seeds: tuple[int, ...]
    corrected: tuple[Corrected, ...]
    fp: int
    fn: int
    fp_ci: tuple[float, float]
    fn_ci: tuple[float, float]
    n_warranted: int
    n_negative: int
    auc_frame: float
    auc_corrected: float

    @property
    def n_flagged(self) -> int:
        return sum(1 for row in self.corrected if not row.certain)


def wilson_interval(successes: int, total: int, *, z: float = WILSON_Z) -> tuple[float, float]:
    """The Wilson score interval for a binomial proportion — ``(0.0, 0.0)`` for an empty stratum.

    Example:
        >>> wilson_interval(2, 15)
    """
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total))
    return (center - half, center + half)


async def run_warrant_audit(*, root: Path | None = None, seeds: tuple[int, ...] = AUDIT_SEEDS) -> str:
    """Re-label the authored golden packet under two seeds and journal the warrant-audit verdict.

    Recomputes the audit sample and refuses unless the packet's manifest provenance matches it and its
    rows are a subset of it, loads the human-labeled packet (raising before any spend when unlabeled),
    proves the golden gate with a panel vote, grades every labeled row through the enforced
    :func:`~athome.research.judge.judge_candidates` path once per seed, writes the sha-stamped sidecar,
    and journals the per-stratum Wilson intervals and the paired AUC.

    Raises:
        WarrantAuditError: the packet's audit provenance drifted from the recomputed sample.
        GoldenGateViolation: the packet is unlabeled or drifted, or the panel disagrees with the human
            labels below the gate floor.
    """
    frame = evalset.EvalFrame.load(root=root)
    version, fire_scores, threshold = golden.incumbent_fire(frame, root=root)
    directory = judged.golden_dir(root=root)
    manifest = json.loads((directory / MANIFEST_NAME).read_text())
    _cross_check(manifest, golden.audit_sample(frame, fire_scores, threshold, n=golden.SAMPLE_N, seed=golden.SEED), version=version, threshold=threshold)
    loaded = await judged.load_golden(directory)
    judge = Judge.bind(Pairwise, backend=judged.JUDGE_BACKEND)
    proof = prove_gate(report=agreement(loaded.human, await judged.panel_labels(judge, loaded, seed=seeds[0])), manifest=loaded.manifest)
    corrected = await _corrected_labels(judge, loaded, frame, proof, manifest, seeds=seeds, root=root)
    result = _reduce(corrected, frame, fire_scores, version=version, threshold=threshold, seeds=seeds)
    _write_sidecar(result, judge=judge, manifest=manifest, frame=frame, root=root)
    return _journal(result, frame, root=root)


async def _corrected_labels(
    judge: Judge[Pairwise],
    loaded: judged.Golden,
    frame: EvalFrame,
    proof: GoldenProof,
    manifest: dict[str, object],
    *,
    seeds: tuple[int, ...],
    root: Path | None,
) -> list[Corrected]:
    stratum_of = {str(row["row_id"]): str(row["stratum"]) for row in _rows(manifest)}
    items = tuple(loaded.contexts.items())
    rows = tuple(
        JudgeRow(row_id=row_id, candidate=judged.render_action(fired=True, context=window), reference=judged.render_action(fired=False, context=window))
        for row_id, window in items
    )
    cache = VoteCache.open(evalset.eval_root(root) / VOTES_NAME)
    votes = {
        seed: await judge_candidates(
            judge,
            rows,
            generator_family=judged.GENERATOR_FAMILY,
            controls=judged.control_pairs(loaded.contexts),
            golden=proof,
            context=VoteContext(prompt_version=judged.PROMPT_VERSION, digest=frame.digest),
            build_prompt=judged.build_prompt,
            seed=seed,
            cache=cache,
        )
        for seed in seeds
    }
    return [
        _correct(row_id, stratum_of[row_id], loaded.human[row_id], tuple(votes[seed][i] for seed in seeds))
        for i, (row_id, _window) in enumerate(items)
    ]


def _correct(row_id: str, stratum: str, human: bool, votes: tuple[Vote, ...]) -> Corrected:
    frame_warranted = stratum == golden.WARRANTED
    if all(vote is Vote.WIN for vote in votes):
        corrected, certain = True, True
    elif all(vote is Vote.LOSS for vote in votes):
        corrected, certain = False, True
    else:
        corrected, certain = frame_warranted, False
    return Corrected(row_id=row_id, stratum=stratum, frame_warranted=frame_warranted, human=human, corrected=corrected, certain=certain, votes=votes)


def _reduce(
    corrected: list[Corrected], frame: EvalFrame, fire_scores: np.ndarray, *, version: str, threshold: float, seeds: tuple[int, ...]
) -> AuditResult:
    warranted = [row for row in corrected if row.stratum == golden.WARRANTED]
    negative = [row for row in corrected if row.stratum == golden.NEGATIVE]
    fp = sum(1 for row in warranted if row.certain and not row.corrected)
    fn = sum(1 for row in negative if row.certain and row.corrected)
    index_of = {row_id: i for i, row_id in enumerate(frame.ids)}
    scores = [float(fire_scores[index_of[row.row_id]]) for row in corrected]
    return AuditResult(
        version=version,
        threshold=threshold,
        seeds=seeds,
        corrected=tuple(corrected),
        fp=fp,
        fn=fn,
        fp_ci=wilson_interval(fp, len(warranted)),
        fn_ci=wilson_interval(fn, len(negative)),
        n_warranted=len(warranted),
        n_negative=len(negative),
        auc_frame=_auc([row.frame_warranted for row in corrected], scores),
        auc_corrected=_auc([row.corrected for row in corrected], scores),
    )


def _cross_check(manifest: dict[str, object], sample: tuple[golden.AuditRow, ...], *, version: str, threshold: float) -> None:
    audit = cast("Mapping[str, object]", manifest["audit"])
    expected = {"seed": golden.SEED, "n": golden.SAMPLE_N, "incumbent_version": version, "stratum_counts": golden.stratum_counts(sample)}
    if {key: audit[key] for key in expected} != expected or not math.isclose(float(cast("float", audit["incumbent_fire_threshold"])), threshold, abs_tol=1e-12):
        raise WarrantAuditError(f"golden audit provenance {dict(audit)} does not match the sample recomputed under incumbent {version}")
    packet_ids = {str(row["row_id"]) for row in _rows(manifest)}
    if not packet_ids <= (sample_ids := {row.row_id for row in sample}):
        raise WarrantAuditError(f"golden packet rows {sorted(packet_ids - sample_ids)} are not a subset of the recomputed audit sample")


def _rows(manifest: dict[str, object]) -> Sequence[Mapping[str, object]]:
    return cast("Sequence[Mapping[str, object]]", manifest["rows"])


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return promotion.sentinel_auc(np.array(labels, dtype=bool), np.array(scores, dtype=float))


def _write_sidecar(result: AuditResult, *, judge: Judge[Pairwise], manifest: dict[str, object], frame: EvalFrame, root: Path | None) -> None:
    rows = sorted(
        (
            {
                "row_id": row.row_id,
                "stratum": row.stratum,
                "frame_warranted": row.frame_warranted,
                "human_label": row.human,
                "corrected": row.corrected,
                "certain": row.certain,
                "votes": {str(seed): vote.value for seed, vote in zip(result.seeds, row.votes, strict=True)},
            }
            for row in result.corrected
        ),
        key=lambda row: row["row_id"],
    )
    identity = judge.identity
    sidecar = {
        "meta": {
            "dataset_digest": frame.digest,
            "watcher_eval_sha256": json.loads((evalset.eval_root(root) / evalset.MANIFEST_NAME).read_text())[evalset.WATCHER_EVAL_NAME],
            "prompt_version": judged.PROMPT_VERSION,
            "seeds": list(result.seeds),
            "judge": {"provider": identity.provider, "model": identity.model, "verdict_schema_sha256": identity.verdict_schema_sha256},
            "incumbent_version": result.version,
            "incumbent_fire_threshold": result.threshold,
            "packet_rows_sha256": manifest["rows_sha256"],
            "self_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
        },
        "rows": rows,
        "strata": {
            golden.WARRANTED: {"n": result.n_warranted, "fp": result.fp, "fp_rate": _rate(result.fp, result.n_warranted), "fp_ci": list(result.fp_ci)},
            golden.NEGATIVE: {"n": result.n_negative, "fn": result.fn, "fn_rate": _rate(result.fn, result.n_negative), "fn_ci": list(result.fn_ci)},
        },
        "paired_auc": {"frame": result.auc_frame, "corrected": result.auc_corrected, "delta": result.auc_corrected - result.auc_frame},
    }
    path = evalset.eval_root(root) / AUDIT_DIRNAME / SIDECAR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def _journal(result: AuditResult, frame: EvalFrame, *, root: Path | None) -> str:
    verdict = (
        f"{result.version}: FP {result.fp}/{result.n_warranted} [{result.fp_ci[0]:.2f},{result.fp_ci[1]:.2f}], "
        f"FN {result.fn}/{result.n_negative} [{result.fn_ci[0]:.2f},{result.fn_ci[1]:.2f}], "
        f"paired AUC {result.auc_frame:.3f} -> {result.auc_corrected:.3f}, {result.n_flagged} uncertain"
    )
    metrics = {
        "fp": float(result.fp),
        "fp_rate": _rate(result.fp, result.n_warranted),
        "fp_ci_lo": result.fp_ci[0],
        "fp_ci_hi": result.fp_ci[1],
        "fn": float(result.fn),
        "fn_rate": _rate(result.fn, result.n_negative),
        "fn_ci_lo": result.fn_ci[0],
        "fn_ci_hi": result.fn_ci[1],
        "auc_frame": result.auc_frame,
        "auc_corrected": result.auc_corrected,
        "auc_delta": result.auc_corrected - result.auc_frame,
        "n_flagged": float(result.n_flagged),
        "n_warranted": float(result.n_warranted),
        "n_negative": float(result.n_negative),
    }
    return promotion.journal(
        JOURNAL_COMPONENT,
        verdict,
        dataset_digest=frame.digest,
        metrics=metrics,
        version=result.version,
        state_dir=evalset.eval_root(root).parent,
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
