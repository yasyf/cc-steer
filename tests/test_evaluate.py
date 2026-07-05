from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from cc_transcript.mining import DedupKey

from cc_steer.evaluate import (
    AuditEstimate,
    GoldenRow,
    estimate,
    evaluate,
    exact_upper_bound,
    flip_report,
    golden_result,
    load_golden,
)
from cc_steer.triage import AUDIT_VERSION, AUDITOR, JUDGE, PROMPT_VERSION
from tests.test_triage import seed, verdict

if TYPE_CHECKING:
    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio


@pytest.mark.unit
def test_exact_upper_bound_matches_rule_of_three() -> None:
    assert exact_upper_bound(0, 60) == pytest.approx(1 - 0.05 ** (1 / 60), abs=1e-6)
    assert exact_upper_bound(0, 3) == pytest.approx(1 - 0.05 ** (1 / 3), abs=1e-6)
    assert exact_upper_bound(5, 5) == 1.0
    assert exact_upper_bound(0, 0) == 1.0


@pytest.mark.unit
def test_estimate_counts_only_audited_rows() -> None:
    rows = [{"dedup_key": "a"}, {"dedup_key": "b"}, {"dedup_key": "c"}]
    audits = {"a": {"accepted": 1}, "b": {"accepted": 0}}
    assert estimate(rows, audits) == AuditEstimate(audited=2, hits=1)
    assert AuditEstimate(audited=0, hits=0).rate is None


@pytest.mark.unit
def test_golden_result_passes_fails_and_flags_unjudged() -> None:
    golden = (
        GoldenRow(dedup_key="k1", source_kind="transcript_message", text="t1", expected=True, note=""),
        GoldenRow(dedup_key="k2", source_kind="transcript_message", text="t2", expected=False, note=""),
        GoldenRow(dedup_key="k3", source_kind="transcript_message", text="t3", expected=True, note=""),
    )
    judge = {
        "k1": {"accepted": 1, "category": "wrong_approach", "rationale": "clear rejection"},
        "k2": {"accepted": 1, "category": "premature", "rationale": "stopped early"},
    }
    result = golden_result(golden, {"k1", "k2", "k3"}, judge, "sha")
    assert (result.total, result.passed) == (3, 1)
    assert [(f.dedup_key, f.category) for f in result.failures] == [("k2", "premature"), ("k3", None)]
    assert [(f.dedup_key, f.rationale) for f in result.failures] == [("k2", "stopped early"), ("k3", None)]


@pytest.mark.unit
def test_golden_result_hard_fails_on_corpus_drift() -> None:
    golden = (GoldenRow(dedup_key="gone", source_kind="transcript_message", text="t", expected=False, note=""),)
    with pytest.raises(LookupError, match="gone"):
        golden_result(golden, {"other"}, {}, "sha")


@pytest.mark.unit
def test_load_golden_maps_labels_to_bool(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    rows = [
        {"dedup_key": "k", "source_kind": "plan_review", "text": "no", "expected": "steering", "note": "terse"},
        {"dedup_key": "k2", "source_kind": "plan_review", "text": "ok", "expected": "noise", "note": "approval"},
    ]
    path.write_text(json.dumps(rows))
    assert load_golden(path) == (
        GoldenRow(dedup_key="k", source_kind="plan_review", text="no", expected=True, note="terse"),
        GoldenRow(dedup_key="k2", source_kind="plan_review", text="ok", expected=False, note="approval"),
    )


@pytest.mark.integration
async def test_evaluate_end_to_end(store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(store)
    rows = await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION)
    keys = [DedupKey(str(row["dedup_key"])) for row in rows]
    accepted, rejected = keys[0], keys[1]
    await store.record_verdict(
        accepted, verdict(), role=JUDGE, prompt_version=PROMPT_VERSION, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        rejected, verdict("status_update"), role=JUDGE, prompt_version=PROMPT_VERSION, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        accepted, verdict(), role=AUDITOR, prompt_version=AUDIT_VERSION, model="opus", fidelity="full"
    )
    await store.record_verdict(
        rejected, verdict("wrong_approach"), role=AUDITOR, prompt_version=AUDIT_VERSION, model="opus", fidelity="full"
    )
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            [
                {
                    "dedup_key": accepted,
                    "source_kind": "transcript_message",
                    "text": "t",
                    "expected": "steering",
                    "note": "",
                },
                {
                    "dedup_key": rejected,
                    "source_kind": "transcript_message",
                    "text": "t",
                    "expected": "steering",
                    "note": "",
                },
            ]
        )
    )
    metrics = await evaluate(store, seed=1, accepts=10, rejects=10, golden_path=golden_path)
    assert (metrics.total, metrics.judged, metrics.accepted) == (2, 2, 1)
    assert (metrics.golden.passed, metrics.golden.total) == (1, 2)
    assert metrics.precision == 1.0  # the auditor agrees with the accept
    assert metrics.contamination == 1.0  # the auditor found steering in the reject
    assert len(metrics.disagreements) == 1
    assert metrics.disagreements[0].judge_category == "status_update"
    assert metrics.disagreements[0].judge_rationale == "r"
    assert metrics.disagreements[0].auditor_rationale == "r"


@pytest.mark.integration
async def test_flip_report_counts_only_side_changes(store: FeedbackStore) -> None:
    await seed(store)
    rows = await store.unjudged(role=JUDGE, prompt_version=1)
    keys = [DedupKey(str(row["dedup_key"])) for row in rows]
    flipper, stayer = keys[0], keys[1]
    await store.record_verdict(
        flipper, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        stayer, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        flipper, verdict("new_task"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    await store.record_verdict(
        stayer, verdict("incorrect_change"), role=JUDGE, prompt_version=2, model="sonnet", fidelity="full"
    )
    report = await flip_report(store, from_version=1, to_version=2)
    assert report.common == 2
    assert [(flip.from_category, flip.to_category) for flip in report.flips] == [("wrong_approach", "new_task")]
    assert report.rate == 0.5
