from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from cc_steer import pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stubs every stage function on the pipeline module, recording call order."""
    calls: list[str] = []

    def stub(name: str, result: Any) -> Any:
        async def run(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(name)
            return result

        return run

    monkeypatch.setattr(pipeline, "run_scan", stub("scan", SimpleNamespace(scanned=3, inserted=2)))
    monkeypatch.setattr(pipeline, "run_triage", stub("triage", SimpleNamespace(judged=5, failed=0, pending=1)))
    monkeypatch.setattr(
        pipeline, "run_refine", stub("refine", SimpleNamespace(refined=4, pairs=6, failed=0, pending=0))
    )
    monkeypatch.setattr(
        pipeline, "run_enrich", stub("enrich", SimpleNamespace(enriched=2, corrections=1, skipped=0, pending=3))
    )
    monkeypatch.setattr(
        pipeline,
        "run_negatives",
        stub("negatives", SimpleNamespace(inserted={"positive_window": 4, "hard_negative": 1}, sessions_sampled=2)),
    )
    monkeypatch.setattr(pipeline, "run_audit", stub("audit", SimpleNamespace(judged=7, failed=0)))
    monkeypatch.setattr(
        pipeline,
        "evaluate",
        stub("eval", SimpleNamespace(golden=SimpleNamespace(passed=57, total=57), precision=0.9, contamination=0.1)),
    )
    monkeypatch.setattr(
        pipeline,
        "attribute_reactions",
        stub("reactions", SimpleNamespace(total=0, summary_line=lambda: "attributed 0 ")),
    )
    monkeypatch.setattr(
        pipeline, "run_export", stub("export", SimpleNamespace(counts={"sft": {"train": 9, "test": 1}}, pushed=True))
    )
    return calls


async def test_nightly_runs_core_stages_in_order(store: FeedbackStore, stages: list[str], tmp_path: Path) -> None:
    report = await pipeline.run_pipeline(store, out=tmp_path, push_to="user/repo")
    assert stages == ["scan", "triage", "refine", "enrich", "negatives", "reactions", "export"]
    assert [outcome.stage for outcome in report.outcomes] == stages
    assert report.failed == ()
    assert "export: sft 10 pushed to user/repo" in report.summary_line()


async def test_weekly_adds_audit_and_eval(store: FeedbackStore, stages: list[str], tmp_path: Path) -> None:
    report = await pipeline.run_pipeline(store, out=tmp_path, push_to=None, weekly=True, audit_seed=7)
    assert stages == ["scan", "triage", "refine", "enrich", "negatives", "audit", "eval", "reactions", "export"]
    eval_outcome = next(outcome for outcome in report.outcomes if outcome.stage == "eval")
    assert eval_outcome.summary == "golden 57/57, precision 0.900, contamination 0.100"


async def test_stage_failure_is_isolated(
    store: FeedbackStore, stages: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("account exhausted")

    monkeypatch.setattr(pipeline, "run_triage", boom)
    report = await pipeline.run_pipeline(store, out=tmp_path, push_to=None)
    assert report.failed == ("triage",)
    assert stages == ["scan", "refine", "enrich", "negatives", "reactions", "export"]
    triage_outcome = next(outcome for outcome in report.outcomes if outcome.stage == "triage")
    assert not triage_outcome.ok
    assert "RuntimeError" in triage_outcome.summary
    assert "triage: FAILED — RuntimeError" in report.summary_line()


async def test_failed_push_downgrades_to_local_export(
    store: FeedbackStore, stages: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str | None] = []

    async def run_export(*_args: Any, push_to: str | None = None, **_kwargs: Any) -> Any:
        attempts.append(push_to)
        if push_to is not None:
            raise ConnectionError("401")
        return SimpleNamespace(counts={"sft": {"train": 9}}, pushed=False)

    monkeypatch.setattr(pipeline, "run_export", run_export)
    report = await pipeline.run_pipeline(store, out=tmp_path, push_to="user/repo")
    assert attempts == ["user/repo", None]
    export_outcome = next(outcome for outcome in report.outcomes if outcome.stage == "export")
    assert export_outcome.ok
    assert "push to user/repo FAILED: ConnectionError" in export_outcome.summary
    assert report.failed == ()


def test_weekly_seed_is_deterministic_per_iso_week() -> None:
    assert pipeline.weekly_seed(date(2026, 7, 7)) == pipeline.weekly_seed(date(2026, 7, 12))
    assert pipeline.weekly_seed(date(2026, 7, 7)) != pipeline.weekly_seed(date(2026, 7, 13))
    assert pipeline.weekly_seed(date(2026, 7, 7)) == 202628


def test_scan_roots_includes_mirrors_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "MIRRORS_DIR", tmp_path / "missing")
    assert pipeline.scan_roots() == (pipeline.CLAUDE_PROJECTS_DIR,)
    (tmp_path / "missing").mkdir()
    assert pipeline.scan_roots() == (pipeline.CLAUDE_PROJECTS_DIR, tmp_path / "missing")
