from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, NoReturn

import cc_transcript.judge
import pytest
from click.testing import CliRunner

import cc_pushback.cli
import cc_pushback.enrich
import cc_pushback.export
import cc_pushback.refine
from cc_pushback.cli import main
from cc_pushback.enrich import EnrichReport
from cc_pushback.export import ExportReport
from cc_pushback.refine import RefineReport
from cc_pushback.triage import TriageReport
from tests.builders import assistant_tool_use, denial_result, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from cc_pushback.store import FeedbackStore

HF_REPO_ID = "yasyf/cc-pushback-traces"

pytestmark = pytest.mark.integration


class ExportCall(NamedTuple):
    out: Path
    push_to: str | None


@dataclass(frozen=True, slots=True)
class StageCase:
    command: str
    module: ModuleType
    attribute: str
    changed: TriageReport | RefineReport | EnrichReport
    unchanged: TriageReport | RefineReport | EnrichReport


STAGES = (
    StageCase(
        "triage",
        cc_pushback.cli,
        "run_triage",
        changed=TriageReport(judged=3, failed=1, pending=2),
        unchanged=TriageReport(judged=0, failed=1, pending=2),
    ),
    StageCase(
        "audit",
        cc_pushback.cli,
        "run_audit",
        changed=TriageReport(judged=2, failed=0, pending=0),
        unchanged=TriageReport(judged=0, failed=2, pending=0),
    ),
    StageCase(
        "refine",
        cc_pushback.refine,
        "refine",
        changed=RefineReport(refined=2, pairs=5, failed=0, pending=1),
        unchanged=RefineReport(refined=0, pairs=0, failed=1, pending=3),
    ),
    StageCase(
        "enrich",
        cc_pushback.enrich,
        "enrich",
        changed=EnrichReport(corrections=2, skipped=1, pending=0),
        unchanged=EnrichReport(corrections=0, skipped=3, pending=4),
    ),
)


@pytest.fixture(autouse=True)
def export_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[ExportCall]:
    calls: list[ExportCall] = []

    async def record(store: FeedbackStore, *, out: Path, push_to: str | None = None) -> ExportReport:
        calls.append(ExportCall(out, push_to))
        return ExportReport(counts={"traces": {"train": 2, "test": 1}}, out=out, pushed=push_to is not None)

    monkeypatch.setattr(cc_pushback.export, "export", record)
    monkeypatch.setattr(cc_pushback.cli, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(cc_pushback.cli, "hf_repo_id", lambda: HF_REPO_ID)
    return calls


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "feedback.db"


@pytest.fixture
def transcripts(tmp_path: Path) -> Path:
    return write_transcript(
        tmp_path / "transcripts" / "proj" / "s.jsonl",
        [
            assistant_tool_use("t1", "Write", {"file_path": "/a.py", "content": "x = 1"}),
            denial_result("t1", said="don't do that"),
            user_text("use a frozen dataclass here instead of a dict"),
        ],
    ).parents[1]


def test_scan_syncs_after_inserting_rows(
    runner: CliRunner, transcripts: Path, db: Path, tmp_path: Path, export_calls: list[ExportCall]
) -> None:
    result = runner.invoke(main, ["scan", "--transcripts", str(transcripts), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert export_calls == [ExportCall(tmp_path, HF_REPO_ID)]
    assert f"syncing dataset to {HF_REPO_ID}" in result.output


def test_scan_skips_sync_when_nothing_inserted(
    runner: CliRunner, transcripts: Path, db: Path, export_calls: list[ExportCall]
) -> None:
    first = runner.invoke(main, ["scan", "--transcripts", str(transcripts), "--db", str(db)])
    assert first.exit_code == 0, first.output
    assert len(export_calls) == 1

    second = runner.invoke(main, ["scan", "--transcripts", str(transcripts), "--db", str(db)])
    assert second.exit_code == 0, second.output
    assert len(export_calls) == 1
    assert "syncing dataset to" not in second.output


def test_no_sync_suppresses_the_push(
    runner: CliRunner, transcripts: Path, db: Path, export_calls: list[ExportCall]
) -> None:
    result = runner.invoke(main, ["scan", "--transcripts", str(transcripts), "--db", str(db), "--no-sync"])
    assert result.exit_code == 0, result.output
    assert re.search(r"scanned 1 files, [1-9]\d* new rows", result.output)
    assert export_calls == []


@pytest.mark.parametrize("changed", [True, False], ids=["changed-pass-syncs", "no-op-pass-skips"])
@pytest.mark.parametrize("case", STAGES, ids=[case.command for case in STAGES])
def test_stage_sync_fires_only_when_the_pass_changed_data(
    case: StageCase,
    changed: bool,
    runner: CliRunner,
    db: Path,
    tmp_path: Path,
    export_calls: list[ExportCall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = case.changed if changed else case.unchanged

    async def stage(store: FeedbackStore, **kwargs: object) -> TriageReport | RefineReport | EnrichReport:
        return report

    monkeypatch.setattr(cc_pushback.cli, "claude_available", lambda: True)
    monkeypatch.setattr(cc_transcript.judge, "resolved_model", lambda tier: "stub-model")
    monkeypatch.setattr(case.module, case.attribute, stage)
    result = runner.invoke(main, [case.command, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert export_calls == ([ExportCall(tmp_path, HF_REPO_ID)] if changed else [])
    assert ("syncing dataset to" in result.output) is changed


def test_export_push_resolves_the_repo_from_the_hf_user(
    runner: CliRunner, db: Path, tmp_path: Path, export_calls: list[ExportCall]
) -> None:
    result = runner.invoke(main, ["export", "--db", str(db), "--out", str(tmp_path / "ds"), "--push"])
    assert result.exit_code == 0, result.output
    assert export_calls == [ExportCall(tmp_path / "ds", HF_REPO_ID)]
    assert f"pushed to {HF_REPO_ID}" in result.output


def test_export_without_push_never_resolves_the_hf_user(
    runner: CliRunner, db: Path, tmp_path: Path, export_calls: list[ExportCall], monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline() -> str:
        raise AssertionError("hf_repo_id must not be called without --push")

    monkeypatch.setattr(cc_pushback.cli, "hf_repo_id", offline)
    result = runner.invoke(main, ["export", "--db", str(db), "--out", str(tmp_path / "ds")])
    assert result.exit_code == 0, result.output
    assert export_calls == [ExportCall(tmp_path / "ds", None)]


def test_push_failure_exits_nonzero(
    runner: CliRunner, transcripts: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(store: FeedbackStore, *, out: Path, push_to: str | None = None) -> ExportReport:
        raise RuntimeError("hub push failed")

    monkeypatch.setattr(cc_pushback.export, "export", explode)
    result = runner.invoke(main, ["scan", "--transcripts", str(transcripts), "--db", str(db)])
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


def test_view_samples_refuses_an_empty_corpus(runner: CliRunner, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def never(*_: object, **__: object) -> NoReturn:
        raise AssertionError("build_summary must not run without samples")

    monkeypatch.setattr(cc_pushback.cli, "claude_available", lambda: True)
    monkeypatch.setattr(cc_pushback.cli, "build_summary", never)
    result = runner.invoke(main, ["view-samples", "--db", str(db)])
    assert result.exit_code != 0
    assert "no judged samples to serve" in result.output
