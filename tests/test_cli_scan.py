from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from cc_pushback.cli import main
from tests.builders import (
    assistant_tool_use,
    denial_result,
    user_text,
    write_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path

PUSHBACK = [
    user_text("don't add a fallback, crash instead"),
    assistant_tool_use("e1", "Edit", {"file_path": "/a.py"}),
    denial_result("e1", "rename it to feedback_db"),
]


def scan(runner: CliRunner, projects: Path, db: Path, *extra: str) -> str:
    result = runner.invoke(
        main,
        ["scan", "--transcripts", str(projects), "--no-github", "--db", str(db), *extra],
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_scan_is_idempotent_across_runs(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    write_transcript(projects / "a.jsonl", PUSHBACK)
    db = tmp_path / "feedback.db"
    runner = CliRunner()

    first = scan(runner, projects, db)
    second = scan(runner, projects, db)

    assert "transcripts=2" in first
    assert second.strip() == "no new rows"


def test_scan_picks_up_appended_entries(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    transcript = write_transcript(projects / "a.jsonl", PUSHBACK)
    db = tmp_path / "feedback.db"
    runner = CliRunner()

    scan(runner, projects, db)
    write_transcript(transcript, PUSHBACK + [user_text("also stop using globals")])

    assert "transcripts=1" in scan(runner, projects, db)


def test_source_filter_restricts_to_one_kind(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    write_transcript(projects / "a.jsonl", PUSHBACK)
    db = tmp_path / "feedback.db"
    runner = CliRunner()

    scan(runner, projects, db, "--source", "transcript_message")

    stats = runner.invoke(main, ["stats", "--db", str(db)])
    assert "transcript_message: 1" in stats.output
    assert "interrupt_rejection" not in stats.output


def test_stats_and_list_report_ingested_rows(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    write_transcript(projects / "a.jsonl", PUSHBACK)
    db = tmp_path / "feedback.db"
    runner = CliRunner()
    scan(runner, projects, db)

    stats = runner.invoke(main, ["stats", "--db", str(db)])
    assert "total: 2" in stats.output
    assert "files: 1" in stats.output

    listed = runner.invoke(main, ["list", "--db", str(db)])
    assert "don't add a fallback, crash instead" in listed.output
    assert "[transcript_message]" in listed.output
