from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cc_pushback.sources.issues import SupersetIssues, changed_issue_files

if TYPE_CHECKING:
    from pathlib import Path

    from cc_pushback.repo import Repository


def write_issues(path: Path, objects: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o) + "\n" for o in objects))
    return path


def test_text_joins_rule_and_suggested_fix(tmp_path: Path) -> None:
    path = write_issues(
        tmp_path / ".context" / "cleanup" / "issues.jsonl",
        [{"rule": "no fallbacks", "suggested_fix": "crash instead", "severity": "major"}],
    )

    cands = list(SupersetIssues().candidates_for_file(path, 1.0))

    assert len(cands) == 1
    assert cands[0].text == "no fallbacks\ncrash instead"
    assert cands[0].payload == {"rule": "no fallbacks", "suggested_fix": "crash instead", "severity": "major"}


def test_text_falls_back_to_rule_when_no_fix(tmp_path: Path) -> None:
    path = write_issues(tmp_path / ".context" / "cleanup" / "issues.jsonl", [{"rule": "ask first"}])

    cands = list(SupersetIssues().candidates_for_file(path, 1.0))

    assert cands[0].text == "ask first"


def test_text_falls_back_to_raw_line_without_rule(tmp_path: Path) -> None:
    path = write_issues(tmp_path / ".context" / "cleanup" / "issues.jsonl", [{"evidence": "some note"}])

    cands = list(SupersetIssues().candidates_for_file(path, 1.0))

    assert cands[0].text == json.dumps({"evidence": "some note"})


def test_changed_issue_files_filters_known_mtimes(feedback_db: Repository, tmp_path: Path) -> None:
    path = write_issues(tmp_path / ".context" / "cleanup" / "issues.jsonl", [{"rule": "x"}])
    mtime = path.stat().st_mtime
    feedback_db.record_file_scan(str(path), mtime, [])

    assert list(changed_issue_files(feedback_db, [tmp_path])) == []


def test_changed_issue_files_finds_new(feedback_db: Repository, tmp_path: Path) -> None:
    path = write_issues(tmp_path / ".context" / "cleanup" / "issues.jsonl", [{"rule": "x"}])

    found = list(changed_issue_files(feedback_db, [tmp_path]))

    assert [p for p, _ in found] == [path]
