from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

from cc_steer.journal import Journal

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def fake_cc_notes(responses: dict[tuple[str, ...], str], calls: list[list[str]]) -> Any:
    def run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        key = tuple(argv[1:3])
        if key not in responses:
            return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="unknown")
        return subprocess.CompletedProcess(argv, returncode=0, stdout=responses[key], stderr="")

    return run


def test_append_finds_existing_log_by_title(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    listed = json.dumps([{"id": "aaa", "title": "other"}, {"id": "bbb", "title": "cc-steer pipeline runs"}])
    monkeypatch.setattr(
        subprocess, "run", fake_cc_notes({("log", "list"): listed, ("log", "append"): ""}, calls)
    )
    journal = Journal(tmp_path)
    assert journal.append("first") is True
    assert journal.append("second") is True
    appends = [argv for argv in calls if argv[1:3] == ["log", "append"]]
    assert [argv[3:] for argv in appends] == [["bbb", "--entry", "first"], ["bbb", "--entry", "second"]]
    assert sum(argv[1:3] == ["log", "list"] for argv in calls) == 1


def test_append_creates_log_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    responses = {("log", "list"): "[]", ("log", "add"): json.dumps({"id": "fresh"}), ("log", "append"): ""}
    monkeypatch.setattr(subprocess, "run", fake_cc_notes(responses, calls))
    assert Journal(tmp_path).append("entry") is True
    add = next(argv for argv in calls if argv[1:3] == ["log", "add"])
    assert add[3] == "cc-steer pipeline runs"
    assert add[add.index("--label") + 1] == "pipeline"
    append = next(argv for argv in calls if argv[1:3] == ["log", "append"])
    assert append[3:] == ["fresh", "--entry", "entry"]


def test_append_degrades_when_binary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("cc-notes")

    monkeypatch.setattr(subprocess, "run", missing)
    assert Journal(tmp_path).append("entry") is False


def test_append_degrades_on_uninitialized_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_cc_notes({}, calls))
    assert Journal(tmp_path).append("entry") is False
