from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from cc_steer.journal import Journal

if TYPE_CHECKING:
    from pathlib import Path


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


@pytest.mark.parametrize(
    ("listed", "added", "expected"),
    [
        pytest.param("null", "null", False, id="both-null"),
        pytest.param("[null]", '{"id": "x"}', True, id="list-of-null-then-add"),
        pytest.param('{"unexpected": 1}', "[]", False, id="listed-dict-not-list"),
        pytest.param('[{"title": "cc-steer pipeline runs"}]', '{"id": "y"}', True, id="match-without-id-then-add"),
        pytest.param("[]", "[1, 2, 3]", False, id="added-non-dict"),
    ],
)
def test_append_never_raises_on_unexpected_json_shapes(
    listed: str, added: str, expected: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    responses = {("log", "list"): listed, ("log", "add"): added, ("log", "append"): ""}
    monkeypatch.setattr(subprocess, "run", fake_cc_notes(responses, calls))
    assert Journal(tmp_path).append("entry") is expected
