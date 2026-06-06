from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from cc_pushback.repo import Repository

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture
def feedback_db(tmp_path: Path) -> Iterator[Repository]:
    with Repository.open(tmp_path / "feedback.db") as repo:
        yield repo


@pytest.fixture
def fake_gh(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], None]:
    canned: dict[str, Any] = {
        "remote": "git@github.com:owner/repo.git",
        "login": "yasyf",
        "pulls": [],
        "comments": {},
    }

    def configure(updates: dict[str, Any]) -> None:
        canned.update(updates)

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        match args:
            case ["git", "-C", _, "remote", "get-url", "origin"]:
                remote = canned["remote"]
                if remote is None:
                    return subprocess.CompletedProcess(args, 128, stdout="", stderr="no origin")
                return subprocess.CompletedProcess(args, 0, stdout=remote + "\n", stderr="")
            case ["gh", "api", "user", "--jq", ".login"]:
                return subprocess.CompletedProcess(args, 0, stdout=canned["login"] + "\n", stderr="")
            case ["gh", "api", path] if "/pulls?" in path:
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(canned["pulls"]), stderr="")
            case ["gh", "api", path] if "/comments?" in path:
                number = int(path.split("/pulls/")[1].split("/")[0])
                return subprocess.CompletedProcess(
                    args, 0, stdout=json.dumps(canned["comments"].get(number, [])), stderr=""
                )
            case _:
                raise AssertionError(f"unexpected argv: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return configure
