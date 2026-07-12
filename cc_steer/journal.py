"""A thin cc-notes journal: unattended passes append one line each to a shared log.

cc-notes is git-native and per-repository, so journaling requires a repo that has
run ``cc-notes init``. Every failure mode — missing binary, uninitialized repo,
malformed output — degrades to ``False`` so an unattended pipeline never fails on
its own bookkeeping.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

PIPELINE_LOG_TITLE = "cc-steer pipeline runs"
PIPELINE_LOG_LABEL = "pipeline"


class Journal:
    """Appends run summaries to one cc-notes log, found or created by title."""

    def __init__(self, repo: Path, *, title: str = PIPELINE_LOG_TITLE, label: str = PIPELINE_LOG_LABEL) -> None:
        self.repo = repo
        self.title = title
        self.label = label
        self._log_id: str | None = None

    def append(self, text: str) -> bool:
        """Appends one entry, creating the log on first use. True when recorded."""
        if (log_id := self._resolve()) is None:
            return False
        return self._run("log", "append", log_id, "--entry", text) is not None

    def _resolve(self) -> str | None:
        if self._log_id is not None:
            return self._log_id
        if (listed := self._run("log", "list", "--json", "--label", self.label)) is not None:
            try:
                logs = json.loads(listed or "[]")
            except json.JSONDecodeError:
                logs = []
            for log in logs:
                if log.get("title") == self.title:
                    self._log_id = str(log["id"])
                    return self._log_id
        if (added := self._run("log", "add", self.title, "--label", self.label, "--json")) is not None:
            try:
                self._log_id = str(json.loads(added)["id"])
            except (json.JSONDecodeError, KeyError):
                return None
        return self._log_id

    def _run(self, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["cc-notes", *args], cwd=self.repo, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None
