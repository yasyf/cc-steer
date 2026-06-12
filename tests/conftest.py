from __future__ import annotations

from typing import TYPE_CHECKING

import cc_transcript.discovery
import pytest

from cc_pushback.store import FeedbackStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "claude-projects"
    root.mkdir()
    monkeypatch.setattr(cc_transcript.discovery, "CLAUDE_PROJECTS_DIR", root)
    monkeypatch.setenv("HOME", str(tmp_path))  # keep the shared ~/.cc-transcript corrections ledger hermetic
    return root


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[FeedbackStore]:
    async with await FeedbackStore.open(tmp_path / "feedback.db") as opened:
        yield opened
