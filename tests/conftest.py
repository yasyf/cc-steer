from __future__ import annotations

from typing import TYPE_CHECKING

import cc_transcript.corrections
import cc_transcript.discovery
import pytest
from spawnllm.backends.base import BackendUnavailable

from cc_steer.store import FeedbackStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call() -> Iterator[None]:
    try:
        yield
    except BackendUnavailable as exc:
        pytest.skip(f"no LLM backend: {exc}")


@pytest.fixture(autouse=True)
def projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "claude-projects"
    root.mkdir()
    monkeypatch.setattr(cc_transcript.discovery, "CLAUDE_PROJECTS_DIR", root)
    # Redirect the shared corrections ledger into the tmp dir so it stays hermetic,
    # without overriding HOME (which would hide the LLM backend's CLI auth).
    ledger = tmp_path / "corrections.db"
    real_open = cc_transcript.corrections.CorrectionLog.open
    monkeypatch.setattr(
        cc_transcript.corrections.CorrectionLog,
        "open",
        classmethod(lambda cls, path=None: real_open.__func__(cls, path or ledger)),
    )
    return root


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[FeedbackStore]:
    async with await FeedbackStore.open(tmp_path / "feedback.db") as opened:
        yield opened
