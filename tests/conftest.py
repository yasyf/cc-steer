from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

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


@pytest.fixture(autouse=True)
def hermetic_cc_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the cc-notes boundary so the retrain journal mirror never touches the real ledger.

    Only ``cc-notes`` invocations are intercepted (degraded to a failing exit); every other
    subprocess passes through. A test that asserts the mirror re-patches ``subprocess.run``.
    """
    real_run = subprocess.run

    def guarded(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "cc-notes":
            return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="hermetic")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)


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
    # Keep the shadow ledger's default path hermetic so a pass that resolves it
    # (export's reactions read, the attribution scan) never touches the real db.
    monkeypatch.setenv("CC_STEER_SHADOW_DB", str(tmp_path / "shadow.db"))
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
