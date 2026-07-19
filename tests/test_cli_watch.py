from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from click.testing import CliRunner

import cc_steer.watcher.daemon
import cc_steer.watcher.drafter_mlx
from cc_steer.cli import main

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


class StopWatcher:
    """Captures the Watcher kwargs and returns immediately from run() so the CLI tail can complete."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, cascade: object, delivery: object, **kwargs: Any) -> None:
        StopWatcher.last_kwargs = kwargs
        self.roots = kwargs["roots"]

    async def run(self) -> None:
        return


async def _sleep_forever(interval_s: float = 30.0) -> None:
    await anyio.sleep_forever()


class FakeMlxDrafter:
    """A stand-in for the real drafter that records idle_ttl_s and exposes a cancellable reaper."""

    last_idle_ttl_s: float | None = None

    def __init__(self, *, threshold: float | None = None, idle_ttl_s: float) -> None:
        FakeMlxDrafter.last_idle_ttl_s = idle_ttl_s
        self.base_model = "fake-base"
        self.threshold = 0.1234
        self.operating_point = "budget"
        self.render_version = 1
        self.version = types.SimpleNamespace(version="v001")
        self.resource = types.SimpleNamespace(run=_sleep_forever)

    async def draft(self, prompt: object) -> None:  # pragma: no cover - never reached, watcher is stubbed
        raise AssertionError("draft must not run under the stubbed watcher")


def test_poll_reaches_the_watcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.registry.current", lambda _component, **_kwargs: None)
    monkeypatch.setattr("cc_steer.cli.claude_available", lambda: True)
    monkeypatch.setattr(cc_steer.watcher.daemon, "Watcher", StopWatcher)
    result = CliRunner().invoke(
        main, ["watch", "--shadow", "--poll", "12.5", "--root", str(tmp_path), "--db", str(tmp_path / "f.db")]
    )
    assert result.exit_code == 0, result.output
    assert StopWatcher.last_kwargs["poll"] == 12.5


def test_stage2_idle_ttl_reaches_the_drafter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.registry.current", lambda _component, **_kwargs: None)
    monkeypatch.setattr(cc_steer.watcher.drafter_mlx, "MlxDrafter", FakeMlxDrafter)
    monkeypatch.setattr(cc_steer.watcher.daemon, "Watcher", StopWatcher)
    result = CliRunner().invoke(
        main,
        ["watch", "--shadow", "--drafter", "mlx", "--stage2-idle-ttl", "42.0", "--root", str(tmp_path),
         "--db", str(tmp_path / "f.db")],
    )
    assert result.exit_code == 0, result.output
    assert FakeMlxDrafter.last_idle_ttl_s == 42.0


def test_watch_help_lists_the_new_options() -> None:
    out = CliRunner().invoke(main, ["watch", "--help"]).output
    assert "--poll" in out
    assert "--stage2-idle-ttl" in out
