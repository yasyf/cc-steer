from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from click.testing import CliRunner, Result

import cc_steer.watcher.cascade
import cc_steer.watcher.daemon
import cc_steer.watcher.drafter_http
import cc_steer.watcher.drafter_mlx
from cc_steer.cli import main

if TYPE_CHECKING:
    from pathlib import Path

    from cc_steer.watcher.types import CascadeConfig

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


class FakeHttpDrafter:
    """A stand-in for the http drafter that records the config kwargs the CLI branch passed it."""

    last_kwargs: dict[str, Any] = {}
    closed: bool = False

    def __init__(
        self, *, endpoint: str, model: str, threshold: float, timeout: float, api_key: str | None = None
    ) -> None:
        FakeHttpDrafter.last_kwargs = {
            "endpoint": endpoint,
            "model": model,
            "threshold": threshold,
            "timeout": timeout,
            "api_key": api_key,
        }
        FakeHttpDrafter.closed = False

    async def draft(self, prompt: object) -> None:  # pragma: no cover - never reached, watcher is stubbed
        raise AssertionError("draft must not run under the stubbed watcher")

    async def aclose(self) -> None:
        FakeHttpDrafter.closed = True


class RecordingCascade:
    """Captures the CascadeConfig the CLI built, so tests can read what reached stage 2."""

    last_config: CascadeConfig | None = None

    def __init__(self, **kwargs: Any) -> None:
        RecordingCascade.last_config = kwargs["config"]


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
        [
            "watch",
            "--shadow",
            "--drafter",
            "mlx",
            "--stage2-idle-ttl",
            "42.0",
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "f.db"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert FakeMlxDrafter.last_idle_ttl_s == 42.0


def test_http_drafter_receives_endpoint_model_threshold_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cc_steer.registry.current", lambda _component, **_kwargs: None)
    monkeypatch.setattr(cc_steer.watcher.drafter_http, "HttpDrafter", FakeHttpDrafter)
    monkeypatch.setattr(cc_steer.watcher.daemon, "Watcher", StopWatcher)
    monkeypatch.setenv("CC_STEER_DRAFTER_API_KEY", "sk-secret")
    result = CliRunner().invoke(
        main,
        [
            "watch",
            "--shadow",
            "--drafter",
            "http",
            "--drafter-endpoint",
            "https://x.modal.run",
            "--drafter-model",
            "watcher-9b",
            "--stage2-threshold",
            "0.3",
            "--drafter-timeout",
            "12.5",
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "f.db"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert FakeHttpDrafter.last_kwargs == {
        "endpoint": "https://x.modal.run",
        "model": "watcher-9b",
        "threshold": 0.3,
        "timeout": 12.5,
        "api_key": "sk-secret",
    }
    assert FakeHttpDrafter.closed is True


def _invoke_http_watch(tmp_path: Path, *extra: str) -> Result:
    return CliRunner().invoke(
        main,
        [
            "watch",
            "--shadow",
            "--drafter",
            "http",
            "--drafter-endpoint",
            "https://x.modal.run",
            "--drafter-model",
            "watcher-9b",
            *extra,
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "f.db"),
        ],
    )


def test_http_drafter_renders_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.registry.current", lambda _component, **_kwargs: None)
    monkeypatch.setattr(cc_steer.watcher.drafter_http, "HttpDrafter", FakeHttpDrafter)
    monkeypatch.setattr(cc_steer.watcher.cascade, "Cascade", RecordingCascade)
    monkeypatch.setattr(cc_steer.watcher.daemon, "Watcher", StopWatcher)
    result = _invoke_http_watch(tmp_path)
    assert result.exit_code == 0, result.output
    assert RecordingCascade.last_config is not None
    assert RecordingCascade.last_config.render_version == 2


def test_http_drafter_requires_an_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.registry.current", lambda _component, **_kwargs: None)
    monkeypatch.setattr(cc_steer.watcher.drafter_http, "HttpDrafter", FakeHttpDrafter)
    monkeypatch.setattr(cc_steer.watcher.daemon, "Watcher", StopWatcher)
    result = CliRunner().invoke(
        main,
        [
            "watch",
            "--shadow",
            "--drafter",
            "http",
            "--drafter-model",
            "watcher-9b",
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "f.db"),
        ],
    )
    assert result.exit_code != 0
    assert "--drafter-endpoint is required" in result.output


def test_auto_selects_mlx_over_http_when_a_watcher_is_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cc_steer.registry.current", lambda component, **_kwargs: object() if component == "watcher" else None
    )
    monkeypatch.setattr("cc_steer.cli._mlx_importable", lambda: True)
    monkeypatch.setattr(cc_steer.watcher.drafter_mlx, "MlxDrafter", FakeMlxDrafter)
    monkeypatch.setattr(cc_steer.watcher.daemon, "Watcher", StopWatcher)
    result = CliRunner().invoke(
        main, ["watch", "--shadow", "--stage2-idle-ttl", "7.0", "--root", str(tmp_path), "--db", str(tmp_path / "f.db")]
    )
    assert result.exit_code == 0, result.output
    assert FakeMlxDrafter.last_idle_ttl_s == 7.0
    assert "drafter: mlx" in result.output


def test_watch_help_lists_the_new_options() -> None:
    out = CliRunner().invoke(main, ["watch", "--help"]).output
    assert "--poll" in out
    assert "--stage2-idle-ttl" in out
    assert "--drafter-endpoint" in out
    assert "--drafter-timeout" in out
