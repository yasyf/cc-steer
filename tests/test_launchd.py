from __future__ import annotations

import plistlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from cc_steer import launchd

if TYPE_CHECKING:
    import pytest


def test_render_produces_loadable_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    data = plistlib.loads(launchd.render("uvx cc-steer", Path("/repo/cc steer"), hour=4))
    assert data["Label"] == launchd.LABEL
    assert data["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}
    assert data["ProgramArguments"][:2] == ["/bin/sh", "-lc"]
    command = data["ProgramArguments"][2]
    assert "exec /opt/homebrew/bin/uvx cc-steer pipeline run --auto-weekly" in command
    assert "--journal-repo '/repo/cc steer'" in command
    assert 'eval "$(ccp env)"' in command
    assert data["StandardOutPath"].endswith("pipeline.log")


def test_pipeline_command_leaves_custom_prefix_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    command = launchd.pipeline_command("uv run --project /repo cc-steer", None)
    assert "exec uv run --project /repo cc-steer pipeline run --auto-weekly" in command
    assert "--journal-repo" not in command


def test_pipeline_command_survives_missing_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert "exec uvx cc-steer pipeline run" in launchd.pipeline_command("uvx cc-steer", None)


def test_render_retrain_produces_weekly_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uv")
    data = plistlib.loads(launchd.render_retrain(Path("/repo/cc-steer-lab"), hour=4))
    assert data["Label"] == launchd.RETRAIN_LABEL
    assert data["StartCalendarInterval"] == {"Weekday": launchd.SUNDAY, "Hour": 4, "Minute": 0}
    assert data["ProgramArguments"][:2] == ["/bin/sh", "-lc"]
    command = data["ProgramArguments"][2]
    assert "exec /opt/homebrew/bin/uv run --project /repo/cc-steer-lab python -m harness.retrain" in command
    assert 'eval "$(ccp env)"' in command
    assert data["StandardOutPath"].endswith("retrain.log")
    assert data["StandardErrorPath"].endswith("retrain.log")


def test_retrain_command_quotes_lab_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uv")
    assert "--project '/repo/cc steer lab'" in launchd.retrain_command(Path("/repo/cc steer lab"))


def test_retrain_command_survives_missing_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert "exec uv run --project /lab python -m harness.retrain" in launchd.retrain_command(Path("/lab"))


def test_render_watch_is_a_keepalive_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    data = plistlib.loads(launchd.render_watch("uv run --project /repo cc-steer"))
    assert data["Label"] == launchd.WATCH_LABEL
    assert data["KeepAlive"] is True
    assert data["RunAtLoad"] is True
    assert "StartCalendarInterval" not in data
    assert data["ProgramArguments"][:2] == ["/bin/sh", "-lc"]
    command = data["ProgramArguments"][2]
    assert "exec uv run --project /repo cc-steer watch --gate lexical --gate-threshold 0.5 --drafter mlx" in command
    assert 'eval "$(ccp env)"' in command
    assert data["StandardOutPath"].endswith("watch.log")
    assert data["StandardErrorPath"].endswith("watch.log")


def test_watch_command_resolves_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    assert "exec /opt/homebrew/bin/uvx cc-steer watch --gate lexical" in launchd.watch_command("uvx cc-steer")


def test_watch_command_survives_missing_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert "exec uvx cc-steer watch --gate lexical --gate-threshold 0.5 --drafter mlx" in launchd.watch_command(
        "uvx cc-steer"
    )
