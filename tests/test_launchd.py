from __future__ import annotations

import plistlib
import shutil
import subprocess
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
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    data = plistlib.loads(launchd.render_retrain("uvx cc-steer", hour=4))
    assert data["Label"] == launchd.RETRAIN_LABEL
    assert data["StartCalendarInterval"] == {"Weekday": launchd.SUNDAY, "Hour": 4, "Minute": 0}
    assert data["ProgramArguments"][:2] == ["/bin/sh", "-lc"]
    command = data["ProgramArguments"][2]
    assert "/opt/homebrew/bin/uvx cc-steer retrain --component gate" in command
    assert 'eval "$(ccp env)"' in command
    assert data["StandardOutPath"].endswith("retrain.log")
    assert data["StandardErrorPath"].endswith("retrain.log")


def test_retrain_command_runs_both_lanes_and_aggregates_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    command = launchd.retrain_command("uvx cc-steer")
    assert command == (
        'command -v ccp >/dev/null 2>&1 && eval "$(ccp env)"; '
        "s=0; /opt/homebrew/bin/uvx cc-steer retrain --component gate || s=1; "
        "/opt/homebrew/bin/uvx cc-steer retrain --component watcher || s=1; exit $s"
    )


def test_retrain_command_exit_status_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    # A gate-lane failure must not skip the watcher lane, yet must surface in the exit status.
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    lanes = launchd.retrain_command("uvx cc-steer").split("; ", 1)[1]
    script = lanes.replace("uvx cc-steer retrain --component gate", "false").replace(
        "uvx cc-steer retrain --component watcher", "echo watcher-ran"
    )
    proc = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, check=False)
    assert proc.stdout == "watcher-ran\n"
    assert proc.returncode == 1


def test_retrain_command_leaves_custom_prefix_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    command = launchd.retrain_command("uv run --project /repo cc-steer")
    assert "uv run --project /repo cc-steer retrain --component gate || s=1" in command
    assert "uv run --project /repo cc-steer retrain --component watcher || s=1" in command


def test_retrain_command_survives_missing_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    command = launchd.retrain_command("uvx cc-steer")
    assert "s=0; uvx cc-steer retrain --component gate || s=1" in command
    assert "uvx cc-steer retrain --component watcher || s=1; exit $s" in command


def test_retrain_prefix_rewrites_only_the_default() -> None:
    assert launchd.retrain_prefix("uvx cc-steer") == launchd.RETRAIN_EXTRA_PREFIX
    assert launchd.retrain_prefix("uv run --project /repo cc-steer") == "uv run --project /repo cc-steer"


def test_install_retrain_default_prefix_resolves_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    captured: dict[str, bytes] = {}
    monkeypatch.setattr(launchd, "_install", lambda path, plist: captured.update(plist=plist) or path)
    launchd.install_retrain("uvx cc-steer")
    command = plistlib.loads(captured["plist"])["ProgramArguments"][2]
    assert "--from 'cc-steer[retrain]'" in command
    assert "cc-steer retrain --component gate" in command
    assert "cc-steer retrain --component watcher" in command


def test_install_retrain_custom_prefix_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    captured: dict[str, bytes] = {}
    monkeypatch.setattr(launchd, "_install", lambda path, plist: captured.update(plist=plist) or path)
    launchd.install_retrain("uv run --project /repo cc-steer")
    command = plistlib.loads(captured["plist"])["ProgramArguments"][2]
    assert "--from 'cc-steer[retrain]'" not in command
    assert "uv run --project /repo cc-steer retrain --component gate" in command


def test_kickstart_watch_true_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(launchd.subprocess, "run", fake_run)
    monkeypatch.setattr(launchd, "_uid", lambda: "501")
    assert launchd.kickstart_watch() is True
    assert calls == [["launchctl", "kickstart", "-k", "gui/501/com.cc-steer.watch"]]


def test_kickstart_watch_false_when_agent_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launchd.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, 3)
    )
    monkeypatch.setattr(launchd, "_uid", lambda: "501")
    assert launchd.kickstart_watch() is False


def test_no_lab_dir_symbol() -> None:
    assert not hasattr(launchd, "LAB_DIR")


def test_render_watch_is_a_keepalive_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    data = plistlib.loads(launchd.render_watch("uv run --project /repo cc-steer"))
    assert data["Label"] == launchd.WATCH_LABEL
    assert data["KeepAlive"] is True
    assert data["RunAtLoad"] is True
    assert "StartCalendarInterval" not in data
    assert data["ProgramArguments"][:2] == ["/bin/sh", "-lc"]
    command = data["ProgramArguments"][2]
    assert "exec uv run --project /repo cc-steer watch --gate lexical --drafter mlx" in command
    assert "--gate-threshold" not in command  # serving honors each promoted gate's fitted threshold
    assert 'eval "$(ccp env)"' in command
    assert data["StandardOutPath"].endswith("watch.log")
    assert data["StandardErrorPath"].endswith("watch.log")


def test_watch_command_resolves_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    assert "exec /opt/homebrew/bin/uvx cc-steer watch --gate lexical" in launchd.watch_command("uvx cc-steer")


def test_watch_command_survives_missing_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert "exec uvx cc-steer watch --gate lexical --drafter mlx" in launchd.watch_command("uvx cc-steer")


def test_watch_prefix_rewrites_only_the_default() -> None:
    assert launchd.watch_prefix("uvx cc-steer") == launchd.WATCH_EXTRA_PREFIX
    assert launchd.watch_prefix("uv run --project /repo cc-steer") == "uv run --project /repo cc-steer"


def test_install_watch_default_prefix_resolves_the_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/uvx")
    captured: dict[str, bytes] = {}
    monkeypatch.setattr(launchd, "_install", lambda path, plist: captured.update(plist=plist) or path)
    launchd.install_watch("uvx cc-steer")
    command = plistlib.loads(captured["plist"])["ProgramArguments"][2]
    assert "--from 'cc-steer[gate,mlx]'" in command
    assert "cc-steer watch --gate lexical --drafter mlx" in command
