from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cc_steer import hooks

if TYPE_CHECKING:
    from pathlib import Path

FOREIGN = {"hooks": [{"type": "command", "command": "uvx capt-hook run SessionEnd --async", "async": True}]}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_install_into_missing_file_adds_group(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    assert hooks.install(settings) == "added"
    groups = read(settings)["hooks"]["SessionEnd"]
    assert groups == [hooks.hook_group()]
    assert groups[0]["hooks"][0]["command"] == "uvx cc-steer scan --no-sync"
    assert groups[0]["hooks"][0]["async"] is True


def test_install_is_idempotent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    assert hooks.install(settings) == "added"
    before = settings.read_text()
    assert hooks.install(settings) == "unchanged"
    assert settings.read_text() == before


def test_install_preserves_foreign_groups_and_other_keys(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus", "hooks": {"SessionEnd": [FOREIGN], "Stop": [FOREIGN]}}))
    assert hooks.install(settings) == "added"
    data = read(settings)
    assert data["model"] == "opus"
    assert data["hooks"]["Stop"] == [FOREIGN]
    assert data["hooks"]["SessionEnd"] == [FOREIGN, hooks.hook_group()]


def test_install_replaces_own_group_on_prefix_change(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    hooks.install(settings)
    assert hooks.install(settings, prefix="uv run --project /repo cc-steer") == "updated"
    groups = read(settings)["hooks"]["SessionEnd"]
    assert len(groups) == 1
    assert groups[0]["hooks"][0]["command"] == "uv run --project /repo cc-steer scan --no-sync"


def test_uninstall_removes_only_own_group(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"SessionEnd": [FOREIGN]}}))
    hooks.install(settings)
    assert hooks.uninstall(settings) == "removed"
    assert read(settings)["hooks"]["SessionEnd"] == [FOREIGN]
    assert hooks.uninstall(settings) == "absent"


def test_uninstall_drops_empty_event_key(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    hooks.install(settings)
    hooks.uninstall(settings)
    assert "SessionEnd" not in read(settings)["hooks"]


def test_uninstall_missing_file_is_absent(tmp_path: Path) -> None:
    assert hooks.uninstall(tmp_path / "settings.json") == "absent"


def test_installed_command_roundtrip(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    assert hooks.installed_command(settings) is None
    hooks.install(settings)
    assert hooks.installed_command(settings) == "uvx cc-steer scan --no-sync"
    hooks.uninstall(settings)
    assert hooks.installed_command(settings) is None
