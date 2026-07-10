"""Global Claude Code hook wiring: an async ``SessionEnd`` scan in every project.

Collection must be machine-global — every session on the machine, including ones
in repositories that carry no project-local hooks — so the scan hook lives in the
user-level ``~/.claude/settings.json``. The merge is idempotent and preserves
every hook group this module does not own, mirroring captain-hook's settings
conventions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PREFIX = "uvx cc-steer"
EVENT = "SessionEnd"


def default_settings_path() -> Path:
    """The user-level Claude Code settings file; hooks here fire in every project."""
    return Path.home() / ".claude" / "settings.json"


def scan_command(prefix: str = DEFAULT_PREFIX) -> str:
    """The hook command: an incremental scan that defers the HF sync to the pipeline."""
    return f"{prefix} scan --no-sync"


def hook_group(prefix: str = DEFAULT_PREFIX) -> dict[str, Any]:
    """The settings hook group this module owns, run async so session exit never blocks."""
    return {"hooks": [{"type": "command", "command": scan_command(prefix), "async": True}]}


def is_cc_steer_group(group: dict[str, Any]) -> bool:
    """Whether a hook group is owned by this module: any command invoking ``cc-steer scan``."""
    commands = (h.get("command") or "" for h in group.get("hooks") or [])
    return any("cc-steer" in command and " scan" in command for command in commands)


def install(settings_path: Path | None = None, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Idempotently merges the SessionEnd scan hook into the settings file.

    Groups owned by other tools are preserved untouched; an existing cc-steer
    group is replaced in place so a prefix change updates rather than duplicates.

    Returns:
        ``"added"``, ``"updated"``, or ``"unchanged"``.
    """
    path = settings_path or default_settings_path()
    existing = json.loads(path.read_text()) if path.exists() else {}
    hooks: dict[str, list[dict[str, Any]]] = existing.get("hooks") or {}
    groups = hooks.get(EVENT) or []
    kept = [group for group in groups if not is_cc_steer_group(group)]
    own = [group for group in groups if is_cc_steer_group(group)]
    fresh = hook_group(prefix)
    if own == [fresh]:
        return "unchanged"
    write_settings(path, existing | {"hooks": hooks | {EVENT: [*kept, fresh]}})
    return "updated" if own else "added"


def uninstall(settings_path: Path | None = None) -> str:
    """Removes the SessionEnd scan hook, leaving every other group untouched.

    Returns:
        ``"removed"`` or ``"absent"``.
    """
    path = settings_path or default_settings_path()
    if not path.exists():
        return "absent"
    existing = json.loads(path.read_text())
    hooks: dict[str, list[dict[str, Any]]] = existing.get("hooks") or {}
    groups = hooks.get(EVENT) or []
    kept = [group for group in groups if not is_cc_steer_group(group)]
    if kept == groups:
        return "absent"
    merged = hooks | {EVENT: kept} if kept else {k: v for k, v in hooks.items() if k != EVENT}
    write_settings(path, existing | {"hooks": merged})
    return "removed"


def installed_command(settings_path: Path | None = None) -> str | None:
    """The scan command currently wired at SessionEnd, or ``None``."""
    path = settings_path or default_settings_path()
    if not path.exists():
        return None
    groups = (json.loads(path.read_text()).get("hooks") or {}).get(EVENT) or []
    for group in groups:
        if is_cc_steer_group(group):
            for hook in group.get("hooks") or []:
                if "cc-steer" in (command := hook.get("command") or ""):
                    return command
    return None


def write_settings(settings_path: Path, data: dict[str, Any]) -> None:
    """Writes settings atomically: temp file next to the target, then rename."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(f"{settings_path.suffix}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, settings_path)
