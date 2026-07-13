"""Global Claude Code hook wiring: the async ``SessionEnd`` scan and the sync ``UserPromptSubmit`` steer.

Both hooks must be machine-global — every session on the machine, including ones in repositories that
carry no project-local hooks — so they live in the user-level ``~/.claude/settings.json``. The scan
hook feeds continual collection; the live hook surfaces a queued steer at prompt submit. Each merge is
idempotent and preserves every hook group this module does not own, mirroring captain-hook's settings
conventions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_PREFIX = "uvx cc-steer"
EVENT = "SessionEnd"
LIVE_EVENT = "UserPromptSubmit"
LIVE_TIMEOUT_S = 5

type Owns = Callable[[dict[str, Any]], bool]


def default_settings_path() -> Path:
    """The user-level Claude Code settings file; hooks here fire in every project."""
    return Path.home() / ".claude" / "settings.json"


def scan_command(prefix: str = DEFAULT_PREFIX) -> str:
    """The scan hook command: an incremental scan that defers the HF sync to the pipeline."""
    return f"{prefix} scan --no-sync"


def live_command(prefix: str = DEFAULT_PREFIX) -> str:
    """The live hook command: pop and surface the freshest queued steer for the current session."""
    return f"{prefix} live hook"


def hook_group(prefix: str = DEFAULT_PREFIX) -> dict[str, Any]:
    """The SessionEnd scan group this module owns, run async so session exit never blocks."""
    return {"hooks": [{"type": "command", "command": scan_command(prefix), "async": True}]}


def live_group(prefix: str = DEFAULT_PREFIX) -> dict[str, Any]:
    """The UserPromptSubmit steer group this module owns, run synchronously so context reaches the model.

    The ``timeout`` hard-bounds a stuck hook at the harness level, backing the hook's own ~200ms budget.
    """
    return {"hooks": [{"type": "command", "command": live_command(prefix), "timeout": LIVE_TIMEOUT_S}]}


def group_commands(group: dict[str, Any]) -> tuple[str, ...]:
    return tuple(h.get("command") or "" for h in group.get("hooks") or [])


def is_cc_steer_group(group: dict[str, Any]) -> bool:
    """Whether a hook group is the cc-steer scan group: any command invoking ``cc-steer scan``."""
    return any("cc-steer" in command and " scan" in command for command in group_commands(group))


def is_live_group(group: dict[str, Any]) -> bool:
    """Whether a hook group is the cc-steer live group: any command invoking ``cc-steer live hook``."""
    return any("cc-steer" in command and " live hook" in command for command in group_commands(group))


def install(settings_path: Path | None = None, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Idempotently merges the SessionEnd scan hook into the settings file.

    Groups owned by other tools are preserved untouched; an existing cc-steer
    group is replaced in place so a prefix change updates rather than duplicates.

    Returns:
        ``"added"``, ``"updated"``, or ``"unchanged"``.
    """
    return merge(settings_path or default_settings_path(), EVENT, hook_group(prefix), is_cc_steer_group)


def install_live(settings_path: Path | None = None, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Idempotently merges the UserPromptSubmit live-steer hook into the settings file.

    Returns:
        ``"added"``, ``"updated"``, or ``"unchanged"``.
    """
    return merge(settings_path or default_settings_path(), LIVE_EVENT, live_group(prefix), is_live_group)


def uninstall(settings_path: Path | None = None) -> str:
    """Removes the SessionEnd scan hook, leaving every other group untouched.

    Returns:
        ``"removed"`` or ``"absent"``.
    """
    return unmerge(settings_path or default_settings_path(), EVENT, is_cc_steer_group)


def uninstall_live(settings_path: Path | None = None) -> str:
    """Removes the UserPromptSubmit live-steer hook, leaving every other group untouched.

    Returns:
        ``"removed"`` or ``"absent"``.
    """
    return unmerge(settings_path or default_settings_path(), LIVE_EVENT, is_live_group)


def installed_command(settings_path: Path | None = None) -> str | None:
    """The scan command currently wired at SessionEnd, or ``None``."""
    return installed(settings_path or default_settings_path(), EVENT, is_cc_steer_group)


def installed_live_command(settings_path: Path | None = None) -> str | None:
    """The live command currently wired at UserPromptSubmit, or ``None``."""
    return installed(settings_path or default_settings_path(), LIVE_EVENT, is_live_group)


def merge(path: Path, event: str, group: dict[str, Any], owns: Owns) -> str:
    existing = json.loads(path.read_text()) if path.exists() else {}
    hooks: dict[str, list[dict[str, Any]]] = existing.get("hooks") or {}
    groups = hooks.get(event) or []
    kept = [g for g in groups if not owns(g)]
    own = [g for g in groups if owns(g)]
    if own == [group]:
        return "unchanged"
    write_settings(path, existing | {"hooks": hooks | {event: [*kept, group]}})
    return "updated" if own else "added"


def unmerge(path: Path, event: str, owns: Owns) -> str:
    if not path.exists():
        return "absent"
    existing = json.loads(path.read_text())
    hooks: dict[str, list[dict[str, Any]]] = existing.get("hooks") or {}
    groups = hooks.get(event) or []
    kept = [g for g in groups if not owns(g)]
    if kept == groups:
        return "absent"
    merged = hooks | {event: kept} if kept else {k: v for k, v in hooks.items() if k != event}
    write_settings(path, existing | {"hooks": merged})
    return "removed"


def installed(path: Path, event: str, owns: Owns) -> str | None:
    if not path.exists():
        return None
    groups = (json.loads(path.read_text()).get("hooks") or {}).get(event) or []
    return next(
        (
            command
            for group in groups
            if owns(group)
            for hook in group.get("hooks") or []
            if "cc-steer" in (command := hook.get("command") or "")
        ),
        None,
    )


def write_settings(settings_path: Path, data: dict[str, Any]) -> None:
    """Writes settings atomically: temp file next to the target, then rename."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(f"{settings_path.suffix}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, settings_path)
