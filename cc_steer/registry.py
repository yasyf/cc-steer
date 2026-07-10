"""The on-disk model registry: versioned artifacts under ``~/.cc-steer/models``.

Every lab-trained component (the stage-1 ``gate`` today, the ``watcher`` LoRA
once E2 lands a recipe) ships as an immutable version directory —
``<component>/v<NNN>-<YYYYMMDD>-<digest12>/`` holding the artifact files plus a
``metadata.json`` — and a per-component ``current`` symlink names the one the
live daemon loads. Registration never flips ``current``: promotion is a
separate, atomic symlink swap, so a bad candidate can sit registered without
ever being served, and a rollback is one flip back. The root is overridable
(the ``root`` parameter, or env ``CC_STEER_MODELS``) so tests and the lab's
retrain flow stay hermetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

MODELS_DIR = Path.home() / ".cc-steer" / "models"
METADATA_NAME = "metadata.json"
CURRENT_LINK = "current"
VERSION_PATTERN = re.compile(r"^v(\d{3,})-(\d{8})-([0-9a-f]{12})$")
DIGEST_CHARS = 12


class RegistryError(RuntimeError):
    """The registry cannot satisfy the request: unknown version, nothing to roll back to."""


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """One registered model version.

    Attributes:
        component: The component the version belongs to.
        version: The full version directory name, ``v<NNN>-<YYYYMMDD>-<digest12>``.
        path: The version directory holding the artifact files and metadata.
        metadata: The parsed ``metadata.json``.
    """

    component: str
    version: str
    path: Path
    metadata: dict[str, Any]

    @property
    def number(self) -> int:
        """The monotonically increasing NNN in ``v<NNN>-...``."""
        match = VERSION_PATTERN.match(self.version)
        assert match is not None
        return int(match.group(1))


def models_root(root: Path | None = None) -> Path:
    """The registry root: the parameter, env ``CC_STEER_MODELS``, or ``~/.cc-steer/models``."""
    if root is not None:
        return root
    override = os.environ.get("CC_STEER_MODELS")
    return Path(override) if override else MODELS_DIR


def components(*, root: Path | None = None) -> list[str]:
    """The components with at least one registered version, sorted."""
    base = models_root(root)
    if not base.is_dir():
        return []
    return sorted(child.name for child in base.iterdir() if child.is_dir() and versions(child.name, root=root))


def versions(component: str, *, root: Path | None = None) -> list[VersionInfo]:
    """Every registered version of a component, oldest first."""
    component_dir = models_root(root) / component
    if not component_dir.is_dir():
        return []
    found = [
        _info(component, child)
        for child in component_dir.iterdir()
        if child.is_dir() and not child.is_symlink() and VERSION_PATTERN.match(child.name)
    ]
    return sorted(found, key=lambda info: info.number)


def current(component: str, *, root: Path | None = None) -> VersionInfo | None:
    """The promoted version the ``current`` symlink names, or None when nothing is promoted."""
    link = models_root(root) / component / CURRENT_LINK
    if not link.is_symlink():
        return None
    target = link.parent / os.readlink(link)
    if not target.is_dir():
        return None
    return _info(component, target)


def register(
    component: str, files: Mapping[str, bytes | Path], metadata: Mapping[str, object], *, root: Path | None = None
) -> VersionInfo:
    """Writes a new immutable version directory; never flips ``current``.

    The directory name embeds the next version number, today's date, and a
    12-hex content digest over the artifact files. ``metadata.json`` is the
    caller's metadata stamped with ``component``, ``version``, and
    ``created_at``.

    Args:
        files: Artifact file name to content — raw bytes, or a path to copy.
        metadata: The version's provenance: dataset digest, config, metrics, thresholds.

    Returns:
        The freshly registered :class:`VersionInfo`.
    """
    if not files:
        raise RegistryError(f"refusing to register an empty {component} version")
    existing = versions(component, root=root)
    number = existing[-1].number + 1 if existing else 1
    now = datetime.now(UTC)
    name = f"v{number:03d}-{now:%Y%m%d}-{_digest(files)}"
    path = models_root(root) / component / name
    path.mkdir(parents=True)
    for filename, content in files.items():
        destination = path / filename
        if isinstance(content, bytes):
            destination.write_bytes(content)
        else:
            shutil.copy2(content, destination)
    stamped = dict(metadata) | {"component": component, "version": name, "created_at": now.isoformat()}
    (path / METADATA_NAME).write_text(json.dumps(stamped, indent=2, sort_keys=True, default=str) + "\n")
    return VersionInfo(component=component, version=name, path=path, metadata=stamped)


def promote(component: str, version: str, *, root: Path | None = None) -> None:
    """Atomically flips ``current`` to the named version.

    ``version`` is the full directory name or its ``v<NNN>`` prefix. The flip
    writes a temporary symlink and renames it over ``current``, so a reader
    never sees a missing or half-written link.
    """
    info = _resolve(component, version, root=root)
    link = info.path.parent / CURRENT_LINK
    staging = info.path.parent / f".{CURRENT_LINK}.tmp"
    staging.unlink(missing_ok=True)
    staging.symlink_to(info.version)
    os.replace(staging, link)


def rollback(component: str, *, root: Path | None = None) -> VersionInfo:
    """Flips ``current`` back to the version registered immediately before it.

    Returns:
        The version now current.

    Raises:
        RegistryError: When nothing is promoted or nothing earlier exists.
    """
    promoted = current(component, root=root)
    if promoted is None:
        raise RegistryError(f"nothing is promoted for {component}; nothing to roll back")
    older = [info for info in versions(component, root=root) if info.number < promoted.number]
    if not older:
        raise RegistryError(f"{component} {promoted.version} has no earlier version to roll back to")
    promote(component, older[-1].version, root=root)
    return older[-1]


def prune(component: str, *, keep: int = 3, root: Path | None = None) -> list[str]:
    """Removes all but the newest ``keep`` versions; the current one always survives.

    Returns:
        The removed version names, oldest first.
    """
    if keep < 1:
        raise RegistryError(f"keep must be >= 1, got {keep}")
    promoted = current(component, root=root)
    survivors = {info.version for info in versions(component, root=root)[-keep:]}
    if promoted is not None:
        survivors.add(promoted.version)
    removed = []
    for info in versions(component, root=root):
        if info.version not in survivors:
            shutil.rmtree(info.path)
            removed.append(info.version)
    return removed


def _info(component: str, path: Path) -> VersionInfo:
    metadata_path = path / METADATA_NAME
    metadata = dict(json.loads(metadata_path.read_text())) if metadata_path.exists() else {}
    return VersionInfo(component=component, version=path.name, path=path, metadata=metadata)


def _resolve(component: str, version: str, *, root: Path | None) -> VersionInfo:
    matches = [info for info in versions(component, root=root) if version in (info.version, info.version.split("-")[0])]
    if not matches:
        known = ", ".join(info.version for info in versions(component, root=root)) or "none registered"
        raise RegistryError(f"no {component} version {version!r} ({known})")
    return matches[-1]


def _digest(files: Mapping[str, bytes | Path]) -> str:
    hasher = hashlib.sha256()
    for filename in sorted(files):
        content = files[filename]
        hasher.update(filename.encode())
        hasher.update(b"\0")
        hasher.update(content if isinstance(content, bytes) else content.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:DIGEST_CHARS]
