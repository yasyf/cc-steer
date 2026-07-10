from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from cc_steer import registry

if TYPE_CHECKING:
    from pathlib import Path


def register(root: Path, *, blob: bytes, metrics: dict[str, float] | None = None) -> registry.VersionInfo:
    metadata = {"dataset_digest": "digest-1", "config": {"C": 4.0}, "metrics": metrics or {"pr_auc": 0.9}}
    return registry.register("gate", {"model.joblib": blob}, metadata, root=root)


class TestRegister:
    def test_writes_versioned_dir_with_files_and_metadata(self, tmp_path: Path) -> None:
        info = register(tmp_path, blob=b"weights")
        assert registry.VERSION_PATTERN.match(info.version)
        assert info.number == 1
        assert (info.path / "model.joblib").read_bytes() == b"weights"
        stored = json.loads((info.path / registry.METADATA_NAME).read_text())
        assert stored["component"] == "gate"
        assert stored["version"] == info.version
        assert stored["dataset_digest"] == "digest-1"
        assert stored["config"] == {"C": 4.0}
        assert stored["created_at"]
        assert registry.versions("gate", root=tmp_path)[0].metadata == stored

    def test_does_not_flip_current(self, tmp_path: Path) -> None:
        register(tmp_path, blob=b"one")
        assert registry.current("gate", root=tmp_path) is None

    def test_numbers_increase_and_sort_oldest_first(self, tmp_path: Path) -> None:
        first = register(tmp_path, blob=b"one")
        second = register(tmp_path, blob=b"two")
        assert [info.version for info in registry.versions("gate", root=tmp_path)] == [first.version, second.version]
        assert second.number == 2

    def test_copies_path_files(self, tmp_path: Path) -> None:
        artifact = tmp_path / "artifact.bin"
        artifact.write_bytes(b"from-path")
        info = registry.register("gate", {"model.joblib": artifact}, {}, root=tmp_path / "models")
        assert (info.path / "model.joblib").read_bytes() == b"from-path"

    def test_empty_files_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(registry.RegistryError, match="empty"):
            registry.register("gate", {}, {}, root=tmp_path)

    def test_env_root_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CC_STEER_MODELS", str(tmp_path / "override"))
        info = registry.register("gate", {"model.joblib": b"x"}, {})
        assert info.path.is_relative_to(tmp_path / "override")
        assert registry.versions("gate")[0].version == info.version


class TestPromote:
    def test_flips_current_atomically_via_relative_symlink(self, tmp_path: Path) -> None:
        first = register(tmp_path, blob=b"one")
        second = register(tmp_path, blob=b"two")
        registry.promote("gate", first.version, root=tmp_path)
        link = tmp_path / "gate" / registry.CURRENT_LINK
        assert os.readlink(link) == first.version
        registry.promote("gate", second.version, root=tmp_path)
        assert os.readlink(link) == second.version
        promoted = registry.current("gate", root=tmp_path)
        assert promoted is not None and promoted.version == second.version
        assert not (tmp_path / "gate" / f".{registry.CURRENT_LINK}.tmp").exists()

    def test_accepts_vnnn_prefix(self, tmp_path: Path) -> None:
        info = register(tmp_path, blob=b"one")
        registry.promote("gate", f"v{info.number:03d}", root=tmp_path)
        promoted = registry.current("gate", root=tmp_path)
        assert promoted is not None and promoted.version == info.version

    def test_unknown_version_raises(self, tmp_path: Path) -> None:
        register(tmp_path, blob=b"one")
        with pytest.raises(registry.RegistryError, match="v999"):
            registry.promote("gate", "v999", root=tmp_path)


class TestRollback:
    def test_flips_to_previous_version(self, tmp_path: Path) -> None:
        first = register(tmp_path, blob=b"one")
        second = register(tmp_path, blob=b"two")
        registry.promote("gate", second.version, root=tmp_path)
        rolled = registry.rollback("gate", root=tmp_path)
        assert rolled.version == first.version
        promoted = registry.current("gate", root=tmp_path)
        assert promoted is not None and promoted.version == first.version

    def test_nothing_promoted_raises(self, tmp_path: Path) -> None:
        register(tmp_path, blob=b"one")
        with pytest.raises(registry.RegistryError, match="nothing is promoted"):
            registry.rollback("gate", root=tmp_path)

    def test_no_earlier_version_raises(self, tmp_path: Path) -> None:
        info = register(tmp_path, blob=b"one")
        registry.promote("gate", info.version, root=tmp_path)
        with pytest.raises(registry.RegistryError, match="no earlier"):
            registry.rollback("gate", root=tmp_path)


class TestPrune:
    def test_keeps_newest_and_current(self, tmp_path: Path) -> None:
        infos = [register(tmp_path, blob=f"blob-{index}".encode()) for index in range(5)]
        registry.promote("gate", infos[0].version, root=tmp_path)
        removed = registry.prune("gate", keep=2, root=tmp_path)
        assert removed == [infos[1].version, infos[2].version]
        survivors = [info.version for info in registry.versions("gate", root=tmp_path)]
        assert survivors == [infos[0].version, infos[3].version, infos[4].version]
        promoted = registry.current("gate", root=tmp_path)
        assert promoted is not None and promoted.version == infos[0].version

    def test_keep_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(registry.RegistryError, match="keep"):
            registry.prune("gate", keep=0, root=tmp_path)


class TestListing:
    def test_components_lists_only_populated_dirs(self, tmp_path: Path) -> None:
        register(tmp_path, blob=b"one")
        (tmp_path / "empty-component").mkdir()
        assert registry.components(root=tmp_path) == ["gate"]

    def test_missing_root_is_empty(self, tmp_path: Path) -> None:
        assert registry.components(root=tmp_path / "nowhere") == []
        assert registry.versions("gate", root=tmp_path / "nowhere") == []
        assert registry.current("gate", root=tmp_path / "nowhere") is None
