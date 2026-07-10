from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from cc_steer import registry
from cc_steer.rendering import DRAFT_CHAR_CAP, tail_messages
from cc_steer.watcher.cascade import flattened
from cc_steer.watcher.drafter_mlx import ADAPTER_CONFIG_NAME, ADAPTER_NAME, MlxDrafter
from cc_steer.watcher.types import Draft

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

METADATA = {
    "base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "thresholds": {"budget": 0.2373046875, "f1": 0.8828125},
    "auc": 0.8534,
    "dataset_digest": "a054d90e87d6f0a9",
    "render_version": 1,
}


class FakeMlxLm:
    def load(self, base_model: str, adapter_path: str) -> tuple[object, object]:
        return object(), object()


def register_watcher(root: Path, metadata: dict[str, Any] | None = None, *, promote: bool = True) -> str:
    info = registry.register(
        "watcher",
        {ADAPTER_NAME: b"fake-safetensors", ADAPTER_CONFIG_NAME: json.dumps({"num_layers": 16}).encode()},
        metadata if metadata is not None else METADATA,
        root=root,
    )
    if promote:
        registry.promote("watcher", info.version, root=root)
    return info.version


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> FakeMlxLm:
    fake = FakeMlxLm()
    monkeypatch.setattr("cc_steer.watcher.drafter_mlx._mlx_lm", lambda: fake)
    return fake


def test_no_promoted_watcher_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no promoted watcher model"):
        MlxDrafter(root=tmp_path)


def test_missing_adapter_artifact_is_a_clear_error(tmp_path: Path) -> None:
    info = registry.register("watcher", {"other.bin": b"x"}, METADATA, root=tmp_path)
    registry.promote("watcher", info.version, root=tmp_path)
    with pytest.raises(RuntimeError, match=ADAPTER_NAME):
        MlxDrafter(root=tmp_path)


def test_missing_threshold_key_is_a_clear_error(tmp_path: Path, fake_mlx: FakeMlxLm) -> None:
    register_watcher(tmp_path, {"base_model": "m", "render_version": 1})
    with pytest.raises(RuntimeError, match="thresholds"):
        MlxDrafter(root=tmp_path)


def test_metadata_drives_threshold_base_and_render_version(tmp_path: Path, fake_mlx: FakeMlxLm) -> None:
    version = register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path)
    assert drafter.version.version == version
    assert drafter.base_model == METADATA["base_model"]
    assert drafter.threshold == 0.2373046875
    assert drafter.operating_point == "budget"
    assert drafter.render_version == 1


def test_operating_point_and_explicit_threshold_override(tmp_path: Path, fake_mlx: FakeMlxLm) -> None:
    register_watcher(tmp_path)
    assert MlxDrafter(root=tmp_path, operating_point="f1").threshold == 0.8828125
    explicit = MlxDrafter(root=tmp_path, threshold=0.5)
    assert explicit.threshold == 0.5
    assert explicit.operating_point == "explicit"


async def test_draft_feeds_decide_the_tail_capped_training_rendering(
    tmp_path: Path, fake_mlx: FakeMlxLm, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path)
    seen: list[str] = []

    def fake_decide(context_tail: str) -> Draft:
        seen.append(context_tail)
        return Draft("steer text", 0.1)

    monkeypatch.setattr(drafter, "decide", fake_decide)
    prompt = [
        {"role": "user", "content": "x" * (DRAFT_CHAR_CAP + 100)},
        {"role": "assistant", "content": "recent turn"},
    ]
    result = await drafter.draft(prompt)
    assert result == Draft("steer text", 0.1)
    assert seen == [flattened(tail_messages(prompt, DRAFT_CHAR_CAP))]
    assert len(seen[0]) < DRAFT_CHAR_CAP + 100
