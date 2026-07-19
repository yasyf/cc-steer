from __future__ import annotations

import json
import sys
import threading
import types
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

RECENT_PROMPT = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}]

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


async def test_draft_runs_inference_on_the_caller_thread(
    tmp_path: Path, fake_mlx: FakeMlxLm, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path)
    ran_on: list[int] = []

    def fake_decide(context_tail: str) -> Draft:
        ran_on.append(threading.get_ident())
        return Draft("steer text", 0.1)

    monkeypatch.setattr(drafter, "decide", fake_decide)
    await drafter.draft([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}])
    assert ran_on == [threading.get_ident()]


class FakeArray:
    def __getitem__(self, _key: object) -> FakeArray:
        return self

    def astype(self, _dtype: object) -> FakeArray:
        return self

    def __sub__(self, _other: object) -> FakeArray:
        return self


class FakeModel:
    def __call__(self, _prompt: object) -> FakeArray:
        return FakeArray()


class RecordingMlxLm:
    def __init__(self) -> None:
        self.loads = 0

    def load(self, base_model: str, adapter_path: str) -> tuple[FakeModel, object]:
        self.loads += 1
        return FakeModel(), object()


@pytest.fixture
def recording_mlx(monkeypatch: pytest.MonkeyPatch) -> RecordingMlxLm:
    fake = RecordingMlxLm()
    monkeypatch.setattr("cc_steer.watcher.drafter_mlx._mlx_lm", lambda: fake)
    return fake


@pytest.fixture
def fake_mx(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"clear_cache": 0}
    core = types.ModuleType("mlx.core")
    core.clear_cache = lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1)
    core.array = lambda _prefix: FakeArray()
    core.float32 = object()
    core.logsumexp = lambda _a: FakeArray()
    core.exp = lambda _a: 0.5
    mlx_mod = types.ModuleType("mlx")
    mlx_mod.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return calls


def test_construction_loads_no_weights(tmp_path: Path, recording_mlx: RecordingMlxLm) -> None:
    register_watcher(tmp_path)
    MlxDrafter(root=tmp_path)
    assert recording_mlx.loads == 0
    assert MlxDrafter(root=tmp_path).resource.loaded is False


async def test_draft_loads_once_then_reuses(
    tmp_path: Path, recording_mlx: RecordingMlxLm, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path)
    monkeypatch.setattr(drafter, "decide", lambda _tail: Draft("steer", 0.1))
    await drafter.draft(RECENT_PROMPT)
    assert recording_mlx.loads == 1
    await drafter.draft(RECENT_PROMPT)
    assert recording_mlx.loads == 1


async def test_idle_sweep_evicts_and_next_draft_reloads(
    tmp_path: Path, recording_mlx: RecordingMlxLm, fake_mx: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path, idle_ttl_s=1.0)
    monkeypatch.setattr(drafter, "decide", lambda _tail: Draft("steer", 0.1))
    await drafter.draft(RECENT_PROMPT)
    assert recording_mlx.loads == 1 and drafter.resource.loaded
    await drafter.resource.sweep(now=1e12)
    assert fake_mx["clear_cache"] == 1
    assert drafter.resource.loaded is False
    assert drafter._loaded is None
    await drafter.draft(RECENT_PROMPT)
    assert recording_mlx.loads == 2


async def test_sweep_refuses_while_a_use_is_inflight(
    tmp_path: Path, recording_mlx: RecordingMlxLm, fake_mx: dict[str, int]
) -> None:
    register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path)
    async with drafter.resource.use():
        await drafter.resource.sweep(now=1e12)
        assert drafter.resource.loaded
        assert fake_mx["clear_cache"] == 0


def test_sync_nosteer_prob_self_wakes_a_fresh_drafter(
    tmp_path: Path, recording_mlx: RecordingMlxLm, fake_mx: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_watcher(tmp_path)
    drafter = MlxDrafter(root=tmp_path)
    monkeypatch.setattr(drafter, "_prefix_and_sentinel", lambda _tail: ([1, 2, 3], 7))
    assert recording_mlx.loads == 0
    assert drafter.nosteer_prob("some context tail") == 0.5
    assert recording_mlx.loads == 1
