from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from cc_steer import registry
from cc_steer.cli import main
from cc_steer.retrain import evalset, lexical
from cc_steer.retrain import watcher as w

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_gate_dispatch_passes_force(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(*, force: bool, fresh_epoch: bool) -> str:
        seen["force"] = force
        seen["fresh_epoch"] = fresh_epoch
        return "gate: promoted v001-x"

    monkeypatch.setattr(lexical, "retrain_gate", fake)
    result = runner.invoke(main, ["retrain", "--component", "gate", "--force"])
    assert result.exit_code == 0, result.output
    assert seen == {"force": True, "fresh_epoch": False}
    assert "gate: promoted v001-x" in result.output


def test_fresh_epoch_threads_to_both_lanes(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        lexical,
        "retrain_gate",
        lambda *, force, fresh_epoch: seen.update(gate=fresh_epoch) or "gate: fresh-epoch promoted v001-x",
    )
    monkeypatch.setattr(
        w,
        "retrain_watcher",
        lambda *, force, fresh_epoch, recipe: seen.update(watcher=fresh_epoch) or "watcher: fresh-epoch promoted v002",
    )
    assert runner.invoke(main, ["retrain", "--component", "gate", "--fresh-epoch"]).exit_code == 0
    assert runner.invoke(main, ["retrain", "--component", "watcher", "--fresh-epoch"]).exit_code == 0
    assert seen == {"gate": True, "watcher": True}


def test_fresh_epoch_rejected_with_register_adapter(runner: CliRunner, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata_json = tmp_path / "meta.json"
    metadata_json.write_text("{}")
    result = runner.invoke(
        main,
        ["retrain", "--component", "watcher", "--fresh-epoch", "--register-adapter", str(adapter),
         "--metadata-json", str(metadata_json)],
    )
    assert result.exit_code != 0
    assert "--fresh-epoch" in result.output


def test_gate_rejects_watcher_only_options(runner: CliRunner, tmp_path: Path) -> None:
    recipe = tmp_path / "r.json"
    recipe.write_text("{}")
    result = runner.invoke(main, ["retrain", "--component", "gate", "--recipe", str(recipe)])
    assert result.exit_code != 0
    assert "watcher only" in result.output


def test_watcher_dispatch_defaults_to_the_e8_recipe(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(*, force: bool, fresh_epoch: bool, recipe: w.WatcherRecipe) -> str:
        seen["force"] = force
        seen["fresh_epoch"] = fresh_epoch
        seen["recipe"] = recipe
        return "watcher: promoted v002-x"

    monkeypatch.setattr(w, "retrain_watcher", fake)
    result = runner.invoke(main, ["retrain", "--component", "watcher"])
    assert result.exit_code == 0, result.output
    assert seen == {"force": False, "fresh_epoch": False, "recipe": w.WatcherRecipe.default()}
    assert "watcher: promoted v002-x" in result.output


def test_watcher_dispatch_parses_recipe(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fields = {
        "tinker_model": "Qwen/Qwen3-8B",
        "mlx_id": "mlx-community/Qwen3-8B-4bit",
        "rank": 16,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "epochs": 1,
        "checkpoint_fracs": [0.5, 1.0],
        "max_tokens": 2048,
        "render_version": 2,
        "val_n": 50,
        "oversample_corrective": 2.0,
        "budget_fires_per_100": 3.0,
        "spend_cap_usd": 5.0,
        "diagnostic_rows": 8,
        "diagnostic_tolerance": 0.1,
        "seed": 7,
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(fields))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        w, "retrain_watcher", lambda *, force, fresh_epoch, recipe: seen.update(recipe=recipe) or "watcher: skipped"
    )
    result = runner.invoke(main, ["retrain", "--component", "watcher", "--recipe", str(recipe_path)])
    assert result.exit_code == 0, result.output
    assert seen["recipe"] == w.WatcherRecipe(**fields)
    assert seen["recipe"].rank == 16


def test_register_adapter_dispatch(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata_json = tmp_path / "meta.json"
    metadata = {"base_model": "mlx-community/Qwen3-8B-4bit", "render_version": 2, "thresholds": {"budget": 0.2}}
    metadata_json.write_text(json.dumps(metadata))
    seen: dict[str, Any] = {}

    def fake(adapter_dir: Path, *, metadata: dict[str, Any]) -> str:
        seen["dir"] = adapter_dir
        seen["metadata"] = metadata
        return "watcher: registered and promoted v009-x"

    monkeypatch.setattr(w, "register_watcher", fake)
    args = ["--register-adapter", str(adapter), "--metadata-json", str(metadata_json)]
    result = runner.invoke(main, ["retrain", "--component", "watcher", *args])
    assert result.exit_code == 0, result.output
    assert str(seen["dir"]) == str(adapter)
    assert seen["metadata"] == metadata
    assert "registered and promoted v009-x" in result.output


def test_register_adapter_requires_metadata(runner: CliRunner, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    result = runner.invoke(main, ["retrain", "--component", "watcher", "--register-adapter", str(adapter)])
    assert result.exit_code != 0
    assert "--metadata-json" in result.output


PLACEHOLDERS = {"A": "register", "M": "metadata", "R": "recipe", "S": "seed"}


@pytest.fixture
def opt_paths(tmp_path: Path) -> dict[str, str]:
    (adapter := tmp_path / "adapter").mkdir()
    (recipe := tmp_path / "recipe.json").write_text("{}")
    (metadata := tmp_path / "meta.json").write_text("{}")
    (cache := tmp_path / "cache.json").write_text("{}")
    return {"register": str(adapter), "recipe": str(recipe), "metadata": str(metadata), "seed": str(cache)}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--metadata-json", "M"], "only valid with --register-adapter"),
        (["--seed-incumbent-probs", "S", "--recipe", "R"], "standalone bootstrap"),
        (["--seed-incumbent-probs", "S", "--register-adapter", "A"], "standalone bootstrap"),
        (["--seed-incumbent-probs", "S", "--metadata-json", "M"], "standalone bootstrap"),
        (["--register-adapter", "A", "--metadata-json", "M", "--recipe", "R"], "takes no --recipe"),
    ],
    ids=[
        "metadata-without-register",
        "seed-plus-recipe",
        "seed-plus-register",
        "seed-plus-metadata",
        "register-plus-recipe",
    ],
)
def test_watcher_rejects_conflicting_modes(
    runner: CliRunner, opt_paths: dict[str, str], args: list[str], message: str
) -> None:
    resolved = [opt_paths[PLACEHOLDERS[token]] if token in PLACEHOLDERS else token for token in args]
    result = runner.invoke(main, ["retrain", "--component", "watcher", *resolved])
    assert result.exit_code != 0
    assert message in result.output


def test_seed_incumbent_probs_dispatch(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text("{}")
    incumbent = registry.VersionInfo(
        component="watcher",
        version="v001-20260101-abcdef123456",
        path=tmp_path,
        metadata={"render_version": 2},
    )
    monkeypatch.setattr(registry, "current", lambda component, *, root=None: incumbent)
    seen: dict[str, Any] = {}

    def fake(path: Path, *, version: str, expected_render: int) -> Path:
        seen["version"] = version
        seen["render"] = expected_render
        return tmp_path / "probs" / "v001.json"

    monkeypatch.setattr(w, "seed_incumbent_probs", fake)
    result = runner.invoke(main, ["retrain", "--component", "watcher", "--seed-incumbent-probs", str(cache)])
    assert result.exit_code == 0, result.output
    assert seen == {"version": "v001-20260101-abcdef123456", "render": 2}
    assert "seeded incumbent" in result.output


def test_freeze_eval_freezes_all_views(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    views: list[str] = []

    def fake(view: str) -> str:
        views.append(view)
        return "abc123def456ghi"

    def fake_frame(**_: object) -> str:
        return "abc123def456ghi"

    monkeypatch.setattr(evalset, "freeze_eval", fake)
    monkeypatch.setattr(evalset, "freeze_steer_type", fake_frame)
    monkeypatch.setattr(evalset, "freeze_pick", fake_frame)
    result = runner.invoke(main, ["freeze-eval"])
    assert result.exit_code == 0, result.output
    assert views == ["gate", "watcher"]
    for view in ("gate", "watcher", "steer_type", "pick"):
        assert f"froze {view} eval (abc123def456)" in result.output
