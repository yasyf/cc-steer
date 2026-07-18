from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from athome.research import loop
from athome.research.driver import StubDriver, StubProposal
from athome.research.spec import Budget, ExperimentSpec
from athome.train import BASE_MODELS, METRIC_FILE, METRIC_KEY

from cc_steer.retrain import sweep
from cc_steer.retrain.watcher import WatcherRecipe

SPEC_PATH = Path(__file__).resolve().parents[1] / "experiments" / "watcher-base-sweep.toml"

STUB_SCORER = (
    "import json\n"
    "from pathlib import Path\n"
    "import anyio\n"
    "from athome.train import write_metric\n"
    "knob = json.loads(Path('watcher_recipe.json').read_text())['knob']\n"
    "anyio.run(write_metric, float(knob))\n"
)


def servable_9b_table() -> dict[str, object]:
    return dict(BASE_MODELS) | {"qwen3.5-9b": dataclasses.replace(BASE_MODELS["qwen3.5-9b"], serves_locally=True)}


def rebased(base_key: str, table: dict[str, object]) -> WatcherRecipe:
    spec = table[base_key]
    return dataclasses.replace(WatcherRecipe.default(), tinker_model=spec.tinker, mlx_id=spec.mlx)


class TestServableArms:
    def test_real_table_yields_only_qwen3_8b(self) -> None:
        assert sweep.servable_arms() == ("qwen3-8b",)
        assert sweep.servable_arms(BASE_MODELS) == ("qwen3-8b",)

    def test_fake_table_with_servable_9b_adds_the_arm(self) -> None:
        assert sweep.servable_arms(servable_9b_table()) == ("qwen3-8b", "qwen3.5-9b")

    def test_base_for_recipe_default_resolves_qwen3_8b(self) -> None:
        assert sweep.base_for_recipe(WatcherRecipe.default()) is BASE_MODELS["qwen3-8b"]

    def test_base_for_recipe_resolves_servable_9b(self) -> None:
        table = servable_9b_table()
        assert sweep.base_for_recipe(rebased("qwen3.5-9b", table), table) is table["qwen3.5-9b"]

    def test_base_for_recipe_refuses_non_servable_arm(self) -> None:
        # The real table marks 9B serves_locally=False, so a 9B recipe has no servable match.
        with pytest.raises(sweep.UnservableArm):
            sweep.base_for_recipe(rebased("qwen3.5-9b", dict(BASE_MODELS)))

    def test_base_for_recipe_refuses_unknown_base(self) -> None:
        recipe = dataclasses.replace(WatcherRecipe.default(), tinker_model="Foo/Bar", mlx_id="foo/bar-4bit")
        with pytest.raises(sweep.UnservableArm):
            sweep.base_for_recipe(recipe)


class TestScoreWatcher:
    def test_writes_metric_and_returns_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        seen: dict[str, object] = {}

        def fake(recipe: WatcherRecipe, *, dataset_dir: Path | None, eval_root: Path | None) -> float:
            seen.update(recipe=recipe, dataset_dir=dataset_dir, eval_root=eval_root)
            return 0.873

        recipe = WatcherRecipe.default()
        result = sweep.score_watcher(
            recipe, train_and_score=fake, dataset_dir=tmp_path / "ds", eval_root=tmp_path / "ev"
        )
        assert result == 0.873
        assert seen == {"recipe": recipe, "dataset_dir": tmp_path / "ds", "eval_root": tmp_path / "ev"}
        assert json.loads((work / METRIC_FILE).read_text()) == {METRIC_KEY: 0.873}

    def test_pure_observer_writes_only_the_metric_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        sweep.score_watcher(WatcherRecipe.default(), train_and_score=lambda recipe, *, dataset_dir, eval_root: 0.5)
        assert {path.name for path in work.iterdir()} == {METRIC_FILE}


class TestSweepSpec:
    def test_recipe_is_the_sole_mutable_path(self) -> None:
        spec = ExperimentSpec.load(SPEC_PATH)
        assert spec.mutable_paths == ("watcher_recipe.json",)
        assert spec.direction == "max"
        assert spec.metric_command[0] == "cc-steer"
        assert "score-watcher" in spec.metric_command

    def test_metric_channel_matches_write_metric(self) -> None:
        # score_watcher reports via write_metric -> METRIC_FILE/METRIC_KEY; the loop reads the same.
        spec = ExperimentSpec.load(SPEC_PATH)
        assert spec.metric_key == METRIC_KEY
        assert spec.metric_file == METRIC_FILE


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(repo),
            "PATH": os.environ["PATH"],
        },
    )


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "watcher_recipe.json").write_text(json.dumps({"knob": 1}))
    (repo / "watcher-base-sweep.toml").write_text("# immutable scoring boundary\n")
    (repo / "score.py").write_text(STUB_SCORER)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "sweep@test")
    git(repo, "config", "user.name", "sweep")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


class TestStubDriverDryRun:
    def test_full_loop_keeps_the_strict_improvements(self, tmp_path: Path) -> None:
        # Spend-free dry run of the greedy loop; the stub scorer stands in for the paid metric_command.
        repo = init_repo(tmp_path)
        spec = ExperimentSpec(
            name="watcher-base-sweep-dryrun",
            metric_command=(sys.executable, "score.py"),
            metric_key="metric",
            direction="max",
            mutable_paths=("watcher_recipe.json",),
            immutable_paths=("watcher-base-sweep.toml",),
            budget=Budget(max_units=3, hard_kill_s=120.0),
        )
        proposals = iter(
            [
                StubProposal({"watcher_recipe.json": json.dumps({"knob": 3})}),
                StubProposal({"watcher_recipe.json": json.dumps({"knob": 2})}),
                StubProposal({"watcher_recipe.json": json.dumps({"knob": 5})}),
            ]
        )
        result = anyio.run(lambda: loop.run(spec, driver=StubDriver(proposals), repo=repo))
        assert result.kept == 2
        assert result.best is not None
        assert result.best.metric == 5.0

    def test_candidate_may_not_edit_the_immutable_boundary(self, tmp_path: Path) -> None:
        # A proposal touching an immutable path is discarded unscored, never kept.
        repo = init_repo(tmp_path)
        spec = ExperimentSpec(
            name="watcher-base-sweep-immutable",
            metric_command=(sys.executable, "score.py"),
            metric_key="metric",
            direction="max",
            mutable_paths=("watcher_recipe.json",),
            immutable_paths=("watcher-base-sweep.toml",),
            budget=Budget(max_units=1, hard_kill_s=120.0),
        )
        proposals = iter([StubProposal({"watcher-base-sweep.toml": "# tampered\n"})])
        result = anyio.run(lambda: loop.run(spec, driver=StubDriver(proposals), repo=repo))
        assert result.kept == 0


def test_cli_score_watcher_invokes_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from cc_steer import cli

    monkeypatch.setattr(sweep, "score_watcher", lambda *, recipe: 0.9123)
    result = CliRunner().invoke(cli.main, ["score-watcher"])
    assert result.exit_code == 0, result.output
    assert "0.9123" in result.output
