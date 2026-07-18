from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from athome.research import loop
from athome.research.driver import StubDriver, StubProposal
from athome.research.spec import Budget, ExperimentSpec
from athome.train import BASE_MODELS, METRIC_FILE, METRIC_KEY
from athome.train.spec import EvalRow

from cc_steer.retrain import sweep
from cc_steer.retrain.watcher import WatcherRecipe

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
ARMS = ("qwen3-8b", "qwen3.5-4b", "qwen3.5-9b")


def rebased(arm: str) -> WatcherRecipe:
    spec = BASE_MODELS[arm]
    return dataclasses.replace(WatcherRecipe.default(), tinker_model=spec.tinker, mlx_id=spec.mlx)


class TestSweepArms:
    def test_every_base_model_is_an_arm(self) -> None:
        assert sweep.sweep_arms() == ARMS
        assert sweep.sweep_arms(BASE_MODELS) == ARMS

    @pytest.mark.parametrize("arm", ARMS)
    def test_base_for_recipe_resolves_every_arm(self, arm: str) -> None:
        # Servability is no filter: the once-refused qwen3.5-9b and qwen3.5-4b resolve like qwen3-8b.
        assert sweep.base_for_recipe(rebased(arm)) is BASE_MODELS[arm]

    def test_base_for_recipe_refuses_unknown_base(self) -> None:
        recipe = dataclasses.replace(WatcherRecipe.default(), tinker_model="Foo/Bar", mlx_id="foo/bar-4bit")
        with pytest.raises(sweep.UnknownArm):
            sweep.base_for_recipe(recipe)


class TestScoreWatcherRefusals:
    def _never(self, *args: object, **kwargs: object) -> float:
        raise AssertionError("train_and_score must not run once a refusal fires")

    def test_arm_mismatch_refused_before_spend(self) -> None:
        with pytest.raises(sweep.ArmMismatch):
            sweep.score_watcher(rebased("qwen3.5-9b"), arm="qwen3-8b", spend_cap_usd=30.0, train_and_score=self._never)

    def test_unknown_arm_key_refused(self) -> None:
        with pytest.raises(sweep.UnknownArm):
            sweep.score_watcher(WatcherRecipe.default(), arm="nonesuch", spend_cap_usd=30.0, train_and_score=self._never)

    def test_unknown_base_refused(self) -> None:
        recipe = dataclasses.replace(WatcherRecipe.default(), tinker_model="Foo/Bar", mlx_id="foo/bar-4bit")
        with pytest.raises(sweep.UnknownArm):
            sweep.score_watcher(recipe, arm="qwen3-8b", spend_cap_usd=30.0, train_and_score=self._never)

    def test_spend_cap_exceeded_refused_before_spend(self) -> None:
        # The packaged recipe pins spend_cap_usd=15; a $10 harness cap refuses it, never clamps.
        assert WatcherRecipe.default().spend_cap_usd == 15.0
        with pytest.raises(sweep.SpendCapExceeded):
            sweep.score_watcher(WatcherRecipe.default(), arm="qwen3-8b", spend_cap_usd=10.0, train_and_score=self._never)

    @pytest.mark.parametrize("cap", [float("nan"), float("inf"), 0.0, -1.0])
    def test_non_finite_or_nonpositive_pinned_cap_refused(self, cap: float) -> None:
        # A NaN/inf cap passes every comparison and would silently disable athome's SpendGuard.
        with pytest.raises(sweep.SpendCapExceeded):
            sweep.score_watcher(WatcherRecipe.default(), arm="qwen3-8b", spend_cap_usd=cap, train_and_score=self._never)

    def test_spend_cap_equal_is_allowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = sweep.score_watcher(
            WatcherRecipe.default(),
            arm="qwen3-8b",
            spend_cap_usd=15.0,
            train_and_score=lambda recipe, *, arm, spend_cap_usd, dataset_dir, eval_root: 0.71,
        )
        assert result == 0.71


class TestScoreReport:
    def test_metric_file_and_report_carry_the_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        seen: dict[str, object] = {}

        def fake(recipe: WatcherRecipe, *, arm: object, spend_cap_usd: float, dataset_dir: object, eval_root: object) -> float:
            seen.update(recipe=recipe, arm=arm, spend_cap_usd=spend_cap_usd, dataset_dir=dataset_dir, eval_root=eval_root)
            return 0.873

        recipe = WatcherRecipe.default()
        result = sweep.score_watcher(
            recipe, arm="qwen3-8b", spend_cap_usd=18.0, train_and_score=fake, dataset_dir=tmp_path / "ds", eval_root=tmp_path / "ev"
        )
        assert result == 0.873
        assert seen == {
            "recipe": recipe,
            "arm": BASE_MODELS["qwen3-8b"],
            "spend_cap_usd": 18.0,
            "dataset_dir": tmp_path / "ds",
            "eval_root": tmp_path / "ev",
        }
        assert json.loads((tmp_path / METRIC_FILE).read_text()) == {METRIC_KEY: 0.873}
        assert json.loads((tmp_path / sweep.SCORE_REPORT_FILE).read_text()) == {
            "metric": 0.873,
            "arm": "qwen3-8b",
            "tinker_model": BASE_MODELS["qwen3-8b"].tinker,
            "mlx_id": BASE_MODELS["qwen3-8b"].mlx,
            "serves_locally": True,
        }

    def test_report_carries_non_local_arm_posture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A non-locally-servable arm is a first-class scored arm; serves_locally rides the report as metadata.
        monkeypatch.chdir(tmp_path)
        assert BASE_MODELS["qwen3.5-9b"].serves_locally is False
        sweep.score_watcher(
            rebased("qwen3.5-9b"),
            arm="qwen3.5-9b",
            spend_cap_usd=30.0,
            train_and_score=lambda recipe, *, arm, spend_cap_usd, dataset_dir, eval_root: 0.5,
        )
        report = json.loads((tmp_path / sweep.SCORE_REPORT_FILE).read_text())
        assert report["arm"] == "qwen3.5-9b"
        assert report["serves_locally"] is False

    def test_pure_observer_writes_only_metric_and_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        sweep.score_watcher(
            WatcherRecipe.default(),
            arm="qwen3-8b",
            spend_cap_usd=18.0,
            train_and_score=lambda recipe, *, arm, spend_cap_usd, dataset_dir, eval_root: 0.5,
        )
        assert {path.name for path in work.iterdir()} == {METRIC_FILE, sweep.SCORE_REPORT_FILE}


class FakeBackend:
    def __init__(self, checkpoints: list[SimpleNamespace], scores: tuple[SimpleNamespace, ...], train_cost_usd: float, *, score_error: Exception | None = None) -> None:
        self.checkpoints = checkpoints
        self.scores = scores
        self.train_cost_usd = train_cost_usd
        self.score_error = score_error
        self.score_calls: list[SimpleNamespace] = []
        self.materialized: list[object] = []
        self.trained: list[object] = []

    async def fit(self, spec: object, *, sink: object, checkpoints: object, eval_rows: object) -> SimpleNamespace:
        return SimpleNamespace(checkpoints=self.checkpoints, train_cost_usd=self.train_cost_usd)

    async def score(self, path: str, rows: object, *, base: object, max_usd: float) -> tuple[SimpleNamespace, ...]:
        self.score_calls.append(SimpleNamespace(path=path, rows=rows, base=base, max_usd=max_usd))
        if self.score_error is not None:
            raise self.score_error
        return self.scores

    async def materialize(self, *args: object, **kwargs: object) -> object:
        self.materialized.append((args, kwargs))
        raise AssertionError("the sweep observer must never materialize a local adapter")

    async def train(self, *args: object, **kwargs: object) -> object:
        self.trained.append((args, kwargs))
        raise AssertionError("the sweep observer must never route through backend.train")


def stub_frame() -> SimpleNamespace:
    return SimpleNamespace(
        tails=["t0", "t1", "t2", "t3"],
        labels=np.array([True, False, True, False]),
        ids=["r0", "r1", "r2", "r3"],
    )


def perfect_scores() -> tuple[SimpleNamespace, ...]:
    # fire = 1 - P(NO_STEER); labels [T,F,T,F] with fire [0.9,0.1,0.8,0.2] separate cleanly -> AUC 1.0.
    return tuple(SimpleNamespace(logprob=math.log(p), weight=1.0) for p in (0.1, 0.9, 0.2, 0.8))


def stub_plan() -> SimpleNamespace:
    checkpoints = [
        SimpleNamespace(step=100, path="tinker://ckpt-a"),
        SimpleNamespace(step=200, path="tinker://ckpt-b"),
        SimpleNamespace(step=150, path="tinker://ckpt-c"),
    ]
    select = {"tinker://ckpt-a": 0.6, "tinker://ckpt-b": 0.9, "tinker://ckpt-c": 0.7}
    return SimpleNamespace(
        spec=SimpleNamespace(),
        policy=SimpleNamespace(),
        eval_rows=(),
        select=lambda checkpoint: select[checkpoint.path],
    ), checkpoints


class TestFitScoreComposition:
    def _run(self, backend: FakeBackend, plan: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, spend_cap_usd: float = 20.0) -> float:
        monkeypatch.setattr(sweep, "ADAPTER_STAGE_DIR", tmp_path / "staging")
        monkeypatch.setattr(sweep.sentinel, "sentinel_eval_row", lambda system, user, mlx_id: EvalRow(tokens=(0,), weights=(1.0,)))
        return sweep.observe_fit_score(
            backend, rebased("qwen3.5-9b"), BASE_MODELS["qwen3.5-9b"], stub_frame(), plan, spend_cap_usd=spend_cap_usd
        )

    def test_scores_selected_checkpoint_never_materializes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Runs the non-locally-servable qwen3.5-9b arm: fit+score succeeds where train/retrain/materialize refuse.
        plan, checkpoints = stub_plan()
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0)
        metric = self._run(backend, plan, tmp_path, monkeypatch)
        assert metric == 1.0
        assert [call.path for call in backend.score_calls] == ["tinker://ckpt-b"]
        assert backend.score_calls[0].base is BASE_MODELS["qwen3.5-9b"]
        assert backend.materialized == []
        assert backend.trained == []

    def test_score_budget_is_cap_minus_fit_cost(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan, checkpoints = stub_plan()
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0)
        self._run(backend, plan, tmp_path, monkeypatch, spend_cap_usd=20.0)
        assert backend.score_calls[0].max_usd == pytest.approx(7.0)

    def test_staging_removed_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan, checkpoints = stub_plan()
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0)
        self._run(backend, plan, tmp_path, monkeypatch)
        assert list((tmp_path / "staging").iterdir()) == []

    def test_staging_removed_on_scorer_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan, checkpoints = stub_plan()
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0, score_error=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            self._run(backend, plan, tmp_path, monkeypatch)
        assert list((tmp_path / "staging").iterdir()) == []


class TestSweepSpec:
    @pytest.mark.parametrize("arm", ARMS)
    def test_each_arm_has_a_pinned_spec(self, arm: str) -> None:
        spec = ExperimentSpec.load(EXPERIMENTS / f"watcher-base-sweep-{arm}.toml")
        assert spec.name == f"watcher-base-sweep-{arm}"
        assert spec.direction == "max"
        assert spec.mutable_paths == ("watcher_recipe.json",)
        assert spec.immutable_paths == (f"watcher-base-sweep-{arm}.toml",)
        command = spec.metric_command
        assert command[0] == "cc-steer" and "score-watcher" in command
        assert command[command.index("--arm") + 1] == arm
        assert float(command[command.index("--spend-cap-usd") + 1]) >= WatcherRecipe.default().spend_cap_usd

    @pytest.mark.parametrize("arm", ARMS)
    def test_metric_channel_matches_write_metric(self, arm: str) -> None:
        spec = ExperimentSpec.load(EXPERIMENTS / f"watcher-base-sweep-{arm}.toml")
        assert spec.metric_key == METRIC_KEY
        assert spec.metric_file == METRIC_FILE

    @pytest.mark.parametrize("arm", ARMS)
    def test_pinned_cap_covers_recipe_default_fit_plus_headroom(self, arm: str) -> None:
        # --spend-cap-usd is the only metric-spend guard; it must cover the default recipe's fit cap
        # ($15) with headroom for the Tinker-frame score, so the default recipe never self-refuses.
        spec = ExperimentSpec.load(EXPERIMENTS / f"watcher-base-sweep-{arm}.toml")
        command = spec.metric_command
        assert float(command[command.index("--spend-cap-usd") + 1]) > WatcherRecipe.default().spend_cap_usd

    @pytest.mark.parametrize("arm", ARMS)
    def test_budget_max_usd_is_a_driver_backstop_not_metric_spend(self, arm: str) -> None:
        # athome's [budget].max_usd meters the proposer's spend, not the metric_command's Tinker spend,
        # so it must not be sized as max_units * --spend-cap-usd (which would wrongly imply a metric bound).
        spec = ExperimentSpec.load(EXPERIMENTS / f"watcher-base-sweep-{arm}.toml")
        command = spec.metric_command
        per_unit = float(command[command.index("--spend-cap-usd") + 1])
        assert spec.budget.max_usd is not None
        assert spec.budget.max_usd < per_unit * spec.budget.max_units


STUB_SCORER = (
    "import json\n"
    "from pathlib import Path\n"
    "from cc_steer.retrain import sweep\n"
    "from cc_steer.retrain.watcher import WatcherRecipe\n"
    "knob = json.loads(Path('watcher_recipe.json').read_text())['knob']\n"
    "sweep.score_watcher(\n"
    "    WatcherRecipe.default(), arm='qwen3-8b', spend_cap_usd=18.0,\n"
    "    train_and_score=lambda recipe, *, arm, spend_cap_usd, dataset_dir, eval_root: float(knob),\n"
    ")\n"
)


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
    (repo / "watcher-base-sweep-qwen3-8b.toml").write_text("# immutable scoring boundary\n")
    (repo / "score.py").write_text(STUB_SCORER)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "sweep@test")
    git(repo, "config", "user.name", "sweep")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


class TestStubDriverDryRun:
    def test_full_loop_keeps_the_strict_improvements(self, tmp_path: Path) -> None:
        # Spend-free dry run of the greedy loop: the metric_command drives the real sweep.score_watcher
        # observer with an injected stub train_and_score, so the loop exercises cc-steer's sweep code.
        repo = init_repo(tmp_path)
        spec = ExperimentSpec(
            name="watcher-base-sweep-dryrun",
            metric_command=(sys.executable, "score.py"),
            metric_key="metric",
            direction="max",
            mutable_paths=("watcher_recipe.json",),
            immutable_paths=("watcher-base-sweep-qwen3-8b.toml",),
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
            immutable_paths=("watcher-base-sweep-qwen3-8b.toml",),
            budget=Budget(max_units=1, hard_kill_s=120.0),
        )
        proposals = iter([StubProposal({"watcher-base-sweep-qwen3-8b.toml": "# tampered\n"})])
        result = anyio.run(lambda: loop.run(spec, driver=StubDriver(proposals), repo=repo))
        assert result.kept == 0


def test_cli_score_watcher_invokes_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from cc_steer import cli

    seen: dict[str, object] = {}

    def fake(recipe: WatcherRecipe, *, arm: str, spend_cap_usd: float) -> float:
        seen.update(arm=arm, spend_cap_usd=spend_cap_usd)
        return 0.9123

    monkeypatch.setattr(sweep, "score_watcher", fake)
    result = CliRunner().invoke(cli.main, ["score-watcher", "--arm", "qwen3-8b", "--spend-cap-usd", "18.0"])
    assert result.exit_code == 0, result.output
    assert "0.9123" in result.output
    assert seen == {"arm": "qwen3-8b", "spend_cap_usd": 18.0}
