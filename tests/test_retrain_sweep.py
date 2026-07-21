from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import anyio
import numpy as np
import pytest
from athome.research import loop
from athome.research.driver import StubDriver, StubProposal
from athome.research.spec import Budget, ExperimentSpec
from athome.train import BASE_MODELS, METRIC_FILE, METRIC_KEY, SpendGuard
from athome.train.spec import EvalRow, Hyperparams, Rows, TinkerSettings, TrainSpec

from cc_steer import instrument
from cc_steer.retrain import sweep
from cc_steer.retrain.watcher import WatcherRecipe

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
ARMS = ("qwen3-8b", "qwen3.5-4b", "qwen3.5-9b")


class RecipeTokenLoad(NamedTuple):
    train_tokens: int
    eval_datum_tokens: int
    snapshots: int
    score_prefill_tokens: int
    score_rows: int


# The measured token load of the untouched default recipe on each arm — the exact schedule
# TinkerBackend.fit projects and the full sentinel frame score bills, captured from athome 0.7.1's
# real render + assemble path over the frozen frame. It sizes the pinned --spend-cap-usd: the load is
# the same on every arm (the recipe is base-swap-cloned, nothing else), so the dollar projection scales
# purely with the arm's per-Mtok price. It reproduces fit=$26.8499 / score=$0.5176 for the 9B at the
# real price sheet — the numbers that make the old $24 cap (and the recipe's $15 fit cap) unusable.
DEFAULT_RECIPE_TOKEN_LOAD = {
    "qwen3-8b": RecipeTokenLoad(19_181_870, 382_164, 4, 1_146_257, 628),
    "qwen3.5-4b": RecipeTokenLoad(19_669_672, 391_602, 4, 1_174_401, 628),
    "qwen3.5-9b": RecipeTokenLoad(19_669_672, 391_602, 4, 1_174_401, 628),
}


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
            train_and_score=stub_scorer(sweep.ArmScore(metric=0.71, probs={"r0": 0.5}, frame_digest="d")),
        )
        assert result == 0.71


ARM_SCORE = sweep.ArmScore(metric=0.873, probs={"r0": 0.1, "r1": 0.9}, frame_digest="digest-a")


def stub_scorer(score: sweep.ArmScore = ARM_SCORE) -> sweep.TrainAndScore:
    return lambda recipe, *, arm, spend_cap_usd, dataset_dir, eval_root: score


class TestScoreReport:
    def test_metric_file_and_report_carry_the_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        seen: dict[str, object] = {}

        def fake(
            recipe: WatcherRecipe, *, arm: object, spend_cap_usd: float, dataset_dir: object, eval_root: object
        ) -> sweep.ArmScore:
            seen.update(recipe=recipe, arm=arm, spend_cap_usd=spend_cap_usd, dataset_dir=dataset_dir, eval_root=eval_root)
            return ARM_SCORE

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
            "dataset_digest": "digest-a",
            "probs": {"r0": 0.1, "r1": 0.9},
        }

    def test_report_carries_non_local_arm_posture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A non-locally-servable arm is a first-class scored arm; serves_locally rides the report as metadata.
        monkeypatch.chdir(tmp_path)
        assert BASE_MODELS["qwen3.5-9b"].serves_locally is False
        sweep.score_watcher(
            rebased("qwen3.5-9b"),
            arm="qwen3.5-9b",
            spend_cap_usd=30.0,
            train_and_score=stub_scorer(),
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
            train_and_score=stub_scorer(),
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
        self.fit_spec: object = None
        self.fit_budget: SpendGuard | None = None
        self.score_budget: SpendGuard | None = None

    async def fit(self, spec: object, *, sink: object, budget: SpendGuard, checkpoints: object, eval_rows: object) -> SimpleNamespace:
        # Mirror athome's fit: the run drains and its actual cost is recorded against the shared envelope,
        # so a score threaded the same budget reserves against the drawn-down remainder.
        self.fit_spec = spec
        self.fit_budget = budget
        await budget.record(0.0, self.train_cost_usd)
        return SimpleNamespace(checkpoints=self.checkpoints, train_cost_usd=self.train_cost_usd)

    async def score(self, path: str, rows: object, *, base: object, budget: SpendGuard) -> tuple[SimpleNamespace, ...]:
        self.score_budget = budget
        self.score_calls.append(SimpleNamespace(path=path, rows=rows, base=base, budget=budget))
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
        digest="stub-digest",
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
    # A real (frozen) TrainSpec so observe_fit_score's dataclasses.replace(..., max_usd=harness_cap)
    # is exercised as it runs in production; max_usd=15.0 is the recipe's own fit cap the sweep overrides.
    spec = TrainSpec(
        name="watcher-base-sweep-stub",
        base=BASE_MODELS["qwen3.5-9b"],
        dataset=Rows(examples=()),
        hyperparams=Hyperparams(steps=3, batch_size=1),
        max_usd=15.0,
    )
    return SimpleNamespace(
        spec=spec,
        policy=SimpleNamespace(),
        eval_rows=(),
        select=lambda checkpoint: select[checkpoint.path],
    ), checkpoints


class TestFitScoreComposition:
    def _run(self, backend: FakeBackend, plan: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, spend_cap_usd: float = 20.0) -> sweep.ArmScore:
        monkeypatch.setattr(sweep, "ADAPTER_STAGE_DIR", tmp_path / "staging")
        monkeypatch.setattr(sweep.sentinel, "sentinel_eval_row", lambda system, user, mlx_id: EvalRow(tokens=(0,), weights=(1.0,)))
        return sweep.observe_fit_score(
            backend, rebased("qwen3.5-9b"), stub_frame(), plan, spend_cap_usd=spend_cap_usd
        )

    def test_scores_selected_checkpoint_never_materializes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Runs the non-locally-servable qwen3.5-9b arm: fit+score succeeds where train/retrain/materialize refuse.
        plan, checkpoints = stub_plan()
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0)
        score = self._run(backend, plan, tmp_path, monkeypatch)
        assert score.metric == 1.0
        # The per-row P(NO_STEER) rides the score keyed by frame row id, ready for a paired comparison.
        assert score.probs == pytest.approx({"r0": 0.1, "r1": 0.9, "r2": 0.2, "r3": 0.8})
        assert score.frame_digest == "stub-digest"
        assert [call.path for call in backend.score_calls] == ["tinker://ckpt-b"]
        assert backend.score_calls[0].base is BASE_MODELS["qwen3.5-9b"]
        assert backend.materialized == []
        assert backend.trained == []

    def test_score_budget_is_cap_minus_fit_cost(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # One SpendGuard spans fit and score: the fit's recorded $13 actual draws down the $20 cap, so
        # the shared envelope the score reserves against has exactly $7 of headroom left.
        plan, checkpoints = stub_plan()
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0)
        self._run(backend, plan, tmp_path, monkeypatch, spend_cap_usd=20.0)
        assert backend.score_budget is backend.fit_budget
        assert backend.score_budget.max_usd - backend.score_budget.spent == pytest.approx(7.0)

    def test_fit_is_bound_by_harness_cap_not_recipe_cap(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The pinned cap must bound fit + score together, so the fit runs under the harness cap, not the
        # recipe's own spend_cap_usd. Binding the fit to the $15 recipe cap starves the pricier 9B arm,
        # whose identical schedule projects ~$26.85 (> $15) and can never establish a baseline.
        plan, checkpoints = stub_plan()
        assert plan.spec.max_usd == 15.0  # the recipe's own (spec-declared) cap; athome's fit no longer reads it
        backend = FakeBackend(checkpoints, perfect_scores(), train_cost_usd=13.0)
        self._run(backend, plan, tmp_path, monkeypatch, spend_cap_usd=24.0)
        assert backend.fit_budget.max_usd == 24.0  # the fit's envelope is the harness cap, not the recipe's $15

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
    def test_pinned_cap_covers_default_recipe_fit_plus_score_projection(self, arm: str) -> None:
        # The real guard finding #2's weaker "cap > $15" check missed: project the default recipe's
        # actual fit schedule and full-frame score cost at athome's real price sheet, and assert the
        # pinned --spend-cap-usd covers both. The token load is the same identical schedule on every
        # arm, so the projection scales purely with the arm's per-Mtok price — the 9B bills 3.325x the
        # 8B train rate, which a cap sized by arm rather than by price silently under-provisions.
        price = TinkerSettings.model_fields["price_per_mtok"].default_factory()[BASE_MODELS[arm].tinker]
        load = DEFAULT_RECIPE_TOKEN_LOAD[arm]
        fit = (load.train_tokens * price.train + load.eval_datum_tokens * load.snapshots * price.prefill) / 1e6
        score = (load.score_prefill_tokens * price.prefill + load.score_rows * price.sample) / 1e6
        spec = ExperimentSpec.load(EXPERIMENTS / f"watcher-base-sweep-{arm}.toml")
        command = spec.metric_command
        cap = float(command[command.index("--spend-cap-usd") + 1])
        # observe_fit_score binds the fit to this cap (not the recipe's $15) and gives the score the
        # remainder, so both the fit alone and fit + score must fit under the pinned cap.
        assert fit <= cap, f"{arm}: default-recipe fit projects ${fit:.4f}, over the pinned cap ${cap:.4f}"
        assert fit + score <= cap, f"{arm}: fit+score projects ${fit + score:.4f}, over the pinned cap ${cap:.4f}"

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
    "score = sweep.ArmScore(metric=float(knob), probs={'r0': 0.5}, frame_digest='stub')\n"
    "sweep.score_watcher(\n"
    "    WatcherRecipe.default(), arm='qwen3-8b', spend_cap_usd=18.0,\n"
    "    train_and_score=lambda recipe, *, arm, spend_cap_usd, dataset_dir, eval_root: score,\n"
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


COMPARE_FRAME = SimpleNamespace(
    ids=[f"r{i}" for i in range(8)],
    labels=np.array([True, True, True, False, False, False, True, False]),
    digest="frame-d",
)


def write_report(path: Path, *, metric: float, probs: dict[str, float] | None, digest: str = "frame-d") -> Path:
    payload: dict[str, object] = {"metric": metric, "arm": "qwen3-8b", "serves_locally": True}
    if probs is not None:
        payload |= {"dataset_digest": digest, "probs": probs}
    path.write_text(json.dumps(payload))
    return path


class TestCompareScoreReports:
    def test_paired_verdict_over_both_persisted_probs_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sweep.evalset.EvalFrame, "load", lambda *, root=None: COMPARE_FRAME)
        probs_a = {rid: p for rid, p in zip(COMPARE_FRAME.ids, (0.1, 0.2, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9))}
        probs_b = {rid: p for rid, p in zip(COMPARE_FRAME.ids, (0.2, 0.1, 0.6, 0.4, 0.7, 0.5, 0.5, 0.8))}
        report_a = write_report(tmp_path / "a.json", metric=0.9, probs=probs_a)
        report_b = write_report(tmp_path / "b.json", metric=0.8, probs=probs_b)
        comparison = sweep.compare_score_reports(report_a, report_b)
        fire_a = 1.0 - np.array([probs_a[rid] for rid in COMPARE_FRAME.ids])
        fire_b = 1.0 - np.array([probs_b[rid] for rid in COMPARE_FRAME.ids])
        assert comparison == instrument.paired_verdict(
            instrument.paired_delong(COMPARE_FRAME.labels, fire_a, fire_b)
        )
        assert comparison.paired is not None  # measured rho, not an assumed one

    def test_frame_drift_fails_loud(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sweep.evalset.EvalFrame, "load", lambda *, root=None: COMPARE_FRAME)
        probs = dict.fromkeys(COMPARE_FRAME.ids, 0.5)
        report_a = write_report(tmp_path / "a.json", metric=0.9, probs=probs)
        report_b = write_report(tmp_path / "b.json", metric=0.8, probs=probs, digest="stale-d")
        with pytest.raises(sweep.FrameMismatch, match="stale-d"):
            sweep.compare_score_reports(report_a, report_b)

    @pytest.mark.parametrize(
        ("metric_b", "expected_actionable", "expected_verdict"),
        [
            (0.86, False, "within noise floor (MDE 0.0453)"),
            (0.96, True, "actionable gain (delta +0.0600, unpaired MDE 0.0453)"),
        ],
        ids=["sub-mde-delta-is-noise", "large-delta-actionable"],
    )
    def test_missing_counterpart_probs_falls_back_to_unpaired_card_floor(
        self, tmp_path: Path, metric_b: float, expected_actionable: bool, expected_verdict: str
    ) -> None:
        card = tmp_path / "card.json"
        card.write_text(json.dumps({"mde": 0.0453, "stopping": {"mde": 0.0453}, "mde_paired": {}}))
        report_a = write_report(tmp_path / "a.json", metric=0.90, probs=None)
        report_b = write_report(
            tmp_path / "b.json", metric=metric_b, probs=dict.fromkeys(COMPARE_FRAME.ids, 0.5)
        )
        comparison = sweep.compare_score_reports(report_a, report_b, card_path=card)
        assert comparison.paired is None
        assert comparison.actionable is expected_actionable
        assert comparison.verdict == expected_verdict


def test_cli_compare_arms_prints_the_card_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from cc_steer import cli

    report_a = write_report(tmp_path / "a.json", metric=0.90, probs=None)
    report_b = write_report(tmp_path / "b.json", metric=0.88, probs=None)
    seen: dict[str, object] = {}

    def fake(a: Path, b: Path, **kwargs: object) -> instrument.Comparison:
        seen.update(a=a, b=b)
        return instrument.unpaired_verdict(0.90, 0.88, frame_mde=0.0453)

    monkeypatch.setattr(sweep, "compare_score_reports", fake)
    result = CliRunner().invoke(cli.main, ["compare-arms", str(report_a), str(report_b)])
    assert result.exit_code == 0, result.output
    assert result.output == "AUC 0.9000 -> 0.8800: within noise floor (MDE 0.0453)\n"
    assert seen == {"a": report_a, "b": report_b}
