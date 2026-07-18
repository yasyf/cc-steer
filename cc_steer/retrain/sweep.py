"""Base-model sweep machinery: score a recipe as a pure observer, filter arms by what serves locally.

The multi-arm sweep is deferred until athome ships a second locally-servable base — today
:data:`athome.train.BASE_MODELS` marks only ``qwen3-8b`` ``serves_locally``, so
:func:`servable_arms` yields a single arm and there is nothing to sweep. This module is the
arm-parameterized instrument that runs the moment a second arm becomes servable:
:func:`servable_arms` enumerates the servable bases, :func:`base_for_recipe` resolves the arm a
recipe trains on, and :func:`score_watcher` trains and scores one recipe through the served-MLX
instrument and reports the served sentinel AUC via :func:`athome.train.write_metric` with zero
registry, journal, or promotion side effects — the sweep loop owns keep/discard, this stays a
measurement. ``experiments/watcher-base-sweep.toml`` is the per-arm
:class:`~athome.research.spec.ExperimentSpec` whose ``metric_command`` invokes ``cc-steer
score-watcher``; its ``metric_key`` is ``"metric"`` to match :func:`~athome.train.write_metric`'s
channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import anyio
import numpy as np
from athome.train import BASE_MODELS, write_metric

from cc_steer.retrain import evalset, promotion
from cc_steer.retrain.watcher import WatcherRecipe, build_train_plan, load_key

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from athome.train import BaseModelSpec


class UnservableArm(RuntimeError):
    """A recipe's base is not one of the locally-servable :data:`athome.train.BASE_MODELS` arms."""


class TrainAndScore(Protocol):
    def __call__(self, recipe: WatcherRecipe, *, dataset_dir: Path | None, eval_root: Path | None) -> float: ...


def servable_arms(models: Mapping[str, BaseModelSpec] = BASE_MODELS) -> tuple[str, ...]:
    """The base-model arms the sweep can train-serve-score: the ``BASE_MODELS`` keys athome serves locally.

    Returns the keys in declaration order whose spec sets ``serves_locally``. Today that is exactly
    ``("qwen3-8b",)``; the sweep gains an arm the moment athome flips a second base's
    ``serves_locally`` flag, with no change here.
    """
    return tuple(name for name, spec in models.items() if spec.serves_locally)


def base_for_recipe(recipe: WatcherRecipe, models: Mapping[str, BaseModelSpec] = BASE_MODELS) -> BaseModelSpec:
    """The servable ``BASE_MODELS`` arm a recipe trains on, matched by its Tinker and MLX ids.

    Unlike :func:`cc_steer.retrain.watcher._base_for`, which pins the production ``qwen3-8b`` arm the
    promotion path serves, this resolves any arm the sweep can run — every base athome marks
    ``serves_locally``. A recipe whose base is unknown or not locally servable is refused, so a
    non-servable arm never reaches the network.

    Raises:
        UnservableArm: no locally-servable arm matches the recipe's base ids.
    """
    for spec in models.values():
        if spec.serves_locally and recipe.tinker_model == spec.tinker and recipe.mlx_id == spec.mlx:
            return spec
    raise UnservableArm(
        f"recipe base {recipe.tinker_model}/{recipe.mlx_id} is not a locally-servable arm; "
        f"servable arms: {[spec.tinker for spec in models.values() if spec.serves_locally]}"
    )


def paid_train_and_score(recipe: WatcherRecipe, *, dataset_dir: Path | None, eval_root: Path | None) -> float:
    """Train the recipe on Tinker, serve the best checkpoint through MLX, and return its sentinel AUC.

    The real, paid default for :func:`score_watcher`: it trains the recipe's LoRA under the recipe's
    own spend cap, materializes the best checkpoint into the local MLX serving stack, and scores the
    frozen eval frame through it — the exact instrument the promotion pass gates on, minus the gate.
    Never called by the test suite (:func:`score_watcher` takes an injected scorer); the orchestrator
    runs it behind its own spend gate.
    """
    import tempfile
    from pathlib import Path

    from athome.progress import RunSink
    from athome.train import retrain as athome_retrain
    from athome.train.gate import GateVerdict
    from athome.train.tinker import TinkerBackend

    from cc_steer.retrain.watcher import ADAPTER_STAGE_DIR, RUN_JOURNAL_NAME

    frame = evalset.EvalFrame.load(root=eval_root)
    plan = build_train_plan(recipe, frame, base_for_recipe(recipe), dataset_dir=dataset_dir)
    load_key()
    backend = TinkerBackend.from_settings()
    ADAPTER_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="watcher-score-", dir=ADAPTER_STAGE_DIR))
    sink = RunSink.open(work_dir / RUN_JOURNAL_NAME)

    async def run() -> float:
        outcome = await athome_retrain(
            backend,
            plan.spec,
            checkpoints=plan.policy,
            eval_rows=plan.eval_rows,
            select=plan.select,
            artifact_scorer=plan.artifact_scorer,
            gate=lambda served: GateVerdict(promote=True, reason="score-only observer", stats={}),
            work_dir=work_dir,
            sink=sink,
        )
        served = np.array([outcome.served[row_id] for row_id in frame.ids], dtype=np.float64)
        return promotion.sentinel_auc(frame.labels, 1.0 - served)

    return anyio.run(run)


def score_watcher(
    recipe: WatcherRecipe,
    *,
    train_and_score: TrainAndScore = paid_train_and_score,
    dataset_dir: Path | None = None,
    eval_root: Path | None = None,
) -> float:
    """Score one candidate recipe as a pure observer and report it on athome's metric channel.

    Trains and scores ``recipe`` through the served-MLX instrument, then writes the served sentinel
    AUC to ``.athome-metric.json`` in the working directory via :func:`athome.train.write_metric` —
    the structured channel the sweep's :class:`~athome.research.spec.ExperimentSpec` reads. Unlike
    :func:`~cc_steer.retrain.watcher.retrain_watcher` it never registers, promotes, journals, or
    writes incumbent probs: the sweep loop owns keep/discard, so this is measurement only. The paid
    ``train_and_score`` is injectable so tests exercise the wiring without spend.

    Returns:
        The served sentinel AUC that was written to the metric file.
    """
    metric = train_and_score(recipe, dataset_dir=dataset_dir, eval_root=eval_root)
    anyio.run(write_metric, metric)
    return metric
