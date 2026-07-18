"""Base-model sweep machinery: rank every base under one uniform hosted Tinker instrument.

Every :data:`athome.train.BASE_MODELS` entry is a sweep arm — servability is not an admission
filter. :func:`sweep_arms` enumerates all of them, :func:`base_for_recipe` resolves the arm a recipe
trains on (refusing only a genuinely unknown base), and :func:`score_watcher` trains and scores one
recipe as a pure observer, reporting the metric on :func:`athome.train.write_metric`'s channel with
zero registry, retrain-journal, or promotion side effects — the sweep loop owns keep/discard, this
stays a measurement.

The instrument is uniform for every arm: the sentinel AUC of the selected checkpoint scored in the
**Tinker frame** via :meth:`athome.train.tinker.TinkerBackend.score` over the frozen sentinel frame.
The scoring path composes hosted ``fit`` + hosted ``score`` directly and never routes through
athome's ``train``/``retrain``/``materialize``, which refuse a non-locally-servable base up front;
composing the two primitives keeps the measurement identical across qwen3-8b, qwen3.5-4b, and
qwen3.5-9b, whatever their deployment posture. Checkpoint selection is caller-side math over the
:class:`~athome.train.TrainReport`'s per-checkpoint eval scores, the same ``select`` the promotion
pass uses (:func:`cc_steer.retrain.watcher.build_train_plan`).

``serves_locally`` is reporting metadata, never a gate: it rides the per-arm score report so a
result carries each arm's deployment posture. The served-MLX gate stays the promotion instrument for
the deployed watcher — a locally-servable winner is confirmed through the normal retrain pipeline
(:func:`cc_steer.retrain.watcher.retrain_watcher`) before any promotion, and a non-local winner
implies a hosted serving path, a downstream deployment decision outside this module.

``experiments/watcher-base-sweep-<arm>.toml`` is the per-arm
:class:`~athome.research.spec.ExperimentSpec`; each pins its arm and the harness spend cap into the
``metric_command`` it invokes, and its ``metric_key`` is ``"metric"`` to match
:func:`~athome.train.write_metric`'s channel.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anyio
import numpy as np
from athome.progress import RunSink
from athome.train import BASE_MODELS, write_metric
from athome.train.tinker import TinkerBackend

from cc_steer.retrain import evalset, promotion, sentinel
from cc_steer.retrain.watcher import (
    ADAPTER_STAGE_DIR,
    RUN_JOURNAL_NAME,
    WatcherRecipe,
    build_train_plan,
    load_key,
)
from cc_steer.watcher.cascade import DRAFT_SYSTEM

if TYPE_CHECKING:
    from collections.abc import Mapping

    from athome.train import BaseModelSpec

    from cc_steer.retrain.watcher import TrainPlan

SCORE_REPORT_FILE = "score_report.json"


class UnknownArm(RuntimeError):
    """A recipe's base ids, or a spec's ``--arm`` key, match no :data:`athome.train.BASE_MODELS` arm."""


class ArmMismatch(RuntimeError):
    """A recipe's base ids resolve to a different arm than the one its spec pinned via ``--arm``."""


class SpendCapExceeded(RuntimeError):
    """The spend cap is refused: a non-finite harness cap, or a recipe cap above the pinned one — never clamped."""


class TrainAndScore(Protocol):
    def __call__(
        self,
        recipe: WatcherRecipe,
        *,
        arm: BaseModelSpec,
        spend_cap_usd: float,
        dataset_dir: Path | None,
        eval_root: Path | None,
    ) -> float: ...


def sweep_arms(models: Mapping[str, BaseModelSpec] = BASE_MODELS) -> tuple[str, ...]:
    """Every base-model arm the sweep ranks: all :data:`athome.train.BASE_MODELS` keys, in declaration order.

    Servability is not an admission filter — a non-locally-servable base is a first-class arm, scored
    under the same uniform hosted Tinker instrument as any other, with its ``serves_locally`` posture
    carried as reporting metadata.
    """
    return tuple(models)


def base_for_recipe(recipe: WatcherRecipe, models: Mapping[str, BaseModelSpec] = BASE_MODELS) -> BaseModelSpec:
    """The :data:`athome.train.BASE_MODELS` arm a recipe trains on, matched by its Tinker and MLX ids.

    Resolves any arm the sweep runs, servable or not; only a base matching no arm at all is refused,
    so a genuinely unknown base never reaches the network.

    Raises:
        UnknownArm: no arm matches the recipe's base ids.
    """
    for spec in models.values():
        if recipe.tinker_model == spec.tinker and recipe.mlx_id == spec.mlx:
            return spec
    raise UnknownArm(
        f"recipe base {recipe.tinker_model}/{recipe.mlx_id} matches no BASE_MODELS arm; "
        f"known arms: {[f'{spec.tinker}/{spec.mlx}' for spec in models.values()]}"
    )


def observe_fit_score(
    backend: TinkerBackend,
    recipe: WatcherRecipe,
    arm: BaseModelSpec,
    frame: evalset.EvalFrame,
    plan: TrainPlan,
    *,
    spend_cap_usd: float,
) -> float:
    """Fit the recipe, select the strongest checkpoint, score it in the Tinker frame, return its sentinel AUC.

    The uniform instrument, composed from the two hosted primitives so it never touches
    ``train``/``retrain``/``materialize``: ``fit`` runs the training schedule and scores every
    checkpoint's val eval on the turnstile; ``plan.select`` (the promotion pass's own checkpoint math)
    picks the argmax over :attr:`~athome.train.TrainReport.checkpoints`; ``score`` evaluates that one
    checkpoint over the full sentinel frame; the fire score ``1 - P(NO_STEER)`` yields the AUC. The
    only on-disk footprint is the run journal in a throwaway staging dir, removed on success and on
    failure. ``score``'s budget is the harness cap minus the fit's metered spend, so the pinned cap
    bounds fit and score together.
    """
    ADAPTER_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="watcher-sweep-", dir=ADAPTER_STAGE_DIR))

    async def run() -> float:
        frame_rows = tuple(sentinel.sentinel_eval_row(DRAFT_SYSTEM, tail, recipe.mlx_id) for tail in frame.tails)
        report = await backend.fit(
            plan.spec, sink=RunSink.open(work_dir / RUN_JOURNAL_NAME), checkpoints=plan.policy, eval_rows=plan.eval_rows
        )
        best = max(report.checkpoints, key=plan.select)
        scored = await backend.score(best.path, frame_rows, base=arm, max_usd=spend_cap_usd - report.train_cost_usd)
        fire = 1.0 - np.exp(np.array([sequence.logprob for sequence in scored], dtype=np.float64))
        return promotion.sentinel_auc(frame.labels, fire)

    try:
        return anyio.run(run)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def paid_train_and_score(
    recipe: WatcherRecipe,
    *,
    arm: BaseModelSpec,
    spend_cap_usd: float,
    dataset_dir: Path | None,
    eval_root: Path | None,
) -> float:
    """The real, paid default for :func:`score_watcher`: hosted fit + hosted Tinker-frame score, no local serving.

    Loads the frozen frame, builds the recipe's :class:`~cc_steer.retrain.watcher.TrainPlan`, and runs
    :func:`observe_fit_score` under the harness-pinned cap. Never called by the test suite
    (:func:`score_watcher` takes an injected scorer); the orchestrator runs it behind its own spend
    gate.
    """
    frame = evalset.EvalFrame.load(root=eval_root)
    plan = build_train_plan(recipe, frame, arm, dataset_dir=dataset_dir)
    load_key()
    return observe_fit_score(TinkerBackend.from_settings(), recipe, arm, frame, plan, spend_cap_usd=spend_cap_usd)


def write_score_report(arm: str, arm_spec: BaseModelSpec, metric: float) -> None:
    """Write the per-arm score report beside the metric file, carrying the arm's deployment posture."""
    Path(SCORE_REPORT_FILE).write_text(
        json.dumps(
            {
                "metric": metric,
                "arm": arm,
                "tinker_model": arm_spec.tinker,
                "mlx_id": arm_spec.mlx,
                "serves_locally": arm_spec.serves_locally,
            }
        )
        + "\n"
    )


def score_watcher(
    recipe: WatcherRecipe,
    *,
    arm: str,
    spend_cap_usd: float,
    train_and_score: TrainAndScore = paid_train_and_score,
    dataset_dir: Path | None = None,
    eval_root: Path | None = None,
) -> float:
    """Score one candidate recipe against its pinned arm as a pure observer, on athome's metric channel.

    Refuses before any spend if the recipe's base ids do not resolve to the harness-pinned ``arm``
    (:class:`ArmMismatch`, or :class:`UnknownArm` for an unknown base or ``arm`` key) or if the
    recipe's ``spend_cap_usd`` exceeds the pinned ``spend_cap_usd`` (:class:`SpendCapExceeded`, never
    clamped). Otherwise trains and scores through the uniform Tinker-frame instrument, writes the
    scalar to ``.athome-metric.json`` via :func:`athome.train.write_metric`, and writes the per-arm
    :data:`SCORE_REPORT_FILE` carrying ``serves_locally`` as reporting metadata. It never registers,
    promotes, or writes the retrain journal: the sweep loop owns keep/discard, so this is measurement
    only. The paid ``train_and_score`` is injectable so tests exercise the wiring without spend.

    Returns:
        The sentinel AUC that was written to the metric file.
    """
    if not (math.isfinite(spend_cap_usd) and spend_cap_usd > 0.0):
        raise SpendCapExceeded(
            f"harness-pinned --spend-cap-usd {spend_cap_usd!r} for arm {arm!r} must be finite and > 0; a NaN or "
            "infinite cap passes every comparison and would silently disable athome's SpendGuard on fit and score"
        )
    if arm not in BASE_MODELS:
        raise UnknownArm(f"--arm {arm!r} is not a BASE_MODELS key; known arms: {list(BASE_MODELS)}")
    arm_spec = BASE_MODELS[arm]
    if base_for_recipe(recipe) is not arm_spec:
        raise ArmMismatch(
            f"recipe base {recipe.tinker_model}/{recipe.mlx_id} matches a BASE_MODELS arm other than the pinned "
            f"--arm {arm!r} ({arm_spec.tinker}/{arm_spec.mlx}); the spec pins one arm and the recipe must not drift it"
        )
    if recipe.spend_cap_usd > spend_cap_usd:
        raise SpendCapExceeded(
            f"recipe spend_cap_usd {recipe.spend_cap_usd} exceeds the harness-pinned cap {spend_cap_usd} for arm "
            f"{arm!r}; the sweep refuses rather than clamp — lower the recipe cap or raise the spec's --spend-cap-usd"
        )
    metric = train_and_score(
        recipe, arm=arm_spec, spend_cap_usd=spend_cap_usd, dataset_dir=dataset_dir, eval_root=eval_root
    )
    anyio.run(write_metric, metric)
    write_score_report(arm, arm_spec, metric)
    return metric
