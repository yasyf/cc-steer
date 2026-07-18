"""The watcher-component retrain lane: train a LoRA on athome, gate it on what we serve, promote it.

The full weekly pass for the generative watcher. It curates the training pool
(:mod:`cc_steer.retrain.data`), hands the recipe to :func:`athome.train.retrain` as a
:class:`~athome.train.spec.TrainSpec` plus three domain callables, and acts on what comes back.
athome trains the E8-winning recipe on Tinker, scores every checkpoint's sentinel eval on the
turnstile, and materializes the best checkpoint into a local mlx-lm adapter; cc-steer scores the
frozen eval frame *through that served artifact* — the 4-bit MLX base plus the converted adapter,
the exact thing production loads — and gates on those served probs.

Every promotion metric (the fresh-epoch AUC floor, the corrected paired gate against the
incumbent via :func:`~cc_steer.retrain.promotion.corrected_gate`, the served threshold) reads the
local probs, so the gate describes the model we actually run — not the full-precision Tinker frame
it trained against, whose 4-bit quantization shifts mid-confidence predictions. Every outcome
journals one line. A projected overspend or an under-filled pool journals a reject and returns
having spent nothing. A gate reject pays one materialize — the served artifact the gate scores
through — but never a register. A serving-drift diagnostic samples the frame Tinker-vs-served and
persists per-row diffs beside the run's artifacts for every outcome that reaches a materialized
candidate (reject included); pre-artifact rejects like the spend cap produce none. It never blocks,
not even on its own failure: it runs behind a boundary that records a mishap and leaves the gate
outcome untouched, and under eval-what-you-serve a conversion bug surfaces directly as a served AUC
at chance, which the floor rejects. :class:`WatcherRecipe` validates every knob at parse so a
degenerate value never reaches the network. Tinker/mlx stay behind athome and
:mod:`cc_steer.watcher.drafter_mlx`; this module imports neither.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import anyio
import numpy as np
from athome.errors import AthomeError
from athome.progress import RunSink
from athome.train import (
    BASE_MODELS,
    Adapter,
    BaseModelSpec,
    CheckpointPolicy,
    Hyperparams,
    InsufficientData,
    LoraSpec,
    RetrainOutcome,
    Rows,
    SavedCheckpoint,
    SftExample,
    SpendExceeded,
    SpendGuard,
    TinkerBackend,
    TrainSpec,
)
from athome.train import retrain as athome_retrain
from athome.train.gate import GateVerdict

from cc_steer import launchd, registry
from cc_steer.retrain import data, evalset, judged, promotion, sentinel
from cc_steer.watcher import drafter_mlx
from cc_steer.watcher.cascade import DRAFT_SYSTEM

if TYPE_CHECKING:
    from collections.abc import Callable

    from athome.train import EvalRow

    from cc_steer.retrain.promotion import GateResult

WATCHER_COMPONENT = drafter_mlx.COMPONENT
KEEP_VERSIONS = 3
ADAPTER_STAGE_DIR: Path = Path.home() / ".cc-steer" / "adapters" / "staging"
DIAGNOSTIC_NAME = "serving_diagnostic.json"
RUN_JOURNAL_NAME = "progress.jsonl"
TINKER_ENV: Path = Path.home() / ".cc-steer" / "tinker.env"


class WatcherRetrainError(RuntimeError):
    """The watcher retrain cannot proceed: no incumbent, unseeded incumbent probs, or an unknown base."""


class FreshEpochError(WatcherRetrainError):
    """``--fresh-epoch`` misuse: the frozen frame already has scored version probs, so the one-shot cutover is over."""


@dataclass(frozen=True, slots=True)
class WatcherRecipe:
    """Every knob of one watcher LoRA retrain, validated at parse so no degenerate value trains.

    The packaged default (``cc_steer/assets/watcher_recipe.json``, via :meth:`default`) is the
    E8-winner recipe. An override JSON must carry every field (missing or extra keys crash) and
    clears the same validation bar.

    Attributes:
        tinker_model: The Tinker base model id to train the LoRA over.
        mlx_id: The local 4-bit MLX id the converted adapter serves against.
        rank: The LoRA rank.
        learning_rate: The AdamW learning rate.
        batch_size: The datums per optimizer step.
        epochs: The passes over the pool; steps derive from it.
        checkpoint_fracs: Fractions of the run to checkpoint and score at.
        max_tokens: Datums longer than this are dropped before batching.
        render_version: The prompt-rendering contract stamped into the registry metadata.
        val_n: The target size of the carved val slice that ranks checkpoints.
        oversample_corrective: The factor corrective positives are oversampled to.
        budget_fires_per_100: The alert budget the served threshold is fitted at.
        spend_cap_usd: The hard Tinker spend cap; a projected overspend never launches.
        diagnostic_rows: The label-stratified rows the serving-drift diagnostic samples Tinker-vs-served.
        diagnostic_tolerance: The absolute nosteer-prob gap above which a diagnostic row counts
            as drifted (never blocks).
        seed: The seed threaded through every deterministic step.
    """

    tinker_model: str
    mlx_id: str
    rank: int
    learning_rate: float
    batch_size: int
    epochs: int
    checkpoint_fracs: tuple[float, ...]
    max_tokens: int
    render_version: int
    val_n: int
    oversample_corrective: float
    budget_fires_per_100: float
    spend_cap_usd: float
    diagnostic_rows: int
    diagnostic_tolerance: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_fracs", tuple(self.checkpoint_fracs))
        for name in (
            "rank", "batch_size", "epochs", "max_tokens", "render_version", "val_n", "diagnostic_rows", "seed"
        ):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an int, got {getattr(self, name)!r}")
        for name in ("learning_rate", "spend_cap_usd", "budget_fires_per_100"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        for name in ("rank", "batch_size", "epochs", "max_tokens", "val_n", "diagnostic_rows"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not (math.isfinite(self.oversample_corrective) and self.oversample_corrective >= 1.0):
            raise ValueError(f"oversample_corrective must be finite and >= 1.0, got {self.oversample_corrective}")
        if not 0.0 < self.diagnostic_tolerance < 1.0:
            raise ValueError(f"diagnostic_tolerance must be in (0, 1), got {self.diagnostic_tolerance}")
        if not self.checkpoint_fracs or any(not 0.0 < frac <= 1.0 for frac in self.checkpoint_fracs):
            raise ValueError(f"every checkpoint_frac must be in (0, 1], got {self.checkpoint_fracs}")
        if self.render_version != 2:
            raise ValueError(
                f"render_version must be 2, got {self.render_version}: the lane renders v2 unconditionally, so any "
                "other value would stamp registry metadata that deadlocks next week's probs-store read"
            )

    @classmethod
    def from_json(cls, path: Path) -> WatcherRecipe:
        """Parse an override recipe; missing or extra keys crash, and every value is validated."""
        return cls(**json.loads(path.read_text()))

    @classmethod
    def default(cls) -> WatcherRecipe:
        """Parse the packaged E8-winner recipe shipped in the wheel (``cc_steer/assets/watcher_recipe.json``)."""
        from cc_steer.assets import WATCHER_RECIPE_PATH

        return cls.from_json(WATCHER_RECIPE_PATH)


class IncumbentGate(NamedTuple):
    version: str
    probs: np.ndarray
    threshold: float


class TrainPlan(NamedTuple):
    """The training request plus the two scoring closures derived from one recipe over a frame's val split."""

    spec: TrainSpec
    policy: CheckpointPolicy
    eval_rows: tuple[EvalRow, ...]
    val_labels: np.ndarray
    select: Callable[[SavedCheckpoint], float]
    artifact_scorer: Callable[[Adapter], dict[str, float]]


def load_key(*, path: Path | None = None) -> None:
    """Export the Tinker credentials from ``~/.cc-steer/tinker.env`` (``path`` overrides) into the env.

    athome's :class:`~athome.train.spec.TinkerSettings` reads ``TINKER_API_KEY`` from the
    environment, so the weekly launchd job and every install keep a single credential location.
    """
    if os.environ.get("TINKER_API_KEY"):
        return
    for raw in (path or TINKER_ENV).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def register_watcher_adapter(
    adapter_dir: Path, *, metadata: dict[str, object], promote: bool = True, root: Path | None = None
) -> registry.VersionInfo:
    """Register (and by default promote) a built mlx-lm adapter as the ``watcher`` component.

    ``adapter_dir`` must hold the mlx-lm pair (``adapters.safetensors`` +
    ``adapter_config.json``); ``metadata`` must carry the keys
    :mod:`cc_steer.watcher.drafter_mlx` serves from: ``base_model``, ``thresholds``, and
    ``render_version``.
    """
    for key in ("base_model", "thresholds", "render_version"):
        if key not in metadata:
            raise ValueError(f"watcher metadata is missing {key!r}")
    if not isinstance(thresholds := metadata["thresholds"], dict) or "budget" not in thresholds:
        raise ValueError(f"watcher metadata thresholds must carry the 'budget' operating point, got {thresholds!r}")
    files = {name: adapter_dir / name for name in (drafter_mlx.ADAPTER_NAME, drafter_mlx.ADAPTER_CONFIG_NAME)}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{adapter_dir} has no {name}")
    info = registry.register(WATCHER_COMPONENT, files, metadata, root=root)
    if promote:
        registry.promote(WATCHER_COMPONENT, info.version, root=root)
    return info


def register_watcher(
    adapter_dir: Path, *, metadata: dict[str, object], registry_root: Path | None = None, state_dir: Path | None = None
) -> str:
    """Register and promote a pre-built adapter (no training), journaling the one-line verdict."""
    info = register_watcher_adapter(adapter_dir, metadata=metadata, root=registry_root)
    return promotion.journal(
        WATCHER_COMPONENT,
        f"registered and promoted {info.version}",
        dataset_digest=str(metadata.get("dataset_digest", "n/a")),
        hf_revision=str(rev) if (rev := metadata.get("hf_revision")) is not None else None,
        version=info.version,
        state_dir=state_dir,
    )


def seed_incumbent_probs(
    path: Path, *, version: str, expected_render: int, eval_root: Path | None = None
) -> Path:
    """Validate an external incumbent probs cache against the frozen frame, then write it through the store.

    The cache is the lab's flat ``{row_id: P(NO_STEER)}`` map. It must cover the current
    frame exactly — a missing row is incomplete, a foreign row means it was scored against a
    drifted eval. Its render is ``expected_render``, the incumbent's OWN contract from its
    registry metadata: a migrated incumbent is scored under the render it serves (the E12
    precedent), and the next retrain's :func:`~cc_steer.retrain.evalset.load_probs` verifies
    against the same metadata. On success it is written through
    :func:`~cc_steer.retrain.evalset.write_probs`, stamped with the frame digest.
    """
    frame = evalset.EvalFrame.load(root=eval_root)
    probs = {row_id: float(value) for row_id, value in json.loads(path.read_text()).items()}
    missing = [row_id for row_id in frame.ids if row_id not in probs]
    extra = [row_id for row_id in probs if row_id not in frame.ids]
    if missing or extra:
        raise evalset.ProbsStoreError(
            f"{path} does not match the frozen eval frame: {len(missing)} missing, {len(extra)} foreign rows; "
            "the seed cache was computed against a drifted eval"
        )
    fire_scores = 1.0 - np.array([probs[row_id] for row_id in frame.ids], dtype=np.float64)
    auc = promotion.sentinel_auc(frame.labels, fire_scores)
    return evalset.write_probs(frame, version, probs, auc=auc, render=expected_render, root=eval_root)


def retrain_watcher(
    *,
    force: bool = False,
    fresh_epoch: bool = False,
    recipe: WatcherRecipe,
    dataset_dir: Path | None = None,
    eval_root: Path | None = None,
    registry_root: Path | None = None,
    state_dir: Path | None = None,
    adapters_dir: Path | None = None,
) -> str:
    """One watcher retrain pass; returns the journaled one-line verdict.

    Skips when the watcher train view is unchanged and not forced. Otherwise translates the recipe
    into a :class:`~athome.train.spec.TrainSpec`, hands training, checkpoint selection, and
    materialization to :func:`athome.train.retrain`, scores the frozen frame through the served
    artifact, and gates on those served probs — on a pass registering, promoting, and seeding the
    new version's served probs before kicking the live watch agent. Every branch — skip,
    spend-cap reject, gate reject, promote — journals exactly once, and a serving-drift diagnostic
    persists beside the artifacts for every candidate that materializes, behind a boundary that
    keeps its own failure off the gate outcome.

    ``fresh_epoch`` is the one-shot clean-slate cutover: the incumbent-relative gate
    (:func:`~cc_steer.retrain.promotion.corrected_gate`) is skipped entirely — the candidate
    promotes on the served AUC floor alone (finite and above chance on the clean frame). It
    first refuses via :class:`FreshEpochError` if any registered version already carries probs
    for the current frozen frame, so it can only run once per frame.
    """
    incumbent = registry.current(WATCHER_COMPONENT, root=registry_root)
    digest = data.train_digest(dataset_dir=dataset_dir)
    hf_revision = data.hf_revision(dataset_dir=dataset_dir)
    if not promotion.should_retrain(incumbent, digest, force=force):
        return promotion.journal(
            WATCHER_COMPONENT,
            f"skipped (no new data at digest {digest})",
            dataset_digest=digest,
            hf_revision=hf_revision,
            state_dir=state_dir,
        )
    frame = evalset.EvalFrame.load(root=eval_root)
    base = _base_for(recipe)
    # Validate the incumbent gate / one-shot guard before any Tinker spend, so a bad probs file fails free.
    if fresh_epoch:
        _refuse_scored_frame(frame, eval_root=eval_root)
        incumbent_gate = None
    else:
        if incumbent is None:
            raise WatcherRetrainError(
                "no promoted watcher incumbent to gate against; register the base adapter first "
                "(cc-steer retrain --component watcher --register-adapter <dir> --metadata-json <json>)"
            )
        incumbent_gate = IncumbentGate(
            incumbent.version,
            _load_incumbent_probs(frame, incumbent, root=eval_root),
            float(incumbent.metadata["thresholds"]["budget"]),
        )

    spec, policy, eval_rows, val_labels, select, artifact_scorer = build_train_plan(
        recipe, frame, base, dataset_dir=dataset_dir
    )
    steps = spec.hyperparams.steps
    warranted = frame.corrective & frame.prose

    def gate(_served: dict[str, float]) -> GateVerdict:
        # The incumbent-relative decision needs the async judged gate, so it is made in run() below via
        # judged_gate; athome only records this placeholder, which the fresh-epoch floor never consults.
        return GateVerdict(promote=True, reason="deferred to judged gate", stats={})

    async def judged_gate(gate_state: IncumbentGate, served: dict[str, float]) -> GateVerdict:
        served_arr = np.array([served[row_id] for row_id in frame.ids], dtype=np.float64)
        candidate_fire, incumbent_fire = 1.0 - served_arr, 1.0 - gate_state.probs
        incumbent_fire_threshold = 1.0 - gate_state.threshold
        harmful = await judged.judged_harmful_favors_incumbent(
            candidate_fire_scores=candidate_fire,
            incumbent_fire_scores=incumbent_fire,
            incumbent_fire_threshold=incumbent_fire_threshold,
            frame=frame,
            warranted=warranted,
            root=eval_root,
        )
        result = promotion.corrected_gate(
            candidate_fire,
            incumbent_fire,
            candidate="candidate",
            incumbent=gate_state.version,
            incumbent_fire_threshold=incumbent_fire_threshold,
            labels=frame.labels,
            warranted=warranted,
            harmful_favors_incumbent=harmful,
        )
        verdict = promotion.watcher_promotable(result)
        return GateVerdict(promote=bool(verdict.promote), reason=verdict.reason, stats=_gate_stats(result))

    load_key()
    backend = TinkerBackend.from_settings()
    stage = adapters_dir or ADAPTER_STAGE_DIR
    stage.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="watcher-", dir=stage))
    sink = RunSink.open(work_dir / RUN_JOURNAL_NAME)

    async def run() -> tuple[RetrainOutcome, GateVerdict, dict[str, float], str]:
        outcome = await athome_retrain(
            backend,
            spec,
            checkpoints=policy,
            eval_rows=eval_rows,
            budget=SpendGuard(max_usd=recipe.spend_cap_usd),
            select=select,
            artifact_scorer=artifact_scorer,
            gate=gate,
            work_dir=work_dir,
            sink=sink,
        )
        # Runs after materialization but before the verdict acts, so per-row evidence survives a reject;
        # the boundary keeps its own failure off the outcome.
        diagnostic, diag_note = await _safe_serving_diagnostic(
            backend, outcome.best.path, frame, outcome.served, base, recipe, work_dir / DIAGNOSTIC_NAME
        )
        # The incumbent-relative gate — free metrics plus the judged harmful-fire term that buys opus
        # votes — runs here in the async path; fresh-epoch has no incumbent and rides the served AUC floor.
        verdict = await judged_gate(incumbent_gate, outcome.served) if incumbent_gate is not None else outcome.verdict
        return outcome, verdict, diagnostic, diag_note

    try:
        outcome, verdict, diagnostic, diag_note = anyio.run(run)
    except (SpendExceeded, InsufficientData) as error:
        return promotion.journal(
            WATCHER_COMPONENT,
            f"rejected ({error})",
            dataset_digest=digest,
            hf_revision=hf_revision,
            state_dir=state_dir,
        )

    served_probs = outcome.served
    served_arr = np.array([served_probs[row_id] for row_id in frame.ids], dtype=np.float64)
    eval_auc = promotion.sentinel_auc(frame.labels, 1.0 - served_arr)
    diag_suffix = f"; {diag_note}" if diag_note else ""

    if incumbent_gate is None:
        if not (np.isfinite(eval_auc) and eval_auc > 0.5):
            raise WatcherRetrainError(
                f"fresh-epoch candidate scores served sentinel AUC {eval_auc} on the clean frozen frame "
                f"(digest {frame.digest}); it must be finite and above chance (> 0.5) to promote"
            )
        gate_stats: dict[str, float] = {}
        reason = f"served AUC {eval_auc:.4f}, no incumbent gate"
    elif not verdict.promote:
        return promotion.journal(
            WATCHER_COMPONENT,
            f"rejected ({verdict.reason}){diag_suffix}",
            dataset_digest=digest,
            hf_revision=hf_revision,
            metrics=verdict.stats | diagnostic,
            state_dir=state_dir,
        )
    else:
        gate_stats = verdict.stats
        reason = verdict.reason

    # Store on the P(NO_STEER) scale (fire iff p < it); the cut is on 1 - p, so 1 - cut (lab ~0.15, run_e12.py:116).
    fire_score_cut = promotion.threshold_for_budget(
        1.0 - served_arr,
        fires_per_100=recipe.budget_fires_per_100,
        total_turns=len(frame),
    )
    threshold = 1.0 - fire_score_cut
    best_val_auc = sentinel.checkpoint_auc(val_labels, outcome.best)
    metadata: dict[str, object] = {
        "base_model": recipe.mlx_id,
        "render_version": recipe.render_version,
        "thresholds": {"budget": threshold},
        "dataset_digest": digest,
        **({"hf_revision": hf_revision} if hf_revision is not None else {}),
        "rank": recipe.rank,
        "learning_rate": recipe.learning_rate,
        "steps": steps,
        "best_step": outcome.best.step,
        "tinker_checkpoint": outcome.best.path,
        "tinker_val_auc": best_val_auc,
        "eval_auc": eval_auc,
        **gate_stats,
        **diagnostic,
    }
    info = register_watcher_adapter(outcome.adapter.adapter_dir, metadata=metadata, root=registry_root)
    registry.prune(WATCHER_COMPONENT, keep=KEEP_VERSIONS, root=registry_root)
    evalset.write_probs(frame, info.version, served_probs, auc=eval_auc, root=eval_root)
    kicked = launchd.kickstart_watch()
    prefix = "fresh-epoch " if fresh_epoch else ""
    return promotion.journal(
        WATCHER_COMPONENT,
        f"{prefix}promoted {info.version} ({reason}); watch kickstart {'ok' if kicked else 'skipped'}{diag_suffix}",
        dataset_digest=digest,
        hf_revision=hf_revision,
        metrics=gate_stats | diagnostic | {"tinker_val_auc": best_val_auc, "threshold_budget": threshold},
        version=info.version,
        state_dir=state_dir,
    )


def build_train_plan(
    recipe: WatcherRecipe, frame: evalset.EvalFrame, base: BaseModelSpec, *, dataset_dir: Path | None = None
) -> TrainPlan:
    """The :class:`~athome.train.TrainSpec` and scoring closures for one recipe over ``frame``'s val split.

    Curates the training pool (near-dup collapse, negative balance, corrective oversample), renders the
    carved-out val split as sentinel eval rows, and derives the checkpoint-selecting ``select`` and the
    served-MLX ``artifact_scorer``. Shared by the promotion pass (:func:`retrain_watcher`) and the
    base-model sweep's pure-observer scorer (:mod:`cc_steer.retrain.sweep`) so both train and score
    through one instrument.
    """
    rows = data.load_train_rows(dataset_dir=dataset_dir)
    rows = [rows[i] for i in data.near_dup_representatives(rows, seed=recipe.seed)[0]]
    val, rest = data.carve_val(rows, n=min(recipe.val_n, len(rows) // 10), seed=recipe.seed)
    pool = data.oversample_corrective_to(
        data.balance_no_steer(rest, seed=recipe.seed)[0], factor=recipe.oversample_corrective, seed=recipe.seed
    )[0]
    examples = tuple(_sft_example(row) for row in pool)
    val_labels = np.array([row.label for row in val], dtype=bool)
    eval_rows = tuple(sentinel.sentinel_eval_row(DRAFT_SYSTEM, row.draft_text(), recipe.mlx_id) for row in val)
    spec = TrainSpec(
        name=WATCHER_COMPONENT,
        base=base,
        dataset=Rows(examples=examples),
        hyperparams=Hyperparams(
            steps=recipe.epochs * math.ceil(len(examples) / recipe.batch_size),
            batch_size=recipe.batch_size,
            learning_rate=recipe.learning_rate,
            max_seq_len=recipe.max_tokens,
            seed=recipe.seed,
        ),
        method="sft",
        lora=LoraSpec(rank=recipe.rank),
        max_usd=recipe.spend_cap_usd,
    )

    def select(saved: SavedCheckpoint) -> float:
        return sentinel.checkpoint_auc(val_labels, saved)

    def artifact_scorer(adapter: Adapter) -> dict[str, float]:
        return _score_frame_local(frame, adapter.adapter_dir, recipe)

    return TrainPlan(
        spec=spec,
        policy=CheckpointPolicy(at=tuple(frac for frac in recipe.checkpoint_fracs if frac < 1.0)),
        eval_rows=eval_rows,
        val_labels=val_labels,
        select=select,
        artifact_scorer=artifact_scorer,
    )


def _sft_example(row: data.WatcherRow) -> SftExample:
    messages = data.training_sample(row, system=DRAFT_SYSTEM)["messages"]
    return SftExample(prompt=tuple(messages[:-1]), completion=(messages[-1],), id=row.id)


def _base_for(recipe: WatcherRecipe) -> BaseModelSpec:
    base = BASE_MODELS["qwen3-8b"]
    if recipe.tinker_model != base.tinker or recipe.mlx_id != base.mlx:
        raise WatcherRetrainError(
            f"recipe base {recipe.tinker_model}/{recipe.mlx_id} has no known BaseModel; "
            f"only {base.tinker}/{base.mlx} is supported"
        )
    return base


def _refuse_scored_frame(frame: evalset.EvalFrame, *, eval_root: Path | None) -> None:
    # Scan the probs store directly, not registry versions: an orphan probs file from a pruned
    # version still means the frame was scored, and the one-shot cutover must refuse on it too.
    probs_dir = evalset.eval_root(eval_root) / evalset.PROBS_DIRNAME
    scored = [path for path in sorted(probs_dir.glob("*.json")) if _stored_digest(path) == frame.digest]
    if scored:
        raise FreshEpochError(
            f"--fresh-epoch is a one-shot clean-slate cutover, but {len(scored)} probs file(s) already cover the "
            f"current frozen frame (digest {frame.digest}); the cutover is over. "
            f"Files: {[str(path) for path in scored]}"
        )


def _stored_digest(path: Path) -> str | None:
    payload = json.loads(path.read_text())
    meta = payload.get("meta") if isinstance(payload, dict) else None
    return meta.get("dataset_digest") if isinstance(meta, dict) else None


def _load_incumbent_probs(
    frame: evalset.EvalFrame, incumbent: registry.VersionInfo, *, root: Path | None
) -> np.ndarray:
    render = int(str(incumbent.metadata["render_version"]))
    try:
        return evalset.load_probs(frame, incumbent.version, expected_render=render, root=root)
    except evalset.ProbsStoreError as error:
        raise WatcherRetrainError(
            f"{error}; seed them with `cc-steer retrain --component watcher --seed-incumbent-probs <cache.json>`"
        ) from error


def _score_frame_local(frame: evalset.EvalFrame, candidate_dir: Path, recipe: WatcherRecipe) -> dict[str, float]:
    """Per-row served ``P(NO_STEER)`` for every frame row, through the converted MLX artifact.

    Loads the 4-bit base plus the candidate adapter — the production serving stack — and scores
    one row at a time, releasing MLX's buffer cache each row so peak memory stays at a single
    forward pass (the E12 Jetsam constraint on full-frame local scoring).
    """
    version = registry.VersionInfo(
        component=WATCHER_COMPONENT,
        version="candidate",
        path=candidate_dir,
        metadata={"base_model": recipe.mlx_id, "render_version": recipe.render_version, "thresholds": {"budget": 0.5}},
    )
    drafter = drafter_mlx.MlxDrafter(version=version, threshold=0.5)
    try:
        probs: dict[str, float] = {}
        for row_id, tail in zip(frame.ids, frame.tails, strict=True):
            probs[row_id] = drafter.nosteer_prob(tail)
            drafter.clear_cache()
        return probs
    finally:
        del drafter


async def _serving_diagnostic(
    backend: TinkerBackend,
    tinker_path: str,
    frame: evalset.EvalFrame,
    served_probs: dict[str, float],
    base: BaseModelSpec,
    recipe: WatcherRecipe,
    out_path: Path,
) -> dict[str, float]:
    """Sample the frame Tinker-vs-served, persist per-row drift to ``out_path``, return the summary.

    Observability only — the summary rides the journal metrics and the sidecar holds the per-row
    diffs, but nothing here gates promotion: under eval-what-you-serve a conversion bug is caught
    by the served AUC floor, not a parity threshold. Raises on a Tinker or sidecar-write failure;
    the lane calls it through :func:`_safe_serving_diagnostic`, which never lets it reach the gate.
    """
    idx = _stratified_indices(frame.labels, n=recipe.diagnostic_rows, seed=recipe.seed)
    rows = [sentinel.sentinel_eval_row(DRAFT_SYSTEM, frame.tails[i], recipe.mlx_id) for i in idx]
    scored = await backend.score(tinker_path, rows, base=base, budget=SpendGuard(max_usd=recipe.spend_cap_usd))
    tinker_probs = {frame.ids[i]: math.exp(scored[k].logprob) for k, i in enumerate(idx)}
    diffs = {i: abs(served_probs[frame.ids[i]] - tinker_probs[frame.ids[i]]) for i in idx}
    summary = {
        "diagnostic_rows": float(len(idx)),
        "diagnostic_max_abs_diff": max(diffs.values()),
        "diagnostic_median_abs_diff": float(np.median(list(diffs.values()))),
        "diagnostic_over_tolerance": float(sum(diff > recipe.diagnostic_tolerance for diff in diffs.values())),
    }
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "dataset_digest": frame.digest,
                    "tinker_checkpoint": tinker_path,
                    "diagnostic_tolerance": recipe.diagnostic_tolerance,
                    "seed": recipe.seed,
                },
                "summary": summary,
                "rows": [
                    {
                        "index": i,
                        "row_id": frame.ids[i],
                        "label": bool(frame.labels[i]),
                        "served": served_probs[frame.ids[i]],
                        "tinker": tinker_probs[frame.ids[i]],
                        "abs_diff": diffs[i],
                    }
                    for i in idx
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


async def _safe_serving_diagnostic(
    backend: TinkerBackend,
    tinker_path: str,
    frame: evalset.EvalFrame,
    served_probs: dict[str, float],
    base: BaseModelSpec,
    recipe: WatcherRecipe,
    out_path: Path,
) -> tuple[dict[str, float], str]:
    """Run :func:`_serving_diagnostic` behind a tight boundary so it can never disturb the outcome.

    Returns ``(summary, "")`` on success. On a Tinker/API failure (an :class:`~athome.errors.AthomeError`
    such as a spend cap, a legacy ``RuntimeError``) or a sidecar-write ``OSError`` it returns
    ``({"diagnostic_failed": 1.0}, "<reason>")`` — a journaled metric marker plus a verdict sub-note
    — so a diagnostic mishap records itself and the promote/reject proceeds untouched.
    """
    try:
        return await _serving_diagnostic(backend, tinker_path, frame, served_probs, base, recipe, out_path), ""
    except (OSError, RuntimeError, AthomeError) as error:
        return {"diagnostic_failed": 1.0}, f"serving diagnostic failed ({type(error).__name__}: {error})"


def _stratified_indices(labels: np.ndarray, *, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    total = len(labels)
    picked: list[int] = []
    for value in (True, False):
        idx = np.flatnonzero(labels == value)
        if not len(idx):
            continue
        take = min(len(idx), max(1, round(n * len(idx) / total)), n - len(picked))
        picked.extend(int(idx[j]) for j in rng.choice(len(idx), size=take, replace=False))
    return sorted(picked)


def _gate_stats(result: GateResult) -> dict[str, float]:
    return {
        "coverage_wins": float(result.coverage_wins),
        "coverage_losses": float(result.coverage_losses),
        "coverage_sign_p": float(result.coverage_sign_p),
        "budget_held": float(result.budget_held),
        "cell_auc": float(result.cell_auc),
        "incumbent_auc": float(result.incumbent_auc),
        "auc_not_regressed": float(result.auc_not_regressed),
    }
