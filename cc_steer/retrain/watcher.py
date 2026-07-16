"""The watcher-component retrain lane: train a LoRA on Tinker, gate it, promote it locally.

The full weekly pass for the generative watcher. It trains the E8-winning recipe on the
curated pool (:mod:`cc_steer.retrain.data`), scores checkpoints on a carved val, gates the
best checkpoint against the incumbent on the frozen eval with the corrected paired gate
(:func:`~cc_steer.retrain.promotion.corrected_gate`), converts the winner to a local mlx-lm
adapter, spot-checks Tinker-vs-serve parity through the production
:class:`~cc_steer.watcher.drafter_mlx.MlxDrafter`, and only then registers and promotes it —
seeding next week's incumbent probabilities. Every outcome journals one line.

A reject costs nothing further: the gate runs before any download or convert, and parity
runs before any register. The spend cap is a gate outcome, not a crash — a projected
overspend journals a reject and returns without ever constructing a Tinker client.
:class:`WatcherRecipe` validates every knob at parse so a degenerate value never reaches
the network. Every ``tinker`` / ``mlx`` import is lazy, behind :mod:`cc_steer.retrain.tinker`
and :mod:`cc_steer.watcher.drafter_mlx`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from cc_steer import launchd, registry
from cc_steer.retrain import data, evalset, promotion
from cc_steer.retrain import tinker as tk
from cc_steer.watcher import drafter_mlx
from cc_steer.watcher.cascade import DRAFT_SYSTEM

if TYPE_CHECKING:
    import tinker

    from cc_steer.retrain.promotion import GateResult

WATCHER_COMPONENT = drafter_mlx.COMPONENT
KEEP_VERSIONS = 3
ADAPTER_STAGE_DIR: Path = Path.home() / ".cc-steer" / "adapters" / "staging"


class WatcherRetrainError(RuntimeError):
    """The watcher retrain cannot proceed: no incumbent, unseeded incumbent probs, or an unknown base."""


class FreshEpochError(WatcherRetrainError):
    """``--fresh-epoch`` misuse: the frozen frame already has scored version probs, so the one-shot cutover is over."""


class ConversionDroppedError(RuntimeError):
    """The PEFT-to-mlx conversion dropped modules; promoting a half-converted adapter is refused."""

    def __init__(self, dropped: list[str]) -> None:
        super().__init__(
            f"watcher adapter conversion dropped {dropped}; refusing to promote a half-converted adapter"
        )
        self.dropped = dropped


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
        parity_rows: The label-stratified rows the Tinker-vs-serve parity check compares.
        parity_tolerance: The max absolute nosteer-prob gap parity tolerates before rejecting.
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
    parity_rows: int
    parity_tolerance: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_fracs", tuple(self.checkpoint_fracs))
        for name in ("rank", "batch_size", "epochs", "max_tokens", "render_version", "val_n", "parity_rows", "seed"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an int, got {getattr(self, name)!r}")
        for name in ("learning_rate", "spend_cap_usd", "budget_fires_per_100"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        for name in ("rank", "batch_size", "epochs", "max_tokens", "val_n", "parity_rows"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not (math.isfinite(self.oversample_corrective) and self.oversample_corrective >= 1.0):
            raise ValueError(f"oversample_corrective must be finite and >= 1.0, got {self.oversample_corrective}")
        if not 0.0 < self.parity_tolerance < 1.0:
            raise ValueError(f"parity_tolerance must be in (0, 1), got {self.parity_tolerance}")
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
    auc = promotion.sentinel_auc(frame.labels, np.array([probs[row_id] for row_id in frame.ids], dtype=np.float64))
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

    Skips when the watcher train view is unchanged and not forced. Otherwise trains the
    recipe, gates the best checkpoint against the incumbent, and on a pass converts,
    parity-checks, registers, promotes, and seeds the new version's probs before kicking the
    live watch agent. Every branch — skip, spend-cap reject, gate reject, parity reject,
    promote — journals exactly once.

    ``fresh_epoch`` is the one-shot clean-slate cutover: the incumbent-relative gate
    (:func:`~cc_steer.retrain.promotion.corrected_gate`) is skipped entirely — the candidate
    promotes on the absolute checks the lane already runs (spend cap, mlx parity) plus a single
    below-chance refusal (sentinel AUC must beat 0.5 on the clean frame). It first refuses via
    :class:`FreshEpochError` if any registered version already carries probs for the current
    frozen frame, so it can only run once per frame.
    """
    incumbent = registry.current(WATCHER_COMPONENT, root=registry_root)
    digest = data.train_digest(dataset_dir=dataset_dir)
    if not promotion.should_retrain(incumbent, digest, force=force):
        return promotion.journal(
            WATCHER_COMPONENT, f"skipped (no new data at digest {digest})", dataset_digest=digest, state_dir=state_dir
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

    rows = data.load_train_rows(dataset_dir=dataset_dir)
    rows = [rows[i] for i in data.near_dup_representatives(rows, seed=recipe.seed)[0]]
    val, rest = data.carve_val(rows, n=min(recipe.val_n, len(rows) // 10), seed=recipe.seed)
    pool = data.oversample_corrective_to(
        data.balance_no_steer(rest, seed=recipe.seed)[0], factor=recipe.oversample_corrective, seed=recipe.seed
    )[0]
    datums = [tk.build_datum(data.training_sample(row, system=DRAFT_SYSTEM)["messages"], recipe.mlx_id) for row in pool]
    valid_rows = [{"system": DRAFT_SYSTEM, "user": row.draft_text(), "assistant": row.reference} for row in val]
    steps = recipe.epochs * math.ceil(
        sum(1 for d in datums if d.model_input.length + 1 <= recipe.max_tokens) / recipe.batch_size
    )

    captured: list[tinker.ServiceClient] = []
    scores: dict[int, float] = {}

    def on_ckpt(step: int, path: str, service_client: tinker.ServiceClient, _base: tk.BaseModel) -> None:
        if not captured:
            captured.append(service_client)
        scores[step] = tk.score_auc_tinker(service_client, path, valid_rows, recipe.mlx_id)["auc"]

    try:
        run = tk.train_lora(
            datums,
            base,
            steps=steps,
            batch_size=recipe.batch_size,
            learning_rate=recipe.learning_rate,
            rank=recipe.rank,
            seed=recipe.seed,
            spend_cap_usd=recipe.spend_cap_usd,
            max_tokens=recipe.max_tokens,
            checkpoint_fractions=recipe.checkpoint_fracs,
            on_checkpoint=on_ckpt,
        )
    except (tk.SpendCapError, tk.InsufficientDatumsError) as error:
        return promotion.journal(WATCHER_COMPONENT, f"rejected ({error})", dataset_digest=digest, state_dir=state_dir)

    svc = captured[0]
    best_step = max(scores, key=lambda step: scores[step])
    candidate_probs = tk.score_frame_tinker(svc, run.checkpoints[best_step], frame, base=base)
    candidate_arr = np.array([candidate_probs[row_id] for row_id in frame.ids], dtype=np.float64)
    eval_auc = promotion.sentinel_auc(frame.labels, candidate_arr)

    if incumbent_gate is None:
        if not (np.isfinite(eval_auc) and eval_auc > 0.5):
            raise WatcherRetrainError(
                f"fresh-epoch candidate scores sentinel AUC {eval_auc} on the clean frozen frame "
                f"(digest {frame.digest}); it must be finite and above chance (> 0.5) to promote"
            )
        gate_stats: dict[str, float] = {}
        reason = f"candidate AUC {eval_auc:.4f}, no incumbent gate"
    else:
        result = promotion.corrected_gate(
            candidate_arr,
            incumbent_gate.probs,
            candidate="candidate",
            incumbent=incumbent_gate.version,
            incumbent_threshold=incumbent_gate.threshold,
            labels=frame.labels,
            corrective=frame.corrective,
            prose=frame.prose,
            harmful_favors_incumbent=None,
        )
        gate_verdict = promotion.watcher_promotable(result)
        if not gate_verdict.promote:
            return promotion.journal(
                WATCHER_COMPONENT,
                f"rejected ({gate_verdict.reason})",
                dataset_digest=digest,
                metrics=_gate_stats(result),
                state_dir=state_dir,
            )
        gate_stats = _gate_stats(result)
        reason = gate_verdict.reason

    candidate_dir = _convert_best(svc, run.checkpoints[best_step], base, recipe, adapters_dir=adapters_dir)
    parity = _parity_check(frame, candidate_probs, candidate_dir, recipe)
    if parity > recipe.parity_tolerance:
        return promotion.journal(
            WATCHER_COMPONENT,
            f"rejected (parity max abs diff {parity:.4f} > tolerance {recipe.parity_tolerance})",
            dataset_digest=digest,
            metrics=gate_stats | {"parity_max_abs_diff": parity},
            state_dir=state_dir,
        )

    # Store on the P(NO_STEER) scale (fire iff p < it); the cut is on 1 - p, so 1 - cut (lab ~0.15, run_e12.py:116).
    fire_score_cut = promotion.threshold_for_budget(
        1.0 - candidate_arr,
        fires_per_100=recipe.budget_fires_per_100,
        total_turns=len(frame),
    )
    threshold = 1.0 - fire_score_cut
    metadata: dict[str, object] = {
        "base_model": recipe.mlx_id,
        "render_version": recipe.render_version,
        "thresholds": {"budget": threshold},
        "dataset_digest": digest,
        "rank": recipe.rank,
        "learning_rate": recipe.learning_rate,
        "steps": steps,
        "best_step": best_step,
        "tinker_checkpoint": run.checkpoints[best_step],
        "tinker_val_auc": scores[best_step],
        "eval_auc": eval_auc,
        "parity_max_abs_diff": parity,
        **gate_stats,
    }
    info = register_watcher_adapter(candidate_dir, metadata=metadata, root=registry_root)
    registry.prune(WATCHER_COMPONENT, keep=KEEP_VERSIONS, root=registry_root)
    evalset.write_probs(frame, info.version, candidate_probs, auc=eval_auc, root=eval_root)
    kicked = launchd.kickstart_watch()
    prefix = "fresh-epoch " if fresh_epoch else ""
    return promotion.journal(
        WATCHER_COMPONENT,
        f"{prefix}promoted {info.version} ({reason}); watch kickstart {'ok' if kicked else 'skipped'}",
        dataset_digest=digest,
        metrics=gate_stats
        | {"parity_max_abs_diff": parity, "tinker_val_auc": scores[best_step], "threshold_budget": threshold},
        version=info.version,
        state_dir=state_dir,
    )


def _base_for(recipe: WatcherRecipe) -> tk.BaseModel:
    if recipe.tinker_model != tk.QWEN3_8B.tinker_model or recipe.mlx_id != tk.QWEN3_8B.mlx_id:
        raise WatcherRetrainError(
            f"recipe base {recipe.tinker_model}/{recipe.mlx_id} has no known BaseModel; "
            f"only {tk.QWEN3_8B.tinker_model}/{tk.QWEN3_8B.mlx_id} is supported"
        )
    return tk.QWEN3_8B


def _refuse_scored_frame(frame: evalset.EvalFrame, *, eval_root: Path | None) -> None:
    # Scan the probs store directly, not registry versions: an orphan probs file from a pruned
    # version still means the frame was scored, and the one-shot cutover must refuse on it too.
    probs_dir = evalset.eval_root(eval_root) / evalset.PROBS_DIRNAME
    scored = [path for path in sorted(probs_dir.glob("*.json")) if _stored_digest(path) == frame.digest]
    if scored:
        raise FreshEpochError(
            f"--fresh-epoch is a one-shot clean-slate cutover, but {len(scored)} probs file(s) already cover the "
            f"current frozen frame (digest {frame.digest}); the cutover is over. Files: {[str(path) for path in scored]}"
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


def _convert_best(
    svc: tinker.ServiceClient,
    tinker_path: str,
    base: tk.BaseModel,
    recipe: WatcherRecipe,
    *,
    adapters_dir: Path | None,
) -> Path:
    import tempfile

    stage = adapters_dir or ADAPTER_STAGE_DIR
    stage.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="watcher-", dir=stage))
    tk.download_adapter(svc, tinker_path, workdir / "peft")
    candidate_dir = workdir / "mlx"
    conversion = tk.convert_peft_to_mlx(workdir / "peft", candidate_dir, num_layers=base.num_layers, rank=recipe.rank)
    if conversion["dropped"]:
        raise ConversionDroppedError(list(conversion["dropped"]))
    return candidate_dir


def _parity_check(
    frame: evalset.EvalFrame, candidate_probs: dict[str, float], candidate_dir: Path, recipe: WatcherRecipe
) -> float:
    version = registry.VersionInfo(
        component=WATCHER_COMPONENT,
        version="candidate",
        path=candidate_dir,
        metadata={"base_model": recipe.mlx_id, "render_version": recipe.render_version, "thresholds": {"budget": 0.5}},
    )
    drafter = drafter_mlx.MlxDrafter(version=version, threshold=0.5)
    diffs = [
        abs(drafter.nosteer_prob(frame.tails[i]) - candidate_probs[frame.ids[i]])
        for i in _stratified_indices(frame.labels, n=recipe.parity_rows, seed=recipe.seed)
    ]
    del drafter
    return max(diffs)


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
