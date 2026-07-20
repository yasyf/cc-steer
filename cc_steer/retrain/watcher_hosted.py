"""Hosted-substrate calibration: score the frozen frame through a live endpoint and mint a distinct lane.

The HTTP sibling of the local frame-scoring-plus-threshold-fit-plus-mint lineage in
:mod:`cc_steer.retrain.watcher` and :mod:`cc_steer.retrain.refit`, for a watcher served over a
hosted vLLM endpoint (the athome ``[serve.modal-vllm]`` scale-to-zero recipe). A served model's
abstain probabilities are substrate-specific — the spike measured the same cutoff firing 9 rows
locally versus 26 through the hosted endpoint — so a hosted deployment needs its own threshold, fit
from probabilities the endpoint actually produces.

:func:`score_frame_http` drives :class:`~cc_steer.watcher.drafter_http.HttpDrafter` as a library:
one ``max_tokens=1`` teacher-forced ``prompt_logprobs`` scoring call per frame row, scoring the same
divergence-found answer position :meth:`~cc_steer.watcher.drafter_mlx.MlxDrafter._prefix_and_sentinel`
finds locally, so the two substrates score the identical prompt and differ only in substrate. The
served ``P(NO_STEER)`` is exact and rank-independent — every row is scorable, so :func:`fit_threshold`
fits the operating point through :func:`~cc_steer.retrain.promotion.threshold_for_budget` over every
row in the same inverted convention the local lanes use. :func:`apply_calibration` mints and promotes
the result under a component distinct from the local ``watcher`` lane, copying the promoted local
adapter's bytes verbatim exactly as :func:`~cc_steer.retrain.refit.apply_refit` does. The local lane
and the running daemon — which reads ``registry.current("watcher")`` by literal — are never touched,
and unlike the daemon this never kicks the watch agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import httpx
import numpy as np

from cc_steer import registry
from cc_steer.retrain import evalset, promotion
from cc_steer.watcher.drafter_http import DEFAULT_THRESHOLD, DrafterResponseError, HttpDrafter

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_FIRES_PER_100 = 2.0
DEFAULT_COMPONENT = "watcher-hosted"
SOURCE_COMPONENT = "watcher"
WATCHER_THRESHOLD_KEY = "budget"
SCORE_CONCURRENCY = 8


class HostedCalibrationError(RuntimeError):
    """The hosted calibration cannot proceed: no promoted adapter, an unscoreable frame, or a below-chance fit."""


@dataclass(frozen=True, slots=True)
class HostedCalibration:
    """A computed hosted-substrate calibration — the fitted threshold and everything the mint needs.

    Attributes:
        component: The distinct registry lane the result is minted into, e.g. ``watcher-hosted``.
        endpoint: The OpenAI-compatible base URL the frame was scored through.
        model: The served model/adapter name — the OpenAI ``model`` field.
        fires_per_100: The fire budget the threshold was fit to, per 100 eval rows.
        n_rows: Every frame row — the budget denominator, all of which are scored and fit.
        eval_auc: The served sentinel AUC over every row — a sanity floor, not a gate.
        current_threshold: The served threshold the parent local adapter carries today.
        fitted_threshold: The hosted-substrate threshold, in the watcher's abstain convention.
        passes: How many scored rows would fire under ``fitted_threshold``.
        parent: The promoted local watcher version whose artifact bytes the new version copies.
        threshold_key: The ``thresholds`` metadata key the fitted value writes to.
        digest: The frozen eval frame's content digest the calibration ran against.
    """

    component: str
    endpoint: str
    model: str
    fires_per_100: float
    n_rows: int
    eval_auc: float
    current_threshold: float
    fitted_threshold: float
    passes: int
    parent: registry.VersionInfo
    threshold_key: str
    digest: str

    @property
    def per_100(self) -> float:
        return 100.0 * self.passes / self.n_rows

    def metadata(self) -> dict[str, object]:
        return self.parent.metadata | {
            "thresholds": self.parent.metadata["thresholds"] | {self.threshold_key: self.fitted_threshold},
            "hosted": {
                "parent_version": self.parent.version,
                "endpoint": self.endpoint,
                "model": self.model,
                "fires_per_100": self.fires_per_100,
                "n_rows": self.n_rows,
                "eval_auc": self.eval_auc,
                "eval_digest": self.digest,
            },
        }

    def report(self) -> str:
        return "\n".join(
            (
                f"component: {self.component}",
                f"endpoint: {self.endpoint}",
                f"model: {self.model}",
                f"eval rows: {self.n_rows}",
                f"served eval sentinel AUC: {self.eval_auc:.4f}",
                f"parent adapter: {self.parent.version}",
                f"current served threshold: {self.current_threshold:.6g}",
                f"fitted threshold: {self.fitted_threshold:.6g}",
                f"with fitted threshold, {self.passes} of {self.n_rows} rows fire in production "
                f"({self.per_100:.2f} per 100)",
            )
        )


async def score_frame_http(
    frame: evalset.EvalFrame,
    *,
    endpoint: str,
    model: str,
    timeout: float,
    api_key: str | None,
    concurrency: int = SCORE_CONCURRENCY,
) -> dict[str, float]:
    """Teacher-forced first-answer-token ``P(NO_STEER)`` for every frame row, through a hosted endpoint.

    Drives :class:`~cc_steer.watcher.drafter_http.HttpDrafter` — the same object the daemon serves
    ``--drafter http`` with — as a library, one ``max_tokens=1`` ``prompt_logprobs`` scoring call per
    row fanned out concurrently under a bounded limiter. Each row scores the divergence-found answer
    position :meth:`~cc_steer.watcher.drafter_mlx.MlxDrafter._prefix_and_sentinel` finds locally, so
    the two substrates score the identical prompt and differ only in substrate. The served
    ``P(NO_STEER)`` is exact and rank-independent, so every row is scorable. Unlike the daemon's
    fail-open, a transport fault, timeout, or malformed response aborts the whole pass: an offline
    fit on a silently partial frame is a worse outcome than a loud failure the operator re-runs.

    Args:
        frame: The frozen watcher eval frame whose ``tails`` are scored.
        endpoint: The vLLM base URL; ``/v1/completions`` is appended.
        model: The served model/adapter name — the OpenAI ``model`` field.
        timeout: Per-request timeout in seconds.
        api_key: The endpoint bearer token, or None for an unauthenticated endpoint.
        concurrency: The maximum in-flight scoring requests.

    Returns:
        ``{row_id: P(NO_STEER)}`` for every frame row.
    """
    drafter = HttpDrafter(endpoint=endpoint, model=model, threshold=DEFAULT_THRESHOLD, timeout=timeout, api_key=api_key)
    limiter = anyio.CapacityLimiter(concurrency)
    probs: dict[str, float] = {}

    async def score(row_id: str, tail: str) -> None:
        async with limiter:
            probs[row_id] = await drafter.nosteer_prob(tail)

    try:
        async with anyio.create_task_group() as group:
            for row_id, tail in zip(frame.ids, frame.tails, strict=True):
                group.start_soon(score, row_id, tail)
    except* (httpx.HTTPError, json.JSONDecodeError, DrafterResponseError) as group:
        raise HostedCalibrationError(
            f"scoring through {endpoint} failed: {group.exceptions[0]!r}; nothing was fit — "
            "re-run once the endpoint is reachable"
        ) from group
    finally:
        await drafter.aclose()
    return probs


def fit_threshold(scored: np.ndarray, *, fires_per_100: float, total_turns: int) -> float:
    """Fit the served ``P(NO_STEER)`` abstain threshold to a fire budget, in the watcher's inverted convention.

    The watcher fires at ``p < threshold`` on ``P(NO_STEER)``, so scores invert to fire-scale
    ``1 - p`` for the fitter and the fitted cut maps back with ``threshold = 1 - cut`` — the exact
    inversion :func:`~cc_steer.retrain.watcher.retrain_watcher` and
    :func:`~cc_steer.retrain.refit.plan_refit` use; getting it backwards silently inverts the
    cascade. ``total_turns`` is the full frame, matching the ``scored`` length since every row is
    scorable on the exact ``prompt_logprobs`` path.
    """
    return 1.0 - promotion.threshold_for_budget(1.0 - scored, fires_per_100=fires_per_100, total_turns=total_turns)


async def plan_calibration(
    *,
    endpoint: str,
    model: str,
    timeout: float,
    api_key: str | None,
    fires_per_100: float = DEFAULT_FIRES_PER_100,
    component: str = DEFAULT_COMPONENT,
    eval_root: Path | None = None,
    registry_root: Path | None = None,
) -> HostedCalibration:
    """Score the frozen frame through the endpoint and fit a hosted threshold, leaving the registry untouched.

    Reads the promoted local ``watcher`` adapter as the byte source for the copy, scores every row of
    the frozen eval frame through ``endpoint`` with the HttpDrafter sentinel semantics, and fits the
    served threshold. Refuses — before minting anything — when no local watcher is promoted, or when
    the hosted-scored frame's sentinel AUC is not finite and above chance (a broken served adapter,
    mirroring the fresh-epoch AUC floor).
    """
    frame = evalset.EvalFrame.load(root=eval_root)
    parent = registry.current(SOURCE_COMPONENT, root=registry_root)
    if parent is None:
        raise HostedCalibrationError(
            f"no promoted {SOURCE_COMPONENT} adapter to copy into the {component} lane; train and promote one first"
        )
    current_threshold = _current_threshold(parent)
    served = await score_frame_http(frame, endpoint=endpoint, model=model, timeout=timeout, api_key=api_key)
    labels = np.asarray(frame.labels, dtype=bool)
    scored = np.array([served[row_id] for row_id in frame.ids], dtype=np.float64)
    eval_auc = float(promotion.sentinel_auc(labels, 1.0 - scored))
    if not (np.isfinite(eval_auc) and eval_auc > 0.5):
        raise HostedCalibrationError(
            f"the hosted-scored frame's sentinel AUC is {eval_auc} on {len(scored)} rows; it must be finite "
            "and above chance (> 0.5) before a hosted threshold can be trusted — the served adapter is likely wrong"
        )
    fitted = fit_threshold(scored, fires_per_100=fires_per_100, total_turns=len(frame))
    return HostedCalibration(
        component=component,
        endpoint=endpoint,
        model=model,
        fires_per_100=fires_per_100,
        n_rows=len(frame),
        eval_auc=eval_auc,
        current_threshold=current_threshold,
        fitted_threshold=fitted,
        passes=int(np.count_nonzero(scored < fitted)),
        parent=parent,
        threshold_key=WATCHER_THRESHOLD_KEY,
        digest=str(frame.digest),
    )


def apply_calibration(
    plan: HostedCalibration, *, registry_root: Path | None = None, state_dir: Path | None = None
) -> str:
    """Mint and promote a new hosted-lane version copying the parent adapter's bytes verbatim, then journal.

    Shaped exactly like :func:`~cc_steer.retrain.refit.apply_refit`, minus the ``launchd`` kick — the
    daemon reads ``registry.current("watcher")`` by literal, so the hosted component is invisible to
    it and kicking the watch agent would be a no-op at best and misleading at worst. The local
    ``watcher`` lane is never touched.
    """
    files: dict[str, Path] = {
        path.name: path for path in plan.parent.path.iterdir() if path.is_file() and path.name != registry.METADATA_NAME
    }
    info = registry.register(plan.component, files, plan.metadata(), root=registry_root)
    registry.promote(plan.component, info.version, root=registry_root)
    verdict = (
        f"hosted calibrate {plan.parent.version} -> {info.version}: {plan.threshold_key} "
        f"{plan.current_threshold:.6g} -> {plan.fitted_threshold:.6g} on {plan.n_rows} rows "
        f"through {plan.model} at {plan.endpoint} "
        f"({plan.passes} would fire, {plan.per_100:.2f}/100)"
    )
    return promotion.journal(
        plan.component,
        verdict,
        dataset_digest=str(plan.parent.metadata.get("dataset_digest")),
        version=info.version,
        state_dir=state_dir,
    )


async def calibrate(
    *,
    endpoint: str,
    model: str,
    timeout: float,
    api_key: str | None,
    fires_per_100: float = DEFAULT_FIRES_PER_100,
    dry_run: bool,
    component: str = DEFAULT_COMPONENT,
    eval_root: Path | None = None,
    registry_root: Path | None = None,
    state_dir: Path | None = None,
) -> str:
    """Score the frozen frame through a hosted endpoint, fit a substrate threshold, and promote it into a distinct lane.

    Dry-run returns the fitted threshold and coverage without touching the registry; otherwise mints
    and promotes a new ``component`` version copying the promoted local watcher adapter's bytes
    verbatim with the hosted threshold, and journals one line. The local ``watcher`` lane and the
    running daemon are never touched.
    """
    plan = await plan_calibration(
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        api_key=api_key,
        fires_per_100=fires_per_100,
        component=component,
        eval_root=eval_root,
        registry_root=registry_root,
    )
    return plan.report() if dry_run else apply_calibration(plan, registry_root=registry_root, state_dir=state_dir)


def _current_threshold(parent: registry.VersionInfo) -> float:
    thresholds = parent.metadata.get("thresholds")
    if not isinstance(thresholds, dict) or WATCHER_THRESHOLD_KEY not in thresholds:
        raise HostedCalibrationError(
            f"{parent.component} version {parent.version} carries no thresholds[{WATCHER_THRESHOLD_KEY!r}] to copy"
        )
    return float(thresholds[WATCHER_THRESHOLD_KEY])
