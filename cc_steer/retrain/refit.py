"""Live-distribution threshold re-fit: move a served operating point onto the scores production actually produces.

The offline gate threshold was fit on a view-ratio proxy and passes zero live turns, so the
cascade can never fire. This lane re-fits the served threshold from the ``scored_moments`` ledger
the running daemon fills — through the same fire-budget fitter the training lanes use
(:func:`~cc_steer.retrain.promotion.threshold_for_budget`) — over a recent window, copies the
promoted version's artifact bytes verbatim, and registers then promotes a new immutable version
carrying only the updated threshold and refit provenance. The model is untouched; one number moves.

The two components fit in opposite directions. The gate scores P(steer) and fires at
``score >= threshold``, so its scores fit directly. The watcher stores P(NO_STEER) and fires at
``p < threshold`` — the abstain convention :class:`~cc_steer.watcher.drafter_mlx.MlxDrafter` serves
and :meth:`~cc_steer.watcher.cascade.Cascade.evaluate` records — so its scores are inverted to
fire-direction ``1 - p`` before fitting and the fitted cut is mapped back with ``threshold = 1 - cut``,
mirroring :func:`~cc_steer.retrain.watcher.retrain_watcher`. Getting this backwards silently inverts
the cascade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from cc_steer import launchd, registry
from cc_steer.retrain import promotion
from cc_steer.watcher import gate
from cc_steer.watcher.delivery import ShadowDelivery

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_FIRES_PER_100 = 2.0
WATCHER_THRESHOLD_KEY = "budget"
WINDOW_PATTERN = re.compile(r"^(\d+)d$")


class RefitError(RuntimeError):
    """The re-fit cannot proceed: a malformed window, an empty window, no eligible rows, or nothing promoted."""


@dataclass(frozen=True, slots=True)
class RefitPlan:
    """A computed threshold re-fit — the fitted operating point and everything the mint needs.

    Attributes:
        component: The component re-fit, ``gate`` or ``watcher``.
        since: The window the live rows were drawn from, e.g. ``7d``.
        window_days: The window in whole days, stamped into the refit provenance.
        fires_per_100: The fire budget the threshold was fit to, per 100 total turns.
        n_rows: Every scored row in the window — the end-to-end budget denominator.
        n_eligible: The rows the fit ranked: all of them for the gate; the gate-passed,
            stage-2-scored rows for the watcher.
        current_threshold: The served threshold the promoted version carries today.
        fitted_threshold: The re-fit threshold, in the component's served convention.
        passes: How many live rows would fire under ``fitted_threshold``.
        parent: The promoted version whose artifact bytes the new version copies.
        threshold_key: The ``thresholds`` metadata key the fitted value writes to.
    """

    component: str
    since: str
    window_days: int
    fires_per_100: float
    n_rows: int
    n_eligible: int
    current_threshold: float
    fitted_threshold: float
    passes: int
    parent: registry.VersionInfo
    threshold_key: str

    @property
    def per_100(self) -> float:
        return 100.0 * self.passes / self.n_rows

    def metadata(self) -> dict[str, object]:
        return self.parent.metadata | {
            "thresholds": self.parent.metadata["thresholds"] | {self.threshold_key: self.fitted_threshold},
            "refit": {
                "parent_version": self.parent.version,
                "window_days": self.window_days,
                "n_rows": self.n_rows,
                "fires_per_100": self.fires_per_100,
            },
        }

    def report(self) -> str:
        return "\n".join(
            (
                f"component: {self.component}",
                f"window: {self.since}",
                f"rows: {self.n_rows} (eligible {self.n_eligible})",
                f"current served threshold: {self.current_threshold:.6g}",
                f"fitted threshold: {self.fitted_threshold:.6g}",
                f"with fitted threshold, {self.passes} of {self.n_rows} live rows would pass "
                f"({self.per_100:.2f} per 100)",
            )
        )


async def plan_refit(
    component: str,
    *,
    since: str,
    fires_per_100: float = DEFAULT_FIRES_PER_100,
    db_path: Path | None = None,
    root: Path | None = None,
    now: datetime | None = None,
) -> RefitPlan:
    """Read the windowed live distribution and fit the served threshold, leaving the registry untouched.

    The watcher fit admits a boundary score inclusively while serving and the replay count fire strictly
    below the fitted threshold, so a row exactly at the threshold counts toward the fit budget but never
    fires in production (conservative, matching the retrain watcher lane).
    """
    window_days = _window_days(since)
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=window_days)).isoformat()
    rows = await _load_rows(cutoff, db_path)
    if not rows:
        raise RefitError(f"no scored moments at or after {cutoff} (window {since}); nothing to fit")
    parent = registry.current(component, root=root)
    if parent is None:
        raise RefitError(f"no promoted {component} version to re-fit; train and promote one first")
    match component:
        case "gate":
            threshold_key = gate.THRESHOLD_KEY
            fire_scores = np.asarray([row["gate_score"] for row in rows], dtype=np.float64)
            fitted = promotion.threshold_for_budget(fire_scores, fires_per_100=fires_per_100, total_turns=len(rows))
            n_eligible, passes = len(rows), int(np.count_nonzero(fire_scores >= fitted))
        case "watcher":
            threshold_key = WATCHER_THRESHOLD_KEY
            nosteer = np.asarray(
                [row["stage2_prob"] for row in rows if row["gate_passed"] and row["stage2_prob"] is not None],
                dtype=np.float64,
            )
            if nosteer.size == 0:
                raise RefitError(f"no gate-passed, stage-2-scored rows in window {since}; nothing to fit")
            fitted = 1.0 - promotion.threshold_for_budget(
                1.0 - nosteer, fires_per_100=fires_per_100, total_turns=len(rows)
            )
            n_eligible, passes = int(nosteer.size), int(np.count_nonzero(nosteer < fitted))
        case _:
            raise RefitError(f"unknown component {component!r}; expected gate or watcher")
    return RefitPlan(
        component=component,
        since=since,
        window_days=window_days,
        fires_per_100=fires_per_100,
        n_rows=len(rows),
        n_eligible=n_eligible,
        current_threshold=_current_threshold(parent, threshold_key),
        fitted_threshold=float(fitted),
        passes=passes,
        parent=parent,
        threshold_key=threshold_key,
    )


def apply_refit(plan: RefitPlan, *, root: Path | None = None, state_dir: Path | None = None) -> str:
    """Mint and promote a new version carrying the re-fit threshold, then kick the watch agent and journal."""
    files: dict[str, Path] = {
        path.name: path for path in plan.parent.path.iterdir() if path.is_file() and path.name != registry.METADATA_NAME
    }
    info = registry.register(plan.component, files, plan.metadata(), root=root)
    registry.promote(plan.component, info.version, root=root)
    kicked = launchd.kickstart_watch()
    verdict = (
        f"re-fit {plan.parent.version} -> {info.version}: {plan.threshold_key} "
        f"{plan.current_threshold:.6g} -> {plan.fitted_threshold:.6g} on {plan.n_rows} live rows over {plan.since} "
        f"({plan.passes} would fire, {plan.per_100:.2f}/100); watch kickstart {'ok' if kicked else 'skipped'}"
    )
    return promotion.journal(
        plan.component,
        verdict,
        dataset_digest=str(plan.parent.metadata.get("dataset_digest")),
        version=info.version,
        state_dir=state_dir,
    )


async def refit(
    component: str,
    *,
    since: str,
    fires_per_100: float = DEFAULT_FIRES_PER_100,
    dry_run: bool,
    db_path: Path | None = None,
    root: Path | None = None,
    state_dir: Path | None = None,
    now: datetime | None = None,
) -> str:
    """Re-fit ``component``'s served threshold from the last ``since`` of live scored moments.

    Dry-run returns the fitted threshold and a replay line without touching the registry; otherwise
    mints, promotes, and journals a new version copying the promoted artifact bytes verbatim.
    """
    plan = await plan_refit(component, since=since, fires_per_100=fires_per_100, db_path=db_path, root=root, now=now)
    return plan.report() if dry_run else apply_refit(plan, root=root, state_dir=state_dir)


def _window_days(since: str) -> int:
    if (match := WINDOW_PATTERN.match(since)) is None:
        raise RefitError(f"window must be <N>d (days), got {since!r}")
    return int(match.group(1))


async def _load_rows(cutoff: str, db_path: Path | None) -> list[dict[str, object]]:
    async with await ShadowDelivery.open(db_path) as delivery:
        return await delivery.scored_moments(since=cutoff)


def _current_threshold(parent: registry.VersionInfo, key: str) -> float:
    thresholds = parent.metadata.get("thresholds")
    if not isinstance(thresholds, dict) or key not in thresholds:
        raise RefitError(f"{parent.component} version {parent.version} metadata carries no thresholds[{key!r}]")
    return float(thresholds[key])
