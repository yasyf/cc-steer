from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from athome.train.gate import threshold_for_budget
from click.testing import CliRunner

from cc_steer import launchd, registry
from cc_steer.cli import main
from cc_steer.retrain import refit as refit_mod
from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.types import ScoredMoment

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
IN_WINDOW = "2026-07-19T00:00:00+00:00"


def gate_row(turn: int, score: float, *, ts: str = IN_WINDOW) -> ScoredMoment:
    return ScoredMoment(
        session_id="s",
        turn_index=turn,
        ts=ts,
        gate_score=score,
        gate_threshold=0.5,
        gate_passed=score >= 0.5,
    )


def watcher_row(turn: int, prob: float | None, *, gate_passed: bool = True, ts: str = IN_WINDOW) -> ScoredMoment:
    return ScoredMoment(
        session_id="s",
        turn_index=turn,
        ts=ts,
        gate_score=0.9,
        gate_threshold=0.5,
        gate_passed=gate_passed,
        stage2_prob=prob,
        stage2_threshold=0.5 if prob is not None else None,
    )


async def seed(db: Path, rows: list[ScoredMoment]) -> None:
    async with await ShadowDelivery.open(db) as delivery:
        for row in rows:
            await delivery.record_scored(row)


def register_parent(
    root: Path,
    component: str,
    *,
    files: dict[str, bytes],
    thresholds: dict[str, float],
    extra: dict[str, object] | None = None,
) -> registry.VersionInfo:
    info = registry.register(
        component, files, {"dataset_digest": "d0", "thresholds": thresholds, **(extra or {})}, root=root
    )
    registry.promote(component, info.version, root=root)
    return info


class TestFit:
    async def test_gate_fits_the_live_distribution(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(db, [gate_row(i, i / 100) for i in range(100)])
        register_parent(models, "gate", files={"model.joblib": b"W"}, thresholds={"budget_2_per_100": 0.9985})
        plan = await refit_mod.plan_refit("gate", since="30d", db_path=db, root=models, now=NOW)
        assert (plan.n_rows, plan.n_eligible, plan.window_days) == (100, 100, 30)
        assert plan.threshold_key == "budget_2_per_100"
        assert plan.current_threshold == 0.9985
        assert plan.fitted_threshold == pytest.approx(0.98)
        assert plan.passes == 2

    async def test_watcher_fit_is_fire_direction_not_abstain_direction(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        probs = [0.125, 0.25] + [round(0.90 + 0.001 * i, 3) for i in range(18)]
        await seed(db, [watcher_row(i, p) for i, p in enumerate(probs)])
        register_parent(
            models, "watcher", files={"adapters.safetensors": b"A"}, thresholds={"budget": 7e-05, "f1": 0.5}
        )
        plan = await refit_mod.plan_refit("watcher", since="30d", fires_per_100=10.0, db_path=db, root=models, now=NOW)
        assert (plan.n_rows, plan.n_eligible) == (20, 20)
        assert plan.threshold_key == "budget"
        assert plan.fitted_threshold == 0.25
        # The boundary row p=0.25 is admitted by the inclusive fit but excluded by the strict `<` serve/replay.
        assert plan.passes == 1
        # Served convention fires at p < threshold: only the LOW P(NO_STEER) row below the boundary.
        assert sorted(p for p in probs if p < plan.fitted_threshold) == [0.125]
        # A fit that mistook stored P(NO_STEER) for a fire score would fire the opposite, HIGH rows.
        naive = threshold_for_budget(np.asarray(probs), fires_per_100=10.0, total_turns=20)
        naive_fired = sorted(p for p in probs if p >= naive)
        assert naive_fired == [0.916, 0.917]
        assert set(p for p in probs if p < plan.fitted_threshold).isdisjoint(naive_fired)

    async def test_watcher_ranks_only_eligible_rows_over_the_full_budget(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        rows = (
            [watcher_row(i, 0.01 + 0.001 * i) for i in range(4)]  # gate-passed, stage-2 scored: eligible
            + [watcher_row(10 + i, None) for i in range(3)]  # gate-passed but stage 2 not yet done
            + [watcher_row(20 + i, None, gate_passed=False) for i in range(5)]  # gate-suppressed
        )
        await seed(db, rows)
        register_parent(models, "watcher", files={"adapters.safetensors": b"A"}, thresholds={"budget": 7e-05})
        plan = await refit_mod.plan_refit("watcher", since="30d", db_path=db, root=models, now=NOW)
        assert (plan.n_rows, plan.n_eligible) == (12, 4)

    async def test_window_filters_rows_by_timestamp(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(
            db,
            [gate_row(i, i / 100) for i in range(50)]
            + [gate_row(500 + i, i / 100, ts="2026-01-01T00:00:00+00:00") for i in range(50)],
        )
        register_parent(models, "gate", files={"model.joblib": b"W"}, thresholds={"budget_2_per_100": 0.5})
        plan = await refit_mod.plan_refit("gate", since="7d", db_path=db, root=models, now=NOW)
        assert plan.n_rows == 50


class TestRefusals:
    async def test_refuses_empty_window(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(db, [gate_row(0, 0.9, ts="2026-01-01T00:00:00+00:00")])
        register_parent(models, "gate", files={"model.joblib": b"W"}, thresholds={"budget_2_per_100": 0.5})
        with pytest.raises(refit_mod.RefitError, match="nothing to fit"):
            await refit_mod.plan_refit("gate", since="7d", db_path=db, root=models, now=NOW)

    async def test_refuses_watcher_with_no_eligible_rows(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(db, [watcher_row(i, None, gate_passed=False) for i in range(5)])
        register_parent(models, "watcher", files={"adapters.safetensors": b"A"}, thresholds={"budget": 7e-05})
        with pytest.raises(refit_mod.RefitError, match="no gate-passed"):
            await refit_mod.plan_refit("watcher", since="30d", db_path=db, root=models, now=NOW)

    async def test_refuses_when_nothing_promoted(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(db, [gate_row(i, i / 100) for i in range(10)])
        with pytest.raises(refit_mod.RefitError, match="no promoted gate"):
            await refit_mod.plan_refit("gate", since="30d", db_path=db, root=models, now=NOW)

    async def test_refuses_malformed_window(self, tmp_path: Path) -> None:
        with pytest.raises(refit_mod.RefitError, match="window must be"):
            await refit_mod.plan_refit("gate", since="7", db_path=tmp_path / "shadow.db", root=tmp_path / "m", now=NOW)

    async def test_refuses_missing_threshold_key(self, tmp_path: Path) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(db, [gate_row(i, i / 100) for i in range(10)])
        register_parent(models, "gate", files={"model.joblib": b"W"}, thresholds={"other": 0.5})
        with pytest.raises(refit_mod.RefitError, match="no thresholds"):
            await refit_mod.plan_refit("gate", since="30d", db_path=db, root=models, now=NOW)


class TestApply:
    async def test_dry_run_mints_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db, models = tmp_path / "shadow.db", tmp_path / "models"
        await seed(db, [gate_row(i, i / 100) for i in range(100)])
        register_parent(models, "gate", files={"model.joblib": b"W"}, thresholds={"budget_2_per_100": 0.9985})
        monkeypatch.setattr(launchd, "kickstart_watch", lambda: pytest.fail("dry-run must not kickstart"))
        report = await refit_mod.refit(
            "gate", since="30d", dry_run=True, db_path=db, root=models, state_dir=tmp_path / "state", now=NOW
        )
        assert "fitted threshold: 0.98" in report
        assert "2 of 100 live rows would pass (2.00 per 100)" in report
        assert len(registry.versions("gate", root=models)) == 1
        assert not (tmp_path / "state").exists()

    async def test_non_dry_mints_promotes_copies_bytes_and_journals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, models, state = tmp_path / "shadow.db", tmp_path / "models", tmp_path / "state"
        await seed(db, [gate_row(i, i / 100) for i in range(100)])
        parent = register_parent(
            models, "gate", files={"model.joblib": b"WEIGHTS"}, thresholds={"budget_2_per_100": 0.9985}
        )
        kicks: list[bool] = []
        monkeypatch.setattr(launchd, "kickstart_watch", lambda: kicks.append(True) or True)
        line = await refit_mod.refit(
            "gate", since="30d", dry_run=False, db_path=db, root=models, state_dir=state, now=NOW
        )
        assert kicks == [True]
        promoted = registry.current("gate", root=models)
        assert promoted is not None and promoted.version != parent.version
        assert len(registry.versions("gate", root=models)) == 2
        assert (promoted.path / "model.joblib").read_bytes() == b"WEIGHTS"
        assert promoted.metadata["thresholds"] == {"budget_2_per_100": pytest.approx(0.98)}
        assert promoted.metadata["refit"] == {
            "parent_version": parent.version,
            "window_days": 30,
            "n_rows": 100,
            "fires_per_100": 2.0,
        }
        assert promoted.metadata["dataset_digest"] == "d0"
        entries = [json.loads(row) for row in (state / "retrain" / "journal.jsonl").read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["component"] == "gate" and entries[0]["version"] == promoted.version
        assert parent.version in entries[0]["verdict"] and promoted.version in entries[0]["verdict"]
        assert "0.9985" in entries[0]["verdict"] and "0.98" in entries[0]["verdict"]
        assert line == f"gate: {entries[0]['verdict']}"

    async def test_watcher_non_dry_updates_budget_and_copies_every_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, models, state = tmp_path / "shadow.db", tmp_path / "models", tmp_path / "state"
        probs = [0.01, 0.02] + [round(0.90 + 0.001 * i, 3) for i in range(18)]
        await seed(db, [watcher_row(i, p) for i, p in enumerate(probs)])
        parent = register_parent(
            models,
            "watcher",
            files={"adapters.safetensors": b"ADAPTER", "adapter_config.json": b'{"r": 8}'},
            thresholds={"budget": 7e-05, "f1": 0.5},
        )
        kicks: list[bool] = []
        monkeypatch.setattr(launchd, "kickstart_watch", lambda: kicks.append(True) or True)
        await refit_mod.refit(
            "watcher", since="30d", fires_per_100=10.0, dry_run=False, db_path=db, root=models, state_dir=state, now=NOW
        )
        assert kicks == [True]
        promoted = registry.current("watcher", root=models)
        assert promoted is not None and promoted.version != parent.version
        assert (promoted.path / "adapters.safetensors").read_bytes() == b"ADAPTER"
        assert (promoted.path / "adapter_config.json").read_bytes() == b'{"r": 8}'
        assert promoted.metadata["thresholds"]["f1"] == 0.5
        assert promoted.metadata["thresholds"]["budget"] == pytest.approx(0.02, abs=1e-9)


@pytest.mark.integration
def test_cli_thresholds_refit_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db, models = tmp_path / "shadow.db", tmp_path / "models"
    ts = datetime.now(UTC).isoformat()

    async def prepare() -> None:
        await seed(db, [gate_row(i, i / 100, ts=ts) for i in range(100)])

    import anyio

    anyio.run(prepare)
    register_parent(models, "gate", files={"model.joblib": b"W"}, thresholds={"budget_2_per_100": 0.9985})
    monkeypatch.setenv("CC_STEER_MODELS", str(models))
    result = CliRunner().invoke(
        main, ["thresholds", "refit", "--component", "gate", "--since", "3650d", "--dry-run", "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "fitted threshold: 0.98" in result.output
    assert len(registry.versions("gate", root=models)) == 1


@pytest.mark.integration
def test_cli_thresholds_refit_rejects_negative_fires_per_100() -> None:
    result = CliRunner().invoke(
        main, ["thresholds", "refit", "--component", "gate", "--since", "7d", "--fires-per-100", "-1", "--dry-run"]
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.output and "--fires-per-100" in result.output
    assert not isinstance(result.exception, ValueError)
