from __future__ import annotations

import functools
import json
import math
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
from click.testing import CliRunner

from cc_steer import launchd, registry
from cc_steer.cli import main
from cc_steer.retrain import evalset, watcher_hosted
from cc_steer.retrain.evalset import EvalFrame
from cc_steer.watcher.cascade import DRAFT_SYSTEM
from cc_steer.watcher.drafter_http import TOP_LOGPROBS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.anyio

ENDPOINT = "https://watcher.modal.run"
MODEL = "watcher-9b"
DIGEST = "testdigest"
# 10 fire rows (low P(NO_STEER)), 8 keep rows (high), 2 below-top-k drops — AUC 1.0, scored 18 of 20.
GOOD_SPECS: list[tuple[str, bool, float | None]] = (
    [(f"fire{i}", True, 0.1) for i in range(10)]
    + [(f"keep{i}", False, 0.9) for i in range(8)]
    + [(f"belowk{i}", True, None) for i in range(2)]
)


def score_payload(top_logprobs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "NO_STEER"},
                "logprobs": {"content": [{"token": "NO", "logprob": -0.01, "top_logprobs": top_logprobs}]},
                "finish_reason": "length",
            }
        ]
    }


def frame_and_probs(specs: list[tuple[str, bool, float | None]]) -> tuple[EvalFrame, dict[str, float | None]]:
    frame = EvalFrame(
        ids=tuple(row_id for row_id, _, _ in specs),
        labels=np.array([label for _, label, _ in specs], dtype=bool),
        corrective=np.array([label for _, label, _ in specs], dtype=bool),
        prose=np.array([True] * len(specs), dtype=bool),
        tails=tuple(row_id for row_id, _, _ in specs),
        digest=DIGEST,
    )
    return frame, {row_id: prob for row_id, _, prob in specs}


def prob_handler(probs: dict[str, float | None]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        tail = next(msg["content"] for msg in json.loads(request.content)["messages"] if msg["role"] == "user")
        top = (
            [{"token": "Wait", "logprob": -0.1}]
            if (prob := probs[tail]) is None
            else [{"token": "NO", "logprob": math.log(prob)}]
        )
        return httpx.Response(200, json=score_payload(top))

    return handler


def inject(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    monkeypatch.setattr(
        watcher_hosted,
        "HttpDrafter",
        functools.partial(watcher_hosted.HttpDrafter, transport=httpx.MockTransport(handler)),
    )


def use_frame(monkeypatch: pytest.MonkeyPatch, frame: EvalFrame) -> None:
    monkeypatch.setattr(evalset.EvalFrame, "load", lambda *, root=None: frame)


def register_parent(root: Path) -> registry.VersionInfo:
    info = registry.register(
        "watcher",
        {"adapters.safetensors": b"ADAPTER", "adapter_config.json": b'{"r": 8}'},
        {
            "dataset_digest": "train-d0",
            "base_model": "mlx-community/Qwen3-8B-4bit",
            "render_version": 2,
            "thresholds": {"budget": 0.15, "f1": 0.5},
        },
        root=root,
    )
    registry.promote("watcher", info.version, root=root)
    return info


class TestScoreFrame:
    async def test_scores_align_by_row_id_and_drop_below_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, probs = frame_and_probs([("a", True, 0.8), ("b", False, None), ("c", True, 0.2), ("d", False, 0.55)])
        inject(monkeypatch, prob_handler(probs))
        result = await watcher_hosted.score_frame_http(frame, endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None)
        assert result == {"a": pytest.approx(0.8), "c": pytest.approx(0.2), "d": pytest.approx(0.55)}

    async def test_scoring_uses_the_local_scaffold_messages_and_logprob_knobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=score_payload([{"token": "NO", "logprob": math.log(0.9)}]))

        inject(monkeypatch, handler)
        frame = EvalFrame(
            ids=("r1",),
            labels=np.array([True]),
            corrective=np.array([True]),
            prose=np.array([True]),
            tails=("delete the prod table",),
            digest=DIGEST,
        )
        await watcher_hosted.score_frame_http(frame, endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None)
        assert seen["messages"] == [
            {"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user", "content": "delete the prod table"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
        ]
        assert seen["max_tokens"] == 1
        assert seen["top_logprobs"] == TOP_LOGPROBS and TOP_LOGPROBS <= 40
        assert seen["continue_final_message"] is True and seen["add_generation_prompt"] is False

    async def test_aborts_the_whole_pass_on_an_endpoint_fault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("endpoint unreachable")

        inject(monkeypatch, handler)
        frame, _ = frame_and_probs([("a", True, 0.8)])
        with pytest.raises(watcher_hosted.HostedCalibrationError, match="endpoint unreachable"):
            await watcher_hosted.score_frame_http(frame, endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None)

    async def test_forwards_the_api_key_as_a_bearer_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200, json=score_payload([{"token": "NO", "logprob": math.log(0.9)}]))

        inject(monkeypatch, handler)
        frame, _ = frame_and_probs([("a", True, 0.9)])
        result = await watcher_hosted.score_frame_http(
            frame, endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key="sk-test"
        )
        assert seen["auth"] == "Bearer sk-test"
        assert result == {"a": pytest.approx(0.9)}


class TestFit:
    def test_fit_threshold_inverts_to_fire_direction(self) -> None:
        probs = np.array([0.125, 0.25] + [round(0.90 + 0.001 * i, 3) for i in range(18)])
        fitted = watcher_hosted.fit_threshold(probs, fires_per_100=10.0, total_turns=20)
        assert fitted == 0.25
        # Served convention fires at p < threshold: only the low P(NO_STEER) row below the boundary.
        assert sorted(float(p) for p in probs if p < fitted) == [0.125]


class TestCalibrate:
    async def test_dry_run_scores_and_fits_but_mints_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = tmp_path / "models"
        register_parent(models)
        frame, probs = frame_and_probs(GOOD_SPECS)
        use_frame(monkeypatch, frame)
        inject(monkeypatch, prob_handler(probs))
        monkeypatch.setattr(launchd, "kickstart_watch", lambda: pytest.fail("dry-run must not kick the daemon"))
        report = await watcher_hosted.calibrate(
            endpoint=ENDPOINT,
            model=MODEL,
            timeout=5.0,
            api_key=None,
            dry_run=True,
            registry_root=models,
            state_dir=tmp_path / "state",
        )
        assert "fitted threshold:" in report
        assert "eval rows: 20 (scored 18, 2 below top-k)" in report
        assert "served eval sentinel AUC: 1.0000" in report
        assert registry.current("watcher-hosted", root=models) is None
        assert registry.versions("watcher-hosted", root=models) == []
        assert not (tmp_path / "state").exists()

    async def test_non_dry_mints_promotes_copies_bytes_and_journals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models, state = tmp_path / "models", tmp_path / "state"
        parent = register_parent(models)
        frame, probs = frame_and_probs(GOOD_SPECS)
        use_frame(monkeypatch, frame)
        inject(monkeypatch, prob_handler(probs))
        monkeypatch.setattr(launchd, "kickstart_watch", lambda: pytest.fail("hosted lane must not kick the daemon"))
        line = await watcher_hosted.calibrate(
            endpoint=ENDPOINT,
            model=MODEL,
            timeout=5.0,
            api_key=None,
            fires_per_100=2.0,
            dry_run=False,
            registry_root=models,
            state_dir=state,
        )
        # The local watcher lane is provably untouched: same promoted version, still one version.
        assert registry.current("watcher", root=models).version == parent.version
        assert len(registry.versions("watcher", root=models)) == 1
        # Distinct lane in its own directory: verbatim bytes reuse the digest-derived version string, which is fine.
        hosted = registry.current("watcher-hosted", root=models)
        assert hosted is not None and hosted.component == "watcher-hosted" and hosted.path != parent.path
        assert len(registry.versions("watcher-hosted", root=models)) == 1
        assert (hosted.path / "adapters.safetensors").read_bytes() == b"ADAPTER"
        assert (hosted.path / "adapter_config.json").read_bytes() == b'{"r": 8}'
        # The fitted budget threshold folds the 2 below-top-k rows in as guaranteed fires (P=0.0).
        expected = watcher_hosted.fit_threshold(
            np.array([0.1] * 10 + [0.9] * 8 + [0.0] * 2), fires_per_100=2.0, total_turns=20
        )
        assert hosted.metadata["thresholds"]["budget"] == pytest.approx(expected)
        assert hosted.metadata["thresholds"]["f1"] == 0.5
        assert hosted.metadata["base_model"] == "mlx-community/Qwen3-8B-4bit"
        provenance = hosted.metadata["hosted"]
        assert provenance["parent_version"] == parent.version
        assert (provenance["endpoint"], provenance["model"]) == (ENDPOINT, MODEL)
        assert (provenance["n_rows"], provenance["n_scored"], provenance["n_below_top_k"]) == (20, 18, 2)
        assert provenance["eval_auc"] == pytest.approx(1.0)
        assert provenance["eval_digest"] == DIGEST
        entries = [json.loads(row) for row in (state / "retrain" / "journal.jsonl").read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["component"] == "watcher-hosted" and entries[0]["version"] == hosted.version
        assert parent.version in entries[0]["verdict"] and hosted.version in entries[0]["verdict"]
        assert line == f"watcher-hosted: {entries[0]['verdict']}"


    async def test_below_top_k_drops_stay_within_budget(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        models = tmp_path / "models"
        register_parent(models)
        # budget = floor(25 * 20 / 100) = 5; 2 below-top-k rows fire unconditionally, leaving 3 for scored rows.
        frame, probs = frame_and_probs(
            [(f"fire{i}", True, round(0.05 * (i + 1), 2)) for i in range(10)]
            + [(f"keep{i}", False, 0.9) for i in range(8)]
            + [(f"belowk{i}", True, None) for i in range(2)]
        )
        use_frame(monkeypatch, frame)
        inject(monkeypatch, prob_handler(probs))
        plan = await watcher_hosted.plan_calibration(
            endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None, fires_per_100=25.0, registry_root=models
        )
        assert (plan.n_scored, plan.n_below_top_k) == (18, 2)
        assert plan.passes == 3
        # The load-bearing invariant: realized production fires (scored passes + forced below-top-k fires) stay in budget.
        assert plan.passes + plan.n_below_top_k <= 5
        assert plan.per_100 == pytest.approx(100.0 * (plan.passes + plan.n_below_top_k) / 20)
        report = plan.report()
        assert "3 of 18 scored rows fire" in report
        assert f"{plan.passes + plan.n_below_top_k} fire in production" in report


class TestRefusals:
    async def test_refuses_when_no_local_watcher_is_promoted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, _ = frame_and_probs(GOOD_SPECS)
        use_frame(monkeypatch, frame)
        with pytest.raises(watcher_hosted.HostedCalibrationError, match="no promoted watcher"):
            await watcher_hosted.plan_calibration(
                endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None, registry_root=tmp_path / "models"
            )

    async def test_refuses_a_below_chance_frame(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        models = tmp_path / "models"
        register_parent(models)
        # Inverted signal: fire rows carry HIGH P(NO_STEER), keep rows LOW — sentinel AUC below chance.
        frame, probs = frame_and_probs(
            [(f"fire{i}", True, 0.9) for i in range(10)] + [(f"keep{i}", False, 0.1) for i in range(10)]
        )
        use_frame(monkeypatch, frame)
        inject(monkeypatch, prob_handler(probs))
        with pytest.raises(watcher_hosted.HostedCalibrationError, match="above chance"):
            await watcher_hosted.plan_calibration(
                endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None, registry_root=models
            )

    async def test_refuses_when_the_endpoint_scores_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = tmp_path / "models"
        register_parent(models)
        frame, probs = frame_and_probs([(f"r{i}", i % 2 == 0, None) for i in range(6)])
        use_frame(monkeypatch, frame)
        inject(monkeypatch, prob_handler(probs))
        with pytest.raises(watcher_hosted.HostedCalibrationError, match="nothing to fit"):
            await watcher_hosted.plan_calibration(
                endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None, registry_root=models
            )


@pytest.mark.integration
def test_cli_hosted_calibrate_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = tmp_path / "models"
    register_parent(models)
    frame, probs = frame_and_probs(GOOD_SPECS)
    use_frame(monkeypatch, frame)
    inject(monkeypatch, prob_handler(probs))
    monkeypatch.setenv("CC_STEER_MODELS", str(models))
    result = CliRunner().invoke(main, ["hosted", "calibrate", "--endpoint", ENDPOINT, "--model", MODEL, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "fitted threshold:" in result.output
    assert "scored 18, 2 below top-k" in result.output
    assert registry.versions("watcher-hosted", root=models) == []


@pytest.mark.integration
def test_cli_hosted_calibrate_forwards_api_key_env_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = tmp_path / "models"
    register_parent(models)
    frame, probs = frame_and_probs(GOOD_SPECS)
    use_frame(monkeypatch, frame)
    seen: dict[str, str] = {}
    score = prob_handler(probs)

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        return score(request)

    inject(monkeypatch, handler)
    monkeypatch.setenv("CC_STEER_MODELS", str(models))
    monkeypatch.setenv("HOSTED_TEST_KEY", "sk-cli")
    result = CliRunner().invoke(
        main,
        ["hosted", "calibrate", "--endpoint", ENDPOINT, "--model", MODEL, "--api-key-env", "HOSTED_TEST_KEY", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert seen["auth"] == "Bearer sk-cli"


@pytest.mark.integration
def test_cli_hosted_calibrate_rejects_negative_fires_per_100() -> None:
    result = CliRunner().invoke(
        main,
        ["hosted", "calibrate", "--endpoint", ENDPOINT, "--model", MODEL, "--fires-per-100", "-1", "--dry-run"],
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.output and "--fires-per-100" in result.output
