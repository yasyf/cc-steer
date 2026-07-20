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

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.anyio

ENDPOINT = "https://watcher.modal.run"
MODEL = "watcher-9b"
DIGEST = "testdigest"
# 10 fire rows (low P(NO_STEER)), 10 keep rows (high) — AUC 1.0, all 20 scored on the exact path.
GOOD_SPECS: list[tuple[str, bool, float]] = [(f"fire{i}", True, 0.1) for i in range(10)] + [
    (f"keep{i}", False, 0.9) for i in range(10)
]


class StubTokenizer:
    """A byte-per-char stand-in for the base tokenizer: chat template is ``role:content`` lines, encode is ``ord``."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        return "".join(f"{m['role']}:{m['content']}\n" for m in messages)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


def completions_payload(sentinel: int, logprob: float) -> dict[str, object]:
    return {"choices": [{"prompt_logprobs": [None, {str(sentinel): {"logprob": logprob}}]}]}


def frame_and_probs(specs: list[tuple[str, bool, float]]) -> tuple[EvalFrame, dict[str, float]]:
    frame = EvalFrame(
        ids=tuple(row_id for row_id, _, _ in specs),
        labels=np.array([label for _, label, _ in specs], dtype=bool),
        corrective=np.array([label for _, label, _ in specs], dtype=bool),
        prose=np.array([True] * len(specs), dtype=bool),
        tails=tuple(row_id for row_id, _, _ in specs),
        digest=DIGEST,
    )
    return frame, {row_id: prob for row_id, _, prob in specs}


def prob_handler(probs: dict[str, float]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["prompt"]
        tail = "".join(chr(i) for i in prompt[:-1]).rsplit("\nassistant:", 1)[0].split("\nuser:")[-1]
        return httpx.Response(200, json=completions_payload(prompt[-1], math.log(probs[tail])))

    return handler


def inject(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    monkeypatch.setattr(
        watcher_hosted,
        "HttpDrafter",
        functools.partial(
            watcher_hosted.HttpDrafter, transport=httpx.MockTransport(handler), tokenizer=StubTokenizer()
        ),
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
    async def test_scores_align_by_row_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, probs = frame_and_probs([("a", True, 0.8), ("b", False, 0.4), ("c", True, 0.2), ("d", False, 0.55)])
        inject(monkeypatch, prob_handler(probs))
        result = await watcher_hosted.score_frame_http(frame, endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None)
        assert result == {
            "a": pytest.approx(0.8),
            "b": pytest.approx(0.4),
            "c": pytest.approx(0.2),
            "d": pytest.approx(0.55),
        }

    async def test_scoring_posts_teacher_forced_ids_to_the_completions_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.update(body)
            seen["path"] = request.url.path
            return httpx.Response(200, json=completions_payload(body["prompt"][-1], math.log(0.9)))

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
        assert seen["path"] == "/v1/completions"
        assert seen["model"] == MODEL
        assert seen["prompt"][-1] == ord("N")
        assert seen["max_tokens"] == 1
        assert seen["temperature"] == 0.0
        assert seen["prompt_logprobs"] == 0
        assert "messages" not in seen

    async def test_aborts_the_whole_pass_on_an_endpoint_fault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("endpoint unreachable")

        inject(monkeypatch, handler)
        frame, _ = frame_and_probs([("a", True, 0.8)])
        with pytest.raises(watcher_hosted.HostedCalibrationError, match="endpoint unreachable"):
            await watcher_hosted.score_frame_http(frame, endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None)

    async def test_forwards_the_api_key_as_a_bearer_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}
        score = prob_handler({"a": 0.9})

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization", "")
            return score(request)

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
        assert "eval rows: 20" in report
        assert "below top-k" not in report
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
        # The fitted budget threshold fits every row directly — no forced-fire folding.
        expected = watcher_hosted.fit_threshold(np.array([0.1] * 10 + [0.9] * 10), fires_per_100=2.0, total_turns=20)
        assert hosted.metadata["thresholds"]["budget"] == pytest.approx(expected)
        assert hosted.metadata["thresholds"]["f1"] == 0.5
        assert hosted.metadata["base_model"] == "mlx-community/Qwen3-8B-4bit"
        provenance = hosted.metadata["hosted"]
        assert provenance["parent_version"] == parent.version
        assert (provenance["endpoint"], provenance["model"]) == (ENDPOINT, MODEL)
        assert provenance["n_rows"] == 20
        assert "n_scored" not in provenance and "n_below_top_k" not in provenance
        assert provenance["eval_auc"] == pytest.approx(1.0)
        assert provenance["eval_digest"] == DIGEST
        entries = [json.loads(row) for row in (state / "retrain" / "journal.jsonl").read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["component"] == "watcher-hosted" and entries[0]["version"] == hosted.version
        assert parent.version in entries[0]["verdict"] and hosted.version in entries[0]["verdict"]
        assert line == f"watcher-hosted: {entries[0]['verdict']}"

    async def test_fitted_threshold_holds_fires_within_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = tmp_path / "models"
        register_parent(models)
        # budget = floor(25 * 20 / 100) = 5; the 5 lowest-P(NO_STEER) rows fire, the rest abstain.
        frame, probs = frame_and_probs(
            [(f"fire{i}", True, round(0.05 * (i + 1), 2)) for i in range(10)]
            + [(f"keep{i}", False, 0.9) for i in range(10)]
        )
        use_frame(monkeypatch, frame)
        inject(monkeypatch, prob_handler(probs))
        plan = await watcher_hosted.plan_calibration(
            endpoint=ENDPOINT, model=MODEL, timeout=5.0, api_key=None, fires_per_100=25.0, registry_root=models
        )
        assert plan.n_rows == 20
        # The load-bearing invariant: realized production fires stay within the floor(25 * 20 / 100) = 5 budget.
        assert plan.passes <= 5
        assert plan.per_100 == pytest.approx(100.0 * plan.passes / 20)
        assert f"{plan.passes} of 20 rows fire in production" in plan.report()


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
    assert "eval rows: 20" in result.output
    assert "below top-k" not in result.output
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
        [
            "hosted",
            "calibrate",
            "--endpoint",
            ENDPOINT,
            "--model",
            MODEL,
            "--api-key-env",
            "HOSTED_TEST_KEY",
            "--dry-run",
        ],
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
