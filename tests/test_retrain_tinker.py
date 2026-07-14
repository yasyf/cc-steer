from __future__ import annotations

import math
import shutil
import sys
import tarfile
import urllib.request
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from cc_steer.retrain import tinker as tk
from cc_steer.retrain.tinker import (
    QWEN3_8B,
    InsufficientDatumsError,
    NonFiniteAUCError,
    SpendCapError,
    TinkerRun,
    build_datum,
    convert_peft_to_mlx,
    download_adapter,
    score_auc_tinker,
    score_frame_tinker,
    train_lora,
)

if TYPE_CHECKING:
    from pathlib import Path


class Deferred:
    def __init__(self, value: Any) -> None:
        self.value = value

    def result(self) -> Any:
        return self.value


class FakeModelInput:
    def __init__(self, ints: list[int]) -> None:
        self.ints = list(ints)

    @property
    def length(self) -> int:
        return len(self.ints)

    @classmethod
    def from_ints(cls, ints: list[int]) -> FakeModelInput:
        return cls(ints)


class FakeTensorData:
    def __init__(self, *, data: list[float], dtype: str, shape: list[int]) -> None:
        self.data = list(data)
        self.dtype = dtype
        self.shape = shape

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data, dtype=np.float64)


class FakeDatum:
    def __init__(self, *, model_input: FakeModelInput, loss_fn_inputs: dict[str, FakeTensorData]) -> None:
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class FakeAdamParams:
    def __init__(self, *, learning_rate: float) -> None:
        self.learning_rate = learning_rate


class FakeTrainingClient:
    def __init__(self) -> None:
        self.fb_calls = 0
        self.saved: list[str] = []

    def forward_backward(self, batch: list[FakeDatum], loss_fn: str) -> Deferred:
        self.fb_calls += 1
        outputs = [
            {
                "logprobs": FakeTensorData(
                    data=[-0.1] * d.model_input.length, dtype="float32", shape=[d.model_input.length]
                )
            }
            for d in batch
        ]
        return Deferred(SimpleNamespace(loss_fn_outputs=outputs))

    def optim_step(self, params: FakeAdamParams) -> Deferred:
        return Deferred(None)

    def save_weights_for_sampler(self, *, name: str) -> Deferred:
        self.saved.append(name)
        return Deferred(SimpleNamespace(path=f"tinker://ckpt/{name}"))


class FakeSamplingClient:
    def __init__(self, model_path: str, sentinel_logprob: float) -> None:
        self.model_path = model_path
        self.sentinel_logprob = sentinel_logprob

    def compute_logprobs(self, *, prompt: FakeModelInput) -> Deferred:
        return Deferred(np.array([-2.0, -2.0, self.sentinel_logprob]))


class FakeServiceClient:
    instances: list[FakeServiceClient] = []
    sentinel_logprob = math.log(0.3)

    def __init__(self) -> None:
        FakeServiceClient.instances.append(self)
        self.training_client: FakeTrainingClient | None = None
        self.lora_kwargs: dict[str, Any] = {}

    def create_lora_training_client(self, **kwargs: Any) -> FakeTrainingClient:
        self.lora_kwargs = kwargs
        self.training_client = FakeTrainingClient()
        return self.training_client

    def create_sampling_client(self, *, model_path: str) -> FakeSamplingClient:
        return FakeSamplingClient(model_path, FakeServiceClient.sentinel_logprob)


class StubTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        parts = [f"{m['role']}:{m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


def fake_tinker_module() -> ModuleType:
    module = ModuleType("tinker")
    module.Datum = FakeDatum  # type: ignore[attr-defined]
    module.ModelInput = FakeModelInput  # type: ignore[attr-defined]
    module.TensorData = FakeTensorData  # type: ignore[attr-defined]
    module.AdamParams = FakeAdamParams  # type: ignore[attr-defined]
    module.ServiceClient = FakeServiceClient  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_tinker(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    FakeServiceClient.instances = []
    module = fake_tinker_module()
    monkeypatch.setitem(sys.modules, "tinker", module)
    return module


@pytest.fixture
def stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda mlx_id: StubTokenizer())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    tk.TOKENIZERS.clear()


@pytest.fixture
def tinker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-key")


def datum(tinker: ModuleType, length: int) -> FakeDatum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(length))),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=[1.0] * length, dtype="float32", shape=[length]),
            "target_tokens": tinker.TensorData(data=list(range(length)), dtype="int64", shape=[length]),
        },
    )


class TestBuildDatum:
    def test_boundary_masks_only_the_completion(self, fake_tinker: ModuleType, stub_tokenizer: None) -> None:
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "GO"},
        ]
        full = [ord(c) for c in "system:S\nuser:U\nassistant:GO\n"]
        boundary = len("system:S\nuser:U\nassistant:")
        result = build_datum(messages, "stub")
        assert result.model_input.length == len(full) - 1
        assert list(result.loss_fn_inputs["target_tokens"].data) == full[1:]
        weights = result.loss_fn_inputs["weights"].to_numpy()
        assert weights.sum() == 3.0  # the completion "GO\n" is 3 chars
        assert weights[: boundary - 1].tolist() == [0.0] * (boundary - 1)
        assert weights[boundary - 1 :].tolist() == [1.0, 1.0, 1.0]


class TestTrainLoraSpendGuards:
    def test_pre_launch_abort_projects_full_cycled_cost_and_never_constructs_client(
        self, fake_tinker: ModuleType
    ) -> None:
        # Full cycled cost = 5 steps x 2 slots x 1000 tokens = 10,000 tokens = $0.004; one pass alone
        # is only $0.0008, so a $0.002 cap catches the run only under the full-cycle projection.
        datums = [datum(fake_tinker, 1000), datum(fake_tinker, 1000)]
        with pytest.raises(SpendCapError, match="projected cost"):
            train_lora(
                datums, QWEN3_8B, steps=5, batch_size=2, learning_rate=1e-4, rank=16, seed=1,
                spend_cap_usd=0.002, max_tokens=100_000,
            )
        assert FakeServiceClient.instances == []  # no client, so zero paid calls

    def test_train_cost_usd_tracks_token_price(self) -> None:
        run = TinkerRun(
            base_model=QWEN3_8B.tinker_model, steps=10, batch_size=2, learning_rate=1e-4,
            rank=16, seed=1, train_unembed=False,
        )
        run.train_tokens = 200_000_000  # 200M tokens at $0.40/1M
        assert run.train_cost_usd == pytest.approx(80.0)


class TestTrainLoraMaxTokens:
    def test_over_length_datums_dropped_and_reported(self, fake_tinker: ModuleType, tinker_env: None) -> None:
        datums = [datum(fake_tinker, 50), datum(fake_tinker, 50), datum(fake_tinker, 5000)]  # third is over max
        run = train_lora(
            datums, QWEN3_8B, steps=2, batch_size=2, learning_rate=1e-4, rank=16, seed=1,
            spend_cap_usd=1000.0, max_tokens=1000,
        )
        assert run.dropped == 1
        assert run.as_dict()["dropped"] == 1
        assert run.train_tokens == 200  # 2 steps x 2 slots x 50 tokens; the 5000-token datum never trained

    def test_empty_pool_after_filter_raises_before_client(self, fake_tinker: ModuleType) -> None:
        datums = [datum(fake_tinker, 5000), datum(fake_tinker, 6000)]  # all over max
        with pytest.raises(InsufficientDatumsError, match="max_tokens"):
            train_lora(
                datums, QWEN3_8B, steps=2, batch_size=2, learning_rate=1e-4, rank=16, seed=1,
                spend_cap_usd=1000.0, max_tokens=100,
            )
        assert FakeServiceClient.instances == []

    def test_pool_smaller_than_one_batch_raises_before_client(self, fake_tinker: ModuleType) -> None:
        datums = [datum(fake_tinker, 50), datum(fake_tinker, 5000)]  # one survivor, batch needs two
        with pytest.raises(InsufficientDatumsError, match="fewer than one batch of 2"):
            train_lora(
                datums, QWEN3_8B, steps=2, batch_size=2, learning_rate=1e-4, rank=16, seed=1,
                spend_cap_usd=1000.0, max_tokens=1000,
            )
        assert FakeServiceClient.instances == []


class TestTrainLoraCheckpoints:
    def test_callback_fires_at_each_fraction(self, fake_tinker: ModuleType, tinker_env: None) -> None:
        calls: list[tuple[int, str, Any, Any]] = []
        run = train_lora(
            [datum(fake_tinker, 50), datum(fake_tinker, 50)],
            QWEN3_8B,
            steps=2,
            batch_size=2,
            learning_rate=1e-4,
            rank=16,
            seed=1,
            spend_cap_usd=1000.0,
            max_tokens=100_000,
            checkpoint_fractions=(0.5, 1.0),
            on_checkpoint=lambda step, path, svc, base: calls.append((step, path, svc, base)),
        )
        svc = FakeServiceClient.instances[0]
        assert run.checkpoints == {1: "tinker://ckpt/step1", 2: "tinker://ckpt/step2"}
        assert calls == [(1, "tinker://ckpt/step1", svc, QWEN3_8B), (2, "tinker://ckpt/step2", svc, QWEN3_8B)]
        assert svc.training_client.saved == ["step1", "step2"]


class TestConvertPeftToMlx:
    def install_mlx(self, monkeypatch: pytest.MonkeyPatch, weights: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        saved: list[tuple[str, dict[str, Any]]] = []
        core = ModuleType("mlx.core")
        core.load = lambda path: weights  # type: ignore[attr-defined]
        core.save_safetensors = lambda path, tensors: saved.append((path, dict(tensors)))  # type: ignore[attr-defined]
        mlx = ModuleType("mlx")
        mlx.core = core  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx", mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", core)
        return saved

    def std_weights(self) -> dict[str, FakeArray]:
        return {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": FakeArray("q_a"),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": FakeArray("q_b"),
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": FakeArray("d_a"),
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": FakeArray("d_b"),
        }

    def test_standard_modules_drop_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.install_mlx(monkeypatch, self.std_weights())
        report = convert_peft_to_mlx(tmp_path / "peft", tmp_path / "mlx", num_layers=36, rank=16)
        assert report["dropped"] == []
        assert report["covered"] == ["mlp.down_proj", "self_attn.q_proj"]
        assert report["n_lora_weights"] == 4
        assert (tmp_path / "mlx" / "adapter_config.json").exists()

    def test_non_standard_modules_are_reported_dropped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        weights = self.std_weights() | {
            "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": FakeArray("x"),
            "base_model.model.lm_head.weight": FakeArray("y"),
        }
        self.install_mlx(monkeypatch, weights)
        report = convert_peft_to_mlx(tmp_path / "peft", tmp_path / "mlx", num_layers=36, rank=16)
        assert report["dropped"] == ["linear_attn.in_proj_qkv", "lm_head.weight"]
        assert report["n_lora_weights"] == 4


class FakeArray:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def T(self) -> FakeArray:
        return FakeArray(f"{self.name}.T")


class TestScoreFrameTinker:
    def test_scores_every_row_via_compute_logprobs(
        self, fake_tinker: ModuleType, stub_tokenizer: None
    ) -> None:
        FakeServiceClient.instances = []
        frame = SimpleNamespace(ids=("a", "b", "c"), tails=("tail a", "tail b", "tail c"))
        probs = score_frame_tinker(FakeServiceClient(), "tinker://ckpt/step2", frame, base=QWEN3_8B)
        assert set(probs) == {"a", "b", "c"}
        assert probs == {"a": pytest.approx(0.3), "b": pytest.approx(0.3), "c": pytest.approx(0.3)}


class TestScoreAucTinker:
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_single_class_val_raises_on_non_finite_auc(
        self, fake_tinker: ModuleType, stub_tokenizer: None
    ) -> None:
        # Every row is a true steer, so the validation labels are single-class and sklearn's AUC is NaN.
        rows = [
            {"system": "S", "user": "U0", "assistant": "steer left"},
            {"system": "S", "user": "U1", "assistant": "steer right"},
        ]
        with pytest.raises(NonFiniteAUCError, match="single-class"):
            score_auc_tinker(FakeServiceClient(), "tinker://ckpt/step1", rows, "stub")


class TestDownloadAdapter:
    def test_extractall_uses_data_filter_blocking_traversal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "payload.txt").write_text("x")
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(tmp_path / "payload.txt", arcname="../escape.txt")  # a path-traversal member
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: shutil.copy(archive, dest))
        rest = SimpleNamespace(
            get_checkpoint_archive_url_from_tinker_path=lambda path: Deferred(SimpleNamespace(url="http://x/evil.tar"))
        )
        service_client = SimpleNamespace(create_rest_client=lambda: rest)
        with pytest.raises(tarfile.FilterError):  # the data filter refuses to write outside out_dir
            download_adapter(service_client, "tinker://ckpt/step1", tmp_path / "out")
