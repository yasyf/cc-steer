"""LoRA training on Thinking Machines' managed Tinker API, under a hard spend cap.

Serving is always local 4-bit MLX, so the loop trains a LoRA on Tinker, converts the
PEFT adapter to an mlx-lm adapter, and scores sentinel AUC — the true gate metric —
against the frozen eval. Only the ``tinker`` SDK is a dependency: tinker-cookbook is
banned (its ``transformers`` pin collides with mlx-lm's), so :func:`build_datum` is
the raw-SDK twin of ``datum_from_model_input_weights`` and rendering goes through the
local tokenizer with thinking disabled, so token ids match Tinker's server tokenizer.

Every ``tinker`` / ``transformers`` / ``mlx`` import is lazy, so the module imports
without those heavy, optional dependencies installed.

:func:`train_lora` guards spend on both sides of launch: it drops over-length datums,
projects the full cycled training cost, and refuses to start when that busts the cap,
then tracks the true cycled spend per step and aborts the moment it crosses the cap.
Checkpoints save at configurable fractions of the run, each firing an optional callback
so a caller can score mid-run.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cc_steer.rendering import NO_STEER
from cc_steer.watcher.cascade import DRAFT_SYSTEM

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import tinker
    from transformers import PreTrainedTokenizerBase

    from cc_steer.retrain.evalset import EvalFrame

TINKER_ENV: Path = Path.home() / ".cc-steer" / "tinker.env"
SEED = 1729
SPEND_CAP_USD = 60.0

# Tinker training price (USD / 1M tokens), from thinkingmachines.ai/tinker.
PRICE_TRAIN = {"Qwen/Qwen3-8B": 0.40}

# The standard-attention modules that map 1:1 into mlx-lm's qwen3 LoRA layers.
STD_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

PEFT_KEY = re.compile(r"base_model\.model\.(model\.layers\.\d+\.(.+?))\.lora_([AB])\.weight")
TOKENIZERS: dict[str, PreTrainedTokenizerBase] = {}


class SpendCapError(RuntimeError):
    """A training run's projected cost exceeds its spend cap."""


class InsufficientDatumsError(RuntimeError):
    """Fewer datums survive the ``max_tokens`` filter than one training batch needs."""


class NonFiniteAUCError(RuntimeError):
    """A scored AUC is non-finite (a single-class validation set), so it cannot rank checkpoints."""


@dataclass(frozen=True, slots=True)
class BaseModel:
    """A trainable base: its Tinker id, local 4-bit MLX id, depth, and whether it serves locally.

    Attributes:
        tinker_model: The Tinker model id to train the LoRA over.
        mlx_id: The local 4-bit MLX model id the converted adapter serves against.
        num_layers: The transformer depth, written into the mlx-lm adapter config.
        label: A short human name.
        serves_locally: Whether the converted adapter loads into mlx-lm as-is.
    """

    tinker_model: str
    mlx_id: str
    num_layers: int
    label: str
    serves_locally: bool


QWEN3_8B = BaseModel("Qwen/Qwen3-8B", "mlx-community/Qwen3-8B-4bit", 36, "qwen3-8b", serves_locally=True)


@dataclass(slots=True)
class TinkerRun:
    """Outcome of a Tinker SFT run: checkpoint paths plus token and spend accounting.

    Attributes:
        base_model: The Tinker model id trained.
        steps: The number of optimizer steps.
        batch_size: The datums per step.
        learning_rate: The AdamW learning rate.
        rank: The LoRA rank.
        seed: The batching/training seed.
        train_unembed: Whether the unembedding matrix was trained.
        checkpoints: Step -> saved sampler-checkpoint tinker path.
        train_tokens: The total tokens processed across cycled steps.
        wall_s: The training wall-clock seconds.
        losses: The per-step training loss.
        dropped: The count of datums dropped for exceeding ``max_tokens``.
    """

    base_model: str
    steps: int
    batch_size: int
    learning_rate: float
    rank: int
    seed: int
    train_unembed: bool
    checkpoints: dict[int, str] = field(default_factory=dict)
    train_tokens: int = 0
    wall_s: float = 0.0
    losses: list[float] = field(default_factory=list)
    dropped: int = 0

    @property
    def price(self) -> float:
        return PRICE_TRAIN[self.base_model]

    @property
    def train_cost_usd(self) -> float:
        return self.train_tokens / 1e6 * self.price

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "rank": self.rank,
            "seed": self.seed,
            "train_unembed": self.train_unembed,
            "checkpoints": self.checkpoints,
            "train_tokens": self.train_tokens,
            "train_cost_usd": round(self.train_cost_usd, 4),
            "wall_s": round(self.wall_s, 1),
            "final_loss": self.losses[-1] if self.losses else None,
            "dropped": self.dropped,
        }


def load_key(*, path: Path | None = None) -> None:
    """Export the Tinker credentials from ``~/.cc-steer/tinker.env`` (``path`` overrides) into the env."""
    if os.environ.get("TINKER_API_KEY"):
        return
    for raw in (path or TINKER_ENV).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tokenizer(mlx_id: str) -> PreTrainedTokenizerBase:
    """The local chat tokenizer for ``mlx_id`` (same vocab as Tinker's server), cached per id."""
    from transformers import AutoTokenizer

    if mlx_id not in TOKENIZERS:
        TOKENIZERS[mlx_id] = AutoTokenizer.from_pretrained(mlx_id)
    return TOKENIZERS[mlx_id]


def build_datum(messages: list[dict[str, str]], mlx_id: str) -> tinker.Datum:
    """One prompt-masked SFT ``Datum``: weight 1 on the assistant completion only.

    Mirrors ``datum_from_model_input_weights`` (reduction "none"): right-shifted input,
    left-shifted targets, per-target weights, the assistant boundary found by longest
    common prefix.
    """
    import tinker

    full = _encode(messages, mlx_id, add_generation_prompt=False)
    prompt = _encode(messages[:-1], mlx_id, add_generation_prompt=True)
    boundary = _boundary(full, prompt)
    mask = np.zeros(len(full), dtype=np.float32)
    mask[boundary:] = 1.0
    weights = mask[1:]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(full[:-1])),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=weights.tolist(), dtype="float32", shape=[len(weights)]),
            "target_tokens": tinker.TensorData(data=list(full[1:]), dtype="int64", shape=[len(full) - 1]),
        },
    )


def prefix_and_sentinel(system: str, user: str, mlx_id: str) -> tuple[list[int], int]:
    """Templated ids up to the answer position and the NO_STEER first-token id there.

    The answer position is found by diverging a NO_STEER assistant turn from a dummy
    one, so it is exact regardless of the ``<think></think>`` scaffold.
    """
    base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ns = _encode(base + [{"role": "assistant", "content": NO_STEER}], mlx_id, add_generation_prompt=False)
    dummy = _encode(base + [{"role": "assistant", "content": "zzz other"}], mlx_id, add_generation_prompt=False)
    i = 0
    while i < len(ns) and i < len(dummy) and ns[i] == dummy[i]:
        i += 1
    return ns[:i], ns[i]


def projected_cost_usd(datums: Sequence[tinker.Datum], *, base: BaseModel) -> float:
    """The training cost of ``datums`` at the base's price.

    Called on the full cycled batch list (every ``steps`` x ``batch_size`` slot), so the
    pre-launch estimate equals the run's realized ``train_tokens`` and the mid-run guard
    is unreachable under a correct projection.
    """
    return sum(datum.model_input.length for datum in datums) / 1e6 * PRICE_TRAIN[base.tinker_model]


def train_lora(
    datums: Sequence[tinker.Datum],
    base: BaseModel,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    rank: int,
    seed: int,
    spend_cap_usd: float,
    max_tokens: int,
    checkpoint_fractions: Sequence[float] = (1.0,),
    train_unembed: bool = False,
    on_checkpoint: Callable[[int, str, tinker.ServiceClient, BaseModel], None] | None = None,
) -> TinkerRun:
    """Run the SFT loop under a spend cap; save a sampler checkpoint at each fraction of ``steps``.

    Drops datums whose length exceeds ``max_tokens`` before batching, then aborts before
    constructing the training client when the full cycled projected cost exceeds
    ``spend_cap_usd``. The projection sums the exact token count of every batch slot, so
    it equals the realized training cost — a run that would overspend never launches.
    Each checkpoint fires ``on_checkpoint(step, path, service_client, base)``.

    Args:
        datums: The pre-built prompt-masked training datums.
        base: The base model to train the LoRA over.
        max_tokens: Datums whose ``model_input.length + 1`` exceeds this are dropped.
        checkpoint_fractions: Fractions of ``steps`` to save sampler checkpoints at.
        on_checkpoint: An optional callback fired after each checkpoint saves.

    Raises:
        InsufficientDatumsError: When fewer datums than one batch survive the ``max_tokens`` filter.
        SpendCapError: When the full projected cost exceeds ``spend_cap_usd``.
    """
    kept = [datum for datum in datums if datum.model_input.length + 1 <= max_tokens]
    dropped = len(datums) - len(kept)
    if len(kept) < batch_size:
        raise InsufficientDatumsError(
            f"only {len(kept)} training datums remain after the max_tokens={max_tokens} filter "
            f"(dropped {dropped}), fewer than one batch of {batch_size}"
        )
    batches = _batches(kept, batch_size, steps, seed)
    if (projected := projected_cost_usd([d for batch in batches for d in batch], base=base)) > spend_cap_usd:
        raise SpendCapError(f"projected cost ${projected:.2f} exceeds cap ${spend_cap_usd:.2f}; not launching")
    checkpoints = _checkpoint_steps(checkpoint_fractions, steps)

    import tinker

    load_key()
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(
        base_model=base.tinker_model, rank=rank, seed=seed, train_mlp=True, train_attn=True, train_unembed=train_unembed
    )
    run = TinkerRun(
        base_model=base.tinker_model,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rank=rank,
        seed=seed,
        train_unembed=train_unembed,
        dropped=dropped,
    )
    start = time.monotonic()
    for step, batch in enumerate(batches, start=1):
        result = tc.forward_backward(batch, "cross_entropy").result()
        tc.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
        run.train_tokens += sum(datum.model_input.length for datum in batch)
        logprobs = np.concatenate([out["logprobs"].to_numpy() for out in result.loss_fn_outputs])
        weights = np.concatenate([datum.loss_fn_inputs["weights"].to_numpy() for datum in batch])
        run.losses.append(float(-np.dot(logprobs, weights) / weights.sum()))
        if step in checkpoints:
            path = str(tc.save_weights_for_sampler(name=f"step{step}").result().path)
            run.checkpoints[step] = path
            if on_checkpoint is not None:
                on_checkpoint(step, path, svc, base)
    run.wall_s = time.monotonic() - start
    return run


def download_adapter(service_client: tinker.ServiceClient, tinker_path: str, out_dir: Path) -> Path:
    """Download and extract a Tinker sampler checkpoint (PEFT format) into ``out_dir``."""
    import tarfile
    import tempfile
    import urllib.request

    rest = service_client.create_rest_client()
    url = rest.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        with tarfile.open(tmp_path) as tar:
            tar.extractall(out_dir, filter="data")
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_dir


def convert_peft_to_mlx(
    peft_dir: Path, out_dir: Path, *, num_layers: int, rank: int = 16, alpha: int = 32
) -> dict[str, Any]:
    """Convert a Tinker PEFT LoRA to an mlx-lm adapter over the standard-attention modules.

    PEFT ``lora_A`` ``[r,in]`` / ``lora_B`` ``[out,r]`` map to mlx-lm ``lora_a`` ``[in,r]``
    / ``lora_b`` ``[r,out]`` by transpose; the effective scale is ``alpha/rank``.
    Non-standard modules (split ``linear_attn.in_proj_*``, ``unembed_tokens``) are
    dropped. Returns the conversion report: covered modules, dropped modules, and the
    weight count.
    """
    import mlx.core as mx

    weights = mx.load(str(peft_dir / "adapter_model.safetensors"))
    out: dict[str, Any] = {}
    covered: set[str] = set()
    dropped: set[str] = set()
    for key, weight in weights.items():
        match = PEFT_KEY.match(key)
        if not match:
            dropped.add(key.split("base_model.model.")[-1])
            continue
        mlx_path, module, ab = match.group(1), match.group(2), match.group(3)
        if module not in STD_MODULES:
            dropped.add(module)
            continue
        out[f"{mlx_path}.lora_{'a' if ab == 'A' else 'b'}"] = weight.T
        covered.add(module)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), out)
    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {"rank": rank, "scale": alpha / rank, "dropout": 0.0, "keys": list(STD_MODULES)},
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {"covered": sorted(covered), "dropped": sorted(dropped), "n_lora_weights": len(out)}


def score_auc_tinker(
    service_client: tinker.ServiceClient, model_path: str, valid_rows: list[dict[str, str]], mlx_id: str
) -> dict[str, Any]:
    """Sentinel-first-token AUC on the Tinker sampler at ``model_path`` over ``valid_rows``."""
    import tinker
    from sklearn.metrics import roc_auc_score

    sampler = service_client.create_sampling_client(model_path=model_path)
    probs: list[float] = []
    labels: list[bool] = []
    for row in valid_rows:
        prefix, sentinel = prefix_and_sentinel(row["system"], row["user"], mlx_id)
        logprobs = sampler.compute_logprobs(prompt=tinker.ModelInput.from_ints([*prefix, int(sentinel)]))
        logprobs = logprobs.result() if hasattr(logprobs, "result") else logprobs
        probs.append(float(np.exp(logprobs[-1])))
        labels.append(row["assistant"].strip() != NO_STEER)
    if not np.isfinite(auc := float(roc_auc_score(labels, [1.0 - p for p in probs]))):
        raise NonFiniteAUCError(
            f"sentinel AUC is {auc} over {len(labels)} rows ({int(sum(labels))} positive); "
            "the validation set is single-class"
        )
    return {"auc": auc, "n_val": len(labels), "n_pos": int(sum(labels))}


def score_frame_tinker(
    service_client: tinker.ServiceClient,
    model_path: str,
    frame: EvalFrame,
    *,
    base: BaseModel,
    system: str = DRAFT_SYSTEM,
) -> dict[str, float]:
    """Per-row ``P(NO_STEER)`` for every eval-frame row via Tinker ``compute_logprobs``.

    Single-shot and fail-fast — no resume cache. Scores the checkpoint at ``model_path``
    against the frame's render-v2 tails, returning ``{row_id: P(NO_STEER)}`` for
    :func:`~cc_steer.retrain.evalset.write_probs`.
    """
    import tinker

    sampler = service_client.create_sampling_client(model_path=model_path)
    probs: dict[str, float] = {}
    for row_id, tail in zip(frame.ids, frame.tails, strict=True):
        prefix, sentinel = prefix_and_sentinel(system, tail, base.mlx_id)
        logprobs = sampler.compute_logprobs(prompt=tinker.ModelInput.from_ints([*prefix, int(sentinel)]))
        logprobs = logprobs.result() if hasattr(logprobs, "result") else logprobs
        probs[row_id] = float(np.exp(logprobs[-1]))
    return probs


def _encode(messages: list[dict[str, str]], mlx_id: str, *, add_generation_prompt: bool) -> list[int]:
    """Render the chat template (thinking disabled) to a string, then encode to ids."""
    tok = tokenizer(mlx_id)
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )
    return tok.encode(text, add_special_tokens=False)


def _boundary(full: list[int], prompt: list[int]) -> int:
    """First index where the assistant completion begins (longest common prefix)."""
    if full[: len(prompt)] == prompt:
        return len(prompt)
    i = 0
    while i < min(len(full), len(prompt)) and full[i] == prompt[i]:
        i += 1
    return i


def _batches(datums: list[tinker.Datum], batch_size: int, steps: int, seed: int) -> list[list[tinker.Datum]]:
    """Seeded shuffled fixed-size batches, cycling the pool to reach ``steps``."""
    rng = np.random.default_rng(seed)
    order: list[int] = []
    out: list[list[tinker.Datum]] = []
    while len(out) < steps:
        if len(order) < batch_size:
            order.extend(int(i) for i in rng.permutation(len(datums)))
        idx, order = order[:batch_size], order[batch_size:]
        out.append([datums[i] for i in idx])
    return out


def _checkpoint_steps(fractions: Sequence[float], steps: int) -> set[int]:
    """The concrete steps to checkpoint at, one per fraction, clamped into ``[1, steps]``."""
    return {max(1, min(steps, round(fraction * steps))) for fraction in fractions}
