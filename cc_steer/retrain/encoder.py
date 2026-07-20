"""The encoder gate lane: fine-tune a transformer encoder as the stage-1 gate, scored paired.

E36 bakes small encoders — Ettin-150m, DeBERTaV3, ModernBERT, SetFit — off against the
incumbent lexical gate on the frozen gate frame. This lane fine-tunes a
``AutoModelForSequenceClassification`` head on the exported gate train view, temperature-scales
it on a held-out val carve (the same calibration idiom as
:mod:`~cc_steer.retrain.lexical` — its :func:`~cc_steer.retrain.lexical.fit_temperature` and
:func:`~cc_steer.retrain.lexical.ece`), and scores the frozen gate eval, landing each arm's
per-row ``P(fire)`` through :func:`~cc_steer.retrain.evalset.write_probs`. So an encoder arm is
paired-comparable with the lexical gate and the incumbent on the same frame.

Encoder training lives OUTSIDE athome: athome's ``TrainBackend`` is generative-LoRA-shaped, the
wrong tool for a classifier head. The specific bake-off model ids stay experiment-local (E36
supplies them); this module carries the reusable machinery and a couple of representative
:class:`EncoderSpec` presets. SetFit is a distinct contrastive-fit path left to E36 — it is not
wired here. ``torch`` and ``transformers`` live behind the ``encoder`` extra and are imported
lazily inside the training and scoring bodies, so importing :mod:`cc_steer` without the extra
never drags torch.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cc_steer import instrument
from cc_steer.retrain import data, evalset
from cc_steer.retrain.data import dataset_digest
from cc_steer.retrain.lexical import ece, fit_temperature, probs_from_logits

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pyarrow as pa
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from cc_steer.retrain.data import DatasetDigest

SCORE_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Recipe for one encoder gate arm: the base model plus its fine-tune and calibration knobs.

    ``val_frac`` and ``calibrate`` are the calibration knobs: a stratified ``val_frac`` slice is
    held out of training to fit the temperature (skipped, leaving ``T = 1``, when ``calibrate`` is
    ``False``). The specific bake-off ids are experiment-local; :data:`PRESETS` carries a couple of
    representative recipes.

    Example:
        >>> EncoderSpec(model_id="bert-base-uncased", epochs=3.0, lr=2e-5)
    """

    model_id: str
    max_length: int = 512
    epochs: float = 3.0
    lr: float = 2e-5
    batch_size: int = 16
    seed: int = 1729
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    val_frac: float = 0.1
    calibrate: bool = True


PRESETS: dict[str, EncoderSpec] = {
    "bert-base": EncoderSpec(model_id="bert-base-uncased"),
    "distilbert": EncoderSpec(model_id="distilbert-base-uncased", batch_size=32),
}


@dataclass(frozen=True, slots=True)
class EncoderFrame:
    """Column-oriented slice of the gate view: the text, the fire labels, and the content digest.

    Mirrors :class:`~cc_steer.retrain.lexical.GateFrame`'s loading so an encoder arm trains and
    scores on the same rows as the lexical gate, and carries the order-invariant ``digest`` that
    :func:`~cc_steer.retrain.evalset.write_probs` stamps into the probs store.
    """

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: np.ndarray
    digest: DatasetDigest

    @classmethod
    def from_table(cls, table: pa.Table) -> EncoderFrame:
        return cls(
            ids=tuple(table.column("id").to_pylist()),
            texts=tuple(table.column("text").to_pylist()),
            labels=np.asarray(table.column("label").to_pylist(), dtype=bool),
            digest=dataset_digest(table.to_pylist()),
        )

    @classmethod
    def load_train(cls, *, dataset_dir: Path | None = None) -> EncoderFrame:
        import pyarrow.parquet as pq

        path = (dataset_dir or data.DATASET_DIR) / "gate" / "train.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no gate train parquet at {path}")
        return cls.from_table(pq.read_table(path))

    @classmethod
    def load_eval(cls, *, root: Path | None = None) -> EncoderFrame:
        return cls.from_table(evalset.load_frozen("gate", root=root))

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, indices: Sequence[int] | np.ndarray) -> EncoderFrame:
        idx = np.asarray(indices, dtype=np.intp)
        return EncoderFrame(
            ids=tuple(self.ids[i] for i in idx),
            texts=tuple(self.texts[i] for i in idx),
            labels=self.labels[idx],
            digest=self.digest,
        )


@dataclass(frozen=True, slots=True)
class EncoderModel:
    """A fine-tuned encoder head plus its fitted temperature and held-out calibration error.

    :meth:`probs` applies the temperature to the model's fire-margin logits, so its output is the
    calibrated ``P(fire)`` that :func:`score_frozen` persists.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: EncoderSpec
    temperature: float
    val_ece: float

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        return encode_logits(self.model, self.tokenizer, texts, max_length=self.spec.max_length)

    def probs(self, texts: Sequence[str]) -> np.ndarray:
        return probs_from_logits(self.logits(texts), self.temperature)


def carve_val(frame: EncoderFrame, *, seed: int, frac: float) -> tuple[EncoderFrame, EncoderFrame]:
    """Deterministic label-stratified carve of a temperature-scaling val slice; returns ``(rest, val)``."""
    rng = np.random.default_rng(seed)
    rest_idx: list[int] = []
    val_idx: list[int] = []
    for value in (False, True):
        idx = np.flatnonzero(frame.labels == value)
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * frac)) if len(idx) else 0
        val_idx.extend(idx[:n_val].tolist())
        rest_idx.extend(idx[n_val:].tolist())
    return frame.take(sorted(rest_idx)), frame.take(sorted(val_idx))


def encode_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int = SCORE_BATCH_SIZE,
) -> np.ndarray:
    """The per-text fire margin ``logit[fire] - logit[no_fire]`` from the sequence-classification head."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    margins: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(margins) if margins else np.zeros(0, dtype=np.float64)


def train_encoder(spec: EncoderSpec, train_frame: EncoderFrame, *, output_dir: Path | None = None) -> EncoderModel:
    """Fine-tune the encoder head on ``train_frame``, temperature-scaled on a held-out val carve.

    Loads ``spec.model_id`` as a two-label sequence classifier, trains it under ``spec``'s knobs,
    then fits the temperature on a stratified ``spec.val_frac`` carve — the lexical lane's
    calibration idiom — and stamps the val expected calibration error onto the artifact. Uses the
    GPU only when CUDA is present, else CPU (never MPS), so a fine-tune is reproducible in
    ``spec.seed`` on the eval hosts.

    Returns:
        The fine-tuned :class:`EncoderModel` carrying the fitted temperature and val ECE.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(spec.seed)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(spec.model_id, num_labels=2)
    rest, val = carve_val(train_frame, seed=spec.seed, frac=spec.val_frac)
    dataset = Dataset.from_dict({"text": list(rest.texts), "labels": [int(label) for label in rest.labels]}).map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=spec.max_length),
        batched=True,
        remove_columns=["text"],
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir or tempfile.mkdtemp(prefix="cc-steer-encoder-")),
            num_train_epochs=spec.epochs,
            learning_rate=spec.lr,
            per_device_train_batch_size=spec.batch_size,
            weight_decay=spec.weight_decay,
            warmup_ratio=spec.warmup_ratio,
            seed=spec.seed,
            data_seed=spec.seed,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=not torch.cuda.is_available(),
        ),
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    val_logits = encode_logits(model, tokenizer, val.texts, max_length=spec.max_length)
    temperature = fit_temperature(val.labels, val_logits) if spec.calibrate else 1.0
    return EncoderModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        temperature=temperature,
        val_ece=ece(val.labels, probs_from_logits(val_logits, temperature)),
    )


def score_frozen(
    model: EncoderModel,
    frame: EncoderFrame,
    *,
    version: str,
    render: int = evalset.RENDER_VERSION,
    root: Path | None = None,
) -> Path:
    """Score the frozen gate eval and persist the calibrated ``P(fire)`` through ``write_probs``.

    The per-row probabilities and their fire AUC land through
    :func:`~cc_steer.retrain.evalset.write_probs`, render-tagged and stamped with the frame digest,
    so the encoder arm is paired-comparable with the lexical gate and the incumbent on the same
    frame.

    Args:
        model: The fine-tuned, temperature-scaled encoder.
        frame: The frozen gate eval frame to score.
        version: The registry version label the stored probs are keyed under.
        render: The render version stamped into the probs store.
        root: Eval root override; defaults to ``~/.cc-steer/eval``.

    Returns:
        The path the per-row probabilities were written to.
    """
    probs = model.probs(frame.texts)
    auc = instrument.auc(probs.tolist(), [int(label) for label in frame.labels])
    return evalset.write_probs(
        frame,
        version,
        {row_id: float(prob) for row_id, prob in zip(frame.ids, probs, strict=True)},
        auc=auc,
        render=render,
        root=root,
    )
