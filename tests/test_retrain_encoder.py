from __future__ import annotations

import importlib.util
import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cc_steer.retrain import encoder

if TYPE_CHECKING:
    from pathlib import Path

HAS_TORCH = bool(importlib.util.find_spec("torch")) and bool(importlib.util.find_spec("transformers"))
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="the [encoder] extra (torch + transformers) is not installed")

GATE_COLUMNS = ("id", "text", "label", "kind", "offset_turns", "source_kind", "category", "session_id", "split")

POS_WORDS = "use a frozen dataclass instead of a plain dict right here"
NEG_WORDS = "the build passed cleanly and the suite looks totally fine today"


def gate_rows(split: str, *, n_pos: int, n_neg: int) -> list[dict[str, object]]:
    pos = [
        {
            "id": f"{split}-p{i}",
            "text": f"{POS_WORDS} number {i}",
            "label": True,
            "kind": "positive",
            "offset_turns": 0,
            "source_kind": "transcript_message",
            "category": "wrong_approach",
            "session_id": f"sp{i}",
            "split": split,
        }
        for i in range(n_pos)
    ]
    neg = [
        {
            "id": f"{split}-n{i}",
            "text": f"{NEG_WORDS} run {i}",
            "label": False,
            "kind": "hard_negative" if i % 2 else "random_negative",
            "offset_turns": 0,
            "source_kind": "question_answer" if i % 2 else "transcript_message",
            "category": "",
            "session_id": f"sn{i}",
            "split": split,
        }
        for i in range(n_neg)
    ]
    return pos + neg


def gate_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.table({column: [row[column] for row in rows] for column in GATE_COLUMNS})


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "gate").mkdir(parents=True)
    pq.write_table(gate_table(gate_rows("train", n_pos=16, n_neg=16)), root / "gate" / "train.parquet")
    pq.write_table(gate_table(gate_rows("test", n_pos=6, n_neg=6)), root / "gate" / "test.parquet")
    return root


@pytest.fixture
def eval_dir(tmp_path: Path, dataset_dir: Path) -> Path:
    from cc_steer.retrain import evalset

    root = tmp_path / "eval"
    evalset.freeze_eval("gate", dataset_dir=dataset_dir, root=root)
    return root


def tiny_encoder_dir(tmp_path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *sorted({*(POS_WORDS + " " + NEG_WORDS).split(), "number", "run"})]
    vocab = {word: index for index, word in enumerate(words)}
    inner = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()
    inner.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=inner, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            num_labels=2,
            pad_token_id=vocab["[PAD]"],
        )
    )
    out = tmp_path / "tiny-encoder"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


@pytest.fixture
def tiny_spec(tmp_path: Path) -> encoder.EncoderSpec:
    return encoder.EncoderSpec(
        model_id=str(tiny_encoder_dir(tmp_path)), max_length=32, epochs=2.0, lr=5e-3, batch_size=8, val_frac=0.25
    )


class TestModuleGating:
    def test_module_imports_without_the_extra(self) -> None:
        # Importing the lane must never require torch; the heavy imports live inside the bodies.
        assert importlib.util.find_spec("cc_steer.retrain.encoder") is not None
        assert encoder.EncoderSpec(model_id="x").model_id == "x"

    def test_training_without_torch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Blocker:
            def find_spec(self, name: str, path: object = None, target: object = None) -> None:
                if name.split(".")[0] in {"torch", "transformers"}:
                    raise ModuleNotFoundError(f"blocked {name} to simulate the missing [encoder] extra")
                return None

        for name in [n for n in sys.modules if n.split(".")[0] in {"torch", "transformers"}]:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
        frame = encoder.EncoderFrame.from_table(gate_table(gate_rows("train", n_pos=2, n_neg=2)))
        with pytest.raises(ModuleNotFoundError, match="torch|transformers"):
            encoder.train_encoder(encoder.EncoderSpec(model_id="x"), frame)


class TestEncoderFrame:
    def test_from_table_carries_ids_labels_and_digest(self) -> None:
        table = gate_table(gate_rows("test", n_pos=3, n_neg=3))
        frame = encoder.EncoderFrame.from_table(table)
        from cc_steer.retrain.data import dataset_digest

        assert frame.ids == tuple(table.column("id").to_pylist())
        assert frame.labels.tolist() == [True, True, True, False, False, False]
        assert frame.digest == dataset_digest(table.to_pylist())

    def test_carve_holds_out_both_classes(self) -> None:
        frame = encoder.EncoderFrame.from_table(gate_table(gate_rows("train", n_pos=8, n_neg=8)))
        rest, val = encoder.carve_val(frame, seed=1729, frac=0.25)
        assert len(rest) + len(val) == len(frame)
        assert val.labels.any() and (~val.labels).any()
        # A carve is deterministic in the seed.
        again, _ = encoder.carve_val(frame, seed=1729, frac=0.25)
        assert again.ids == rest.ids


@requires_torch
class TestTrainScoreRoundtrip:
    def test_train_score_and_write_probs(
        self, tiny_spec: encoder.EncoderSpec, dataset_dir: Path, eval_dir: Path
    ) -> None:
        model = encoder.train_encoder(tiny_spec, encoder.EncoderFrame.load_train(dataset_dir=dataset_dir))
        assert model.temperature > 0.0
        assert 0.0 <= model.val_ece <= 1.0
        eval_frame = encoder.EncoderFrame.load_eval(root=eval_dir)
        path = encoder.score_frozen(model, eval_frame, version="enc1", root=eval_dir)
        payload = json.loads(path.read_text())
        assert set(payload["probs"]) == set(eval_frame.ids)
        assert all(0.0 <= value <= 1.0 for value in payload["probs"].values())
        assert payload["meta"]["render"] == 2
        assert payload["meta"]["dataset_digest"] == eval_frame.digest
        assert 0.0 <= payload["meta"]["auc"] <= 1.0

    def test_same_seed_gives_same_probs(
        self, tiny_spec: encoder.EncoderSpec, dataset_dir: Path, eval_dir: Path
    ) -> None:
        train_frame = encoder.EncoderFrame.load_train(dataset_dir=dataset_dir)
        eval_frame = encoder.EncoderFrame.load_eval(root=eval_dir)
        first = encoder.train_encoder(tiny_spec, train_frame).probs(eval_frame.texts)
        second = encoder.train_encoder(tiny_spec, train_frame).probs(eval_frame.texts)
        np.testing.assert_allclose(first, second, atol=1e-5)

    def test_uncalibrated_leaves_temperature_at_one(self, tiny_spec: encoder.EncoderSpec, dataset_dir: Path) -> None:
        from dataclasses import replace

        model = encoder.train_encoder(
            replace(tiny_spec, calibrate=False), encoder.EncoderFrame.load_train(dataset_dir=dataset_dir)
        )
        assert model.temperature == 1.0
