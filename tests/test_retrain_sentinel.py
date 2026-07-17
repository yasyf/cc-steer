from __future__ import annotations

import math

import pytest
from athome.train.spec import SavedCheckpoint, ScoredSequence

from cc_steer.rendering import NO_STEER
from cc_steer.retrain import sentinel
from cc_steer.retrain.sentinel import NonFiniteAUCError, checkpoint_auc, prefix_and_sentinel, sentinel_eval_row


def saved_with(logprobs: list[float]) -> SavedCheckpoint:
    return SavedCheckpoint(
        step=1, path="tinker://run/x", final=True, scores=tuple(ScoredSequence(lp, 1.0) for lp in logprobs)
    )


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


@pytest.fixture
def stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    import athome.train.data as adata

    monkeypatch.setattr(adata, "tokenizer", lambda mlx_id: StubTokenizer())


def render(content: str) -> list[int]:
    return [ord(char) for char in f"system:S\nuser:U\nassistant:{content}\n"]


class TestPrefixAndSentinel:
    def test_sentinel_is_the_first_divergent_token(self, stub_tokenizer: None) -> None:
        ns_ids, dummy_ids = render(NO_STEER), render("zzz other")
        div = next(i for i, (a, b) in enumerate(zip(ns_ids, dummy_ids, strict=False)) if a != b)
        prefix, token = prefix_and_sentinel("S", "U", "stub")
        assert prefix == ns_ids[:div]
        assert token == ns_ids[div]


class TestSentinelEvalRow:
    def test_weight_lands_only_on_the_sentinel_position(self, stub_tokenizer: None) -> None:
        prefix, token = prefix_and_sentinel("S", "U", "stub")
        row = sentinel_eval_row("S", "U", "stub")
        assert row.tokens == (*prefix, token)
        assert row.weights == (*([0.0] * len(prefix)), 1.0)
        assert len(row.weights) == len(row.tokens)
        assert sum(row.weights) == 1.0 and row.weights[-1] == 1.0


class TestCheckpointAuc:
    def test_high_fire_on_true_steers_scores_perfect(self) -> None:
        # Positives (should fire) get low P(NO_STEER); the fire score 1 - P separates them cleanly.
        labels = [True, True, False, False]
        logprobs = [math.log(0.1), math.log(0.2), math.log(0.8), math.log(0.9)]
        assert checkpoint_auc(labels, saved_with(logprobs)) == pytest.approx(1.0)

    def test_high_nosteer_prob_on_true_steers_inverts(self) -> None:
        # A model confident that true steers are NO_STEER fires least on them: the orientation flips AUC to 0.
        labels = [True, True, False, False]
        logprobs = [math.log(0.9), math.log(0.8), math.log(0.2), math.log(0.1)]
        assert checkpoint_auc(labels, saved_with(logprobs)) == pytest.approx(0.0)

    def test_non_finite_auc_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentinel, "sentinel_auc", lambda labels, scores: float("nan"))
        with pytest.raises(NonFiniteAUCError, match="single-class"):
            checkpoint_auc([True, False], saved_with([math.log(0.2), math.log(0.8)]))

    def test_scoreless_checkpoint_refuses_to_rank(self) -> None:
        scoreless = SavedCheckpoint(step=3, path="tinker://run/x", final=False, scores=None)
        with pytest.raises(ValueError, match="no eval scores"):
            checkpoint_auc([True, False], scoreless)
