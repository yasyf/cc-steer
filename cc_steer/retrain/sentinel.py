"""Sentinel-token scoring: the frozen eval's fire signal, expressed as athome eval rows.

The watcher's true gate metric is the model's confidence in the ``NO_STEER`` sentinel at
the answer position — a single next-token discriminator. This module renders that position
through the local chat tokenizer (thinking disabled, so ids match Tinker's server tokenizer)
and packages it as an :class:`~athome.train.spec.EvalRow` that weights only the sentinel:
:func:`sentinel_eval_row` feeds the eval seam of :func:`athome.train.retrain`, and
:func:`checkpoint_auc` ranks checkpoints from the sentinel logprobs those rows score to.

Rendering goes through :func:`athome.train.data.chat_ids` / :func:`~athome.train.data.boundary`;
there is no ``tinker`` or ``mlx`` dependency here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from athome.train.data import Message, boundary, chat_ids
from athome.train.gate import sentinel_auc
from athome.train.spec import EvalRow, MlxModelId

from cc_steer.rendering import NO_STEER

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from athome.train.spec import SavedCheckpoint

DUMMY_ANSWER = "zzz other"


class NonFiniteAUCError(RuntimeError):
    """A scored AUC is non-finite (a single-class validation set), so it cannot rank checkpoints."""


def prefix_and_sentinel_via(render: Callable[[str], list[int]]) -> tuple[list[int], int]:
    """Templated ids up to the answer position and the ``NO_STEER`` first-token id there.

    ``render`` maps an assistant answer to the full templated id sequence for the
    ``[system, user, assistant=answer]`` turn (``add_generation_prompt=False``). Diverging the
    ``NO_STEER`` rendering from a dummy one finds the answer position exactly, regardless of the
    ``<think></think>`` scaffold the template injects — the shared divergence core the local mlx
    drafter, the offline eval, and the hosted http drafter all score at.
    """
    ns = render(NO_STEER)
    return ns[: (cut := boundary(ns, render(DUMMY_ANSWER)))], ns[cut]


def prefix_and_sentinel(system: str, user: str, mlx_id: str) -> tuple[list[int], int]:
    """Templated ids up to the answer position and the ``NO_STEER`` first-token id there.

    The answer position is found by diverging a ``NO_STEER`` assistant turn from a dummy one,
    so it is exact regardless of the ``<think></think>`` scaffold.
    """
    model = MlxModelId(mlx_id)
    base = [Message(role="system", content=system), Message(role="user", content=user)]
    return prefix_and_sentinel_via(
        lambda answer: chat_ids([*base, Message(role="assistant", content=answer)], model, add_generation_prompt=False)
    )


def sentinel_eval_row(system: str, user: str, mlx_id: str) -> EvalRow:
    """A scoring row over the full templated sequence, weighting only the sentinel's position."""
    prefix, sentinel = prefix_and_sentinel(system, user, mlx_id)
    return EvalRow(tokens=(*prefix, sentinel), weights=(*(0.0 for _ in prefix), 1.0))


def checkpoint_auc(labels: npt.ArrayLike, saved: SavedCheckpoint) -> float:
    """Sentinel AUC of the fire score ``1 - P(NO_STEER)`` against the fire labels.

    ``saved.scores`` carry per-row ``log P(NO_STEER first token | prefix)`` — a higher one is
    the model more sure the row should abstain, so it fires less. The fire score inverts that,
    so a checkpoint that best separates true steers from ``NO_STEER`` rows ranks highest.

    Raises:
        ValueError: The checkpoint carries no eval scores to rank by.
        NonFiniteAUCError: The AUC is non-finite because the validation labels are single-class.
    """
    if saved.scores is None:
        raise ValueError(f"checkpoint step {saved.step} carries no eval scores to rank by")
    label_arr = np.asarray(labels, dtype=bool)
    scores = 1.0 - np.exp(np.asarray([scored.logprob for scored in saved.scores], dtype=np.float64))
    if not np.isfinite(auc := sentinel_auc(label_arr, scores)):
        raise NonFiniteAUCError(
            f"sentinel AUC is {auc} over {label_arr.size} rows ({int(label_arr.sum())} positive); "
            "the validation set is single-class"
        )
    return auc
