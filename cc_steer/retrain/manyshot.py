"""Many-shot frontier watcher: score the frozen frame through a cached exemplar prefix.

E32 asks whether a frontier model, shown many labeled steering examples in a
byte-constant system prompt, matches the trained watcher on the frozen n=628 frame.
:func:`build_exemplar_system` renders a seeded, budget-bounded block of watcher
demonstrations into one system string; :func:`score_frame` reuses that exact string
across every scoring call so the provider's prompt cache amortizes it — a cache read
is 7-13x cheaper than re-sending the prefix — batching the frame rows so each call
carries fifteen-to-twenty-five queries against the shared prefix. The per-row
``P(NO_STEER)`` lands through :func:`~cc_steer.retrain.evalset.write_probs`, so a
many-shot arm is paired-comparable with any trained watcher on the same frame.

The cache is load-bearing, not a nicety: :func:`score_frame` reads
``cache_read_input_tokens`` off every turn's usage and aborts loudly on a streak of
cache misses, because a silently-cold cache turns the run's cost into per-row prefix
resends with no error to show for it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from cc_steer import instrument
from cc_steer.claude import run_claude
from cc_steer.retrain import evalset
from cc_steer.retrain.data import SEED

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_steer.retrain.data import WatcherRow
    from cc_steer.retrain.evalset import EvalFrame

EXEMPLAR_BUDGET = 40_000
BATCH_SIZE = 20
MAX_CACHE_MISS_STREAK = 3

SYSTEM_PREFIX = """\
You are a watcher over one developer's coding agent. From the labeled examples below \
you learn when THIS developer steers the agent — correcting it, redirecting it, or \
resolving a choice it raised — versus when they let it proceed without steering.

Each example shows the context the developer saw, then `=>` their steer, or \
`=> NO_STEER` when they did not steer.

<examples>
"""

SYSTEM_SUFFIX = """
</examples>

The user message holds numbered contexts. For each one, output the probability in \
[0, 1] that this developer would steer at that moment (1.0 = certainly steers, 0.0 = \
certainly does not). Output exactly one line per context, `<number>: <probability>`, \
and nothing else."""

_PROB_LINE = re.compile(r"^\s*(\d+)\s*[:.)]\s*(-?\d*\.?\d+)", re.MULTILINE)


class ManyShotError(RuntimeError):
    """A many-shot scoring pass failed: an unparseable batch or a cold prompt cache."""


def exemplar_demo(row: WatcherRow) -> str:
    """One labeled demonstration: the context tail and the developer's steer (or ``NO_STEER``)."""
    return f"<example>\n{row.draft_text()}\n=>\n{row.reference}\n</example>"


def build_exemplar_system(rows: Sequence[WatcherRow], *, budget_chars: int = EXEMPLAR_BUDGET, seed: int = SEED) -> str:
    """The byte-constant many-shot system prompt: seeded demonstrations under ``budget_chars``.

    Walks ``rows`` in a seeded permutation, greedily keeping each demonstration whose
    body fits the remaining budget and skipping any that would overflow, then wraps
    them in the fixed instruction frame. Deterministic in ``(rows, budget_chars, seed)``
    and byte-identical every call, so :func:`score_frame` can reuse one prefix across
    every batch and hit the provider's prompt cache.
    """
    order = np.random.default_rng(seed).permutation(len(rows))
    demos: list[str] = []
    used = len(SYSTEM_PREFIX) + len(SYSTEM_SUFFIX)
    for index in order:
        if used + len(demo := exemplar_demo(rows[index])) + 1 <= budget_chars:
            demos.append(demo)
            used += len(demo) + 1
    return SYSTEM_PREFIX + "\n".join(demos) + SYSTEM_SUFFIX


def batch_prompt(tails: Sequence[str]) -> str:
    """The user message for one batch: each frame tail as a numbered context."""
    contexts = "\n\n".join(f"CONTEXT {position}:\n{tail}" for position, tail in enumerate(tails, start=1))
    return f"{contexts}\n\nReturn exactly {len(tails)} lines, `<number>: <probability>`."


def parse_batch_probs(text: str, count: int) -> list[float]:
    """The per-context steer probabilities parsed from one batch reply, clamped to [0, 1].

    Raises:
        ManyShotError: When the reply does not carry a line for every context.
    """
    found = {int(match.group(1)): float(match.group(2)) for match in _PROB_LINE.finditer(text)}
    if missing := [position for position in range(1, count + 1) if position not in found]:
        raise ManyShotError(f"batch reply missing {len(missing)} of {count} contexts: {missing}")
    return [min(1.0, max(0.0, found[position])) for position in range(1, count + 1)]


async def score_frame(
    frame: EvalFrame,
    exemplars: Sequence[WatcherRow],
    *,
    version: str,
    model: str,
    budget_chars: int = EXEMPLAR_BUDGET,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    eval_root: Path | None = None,
    max_cache_miss_streak: int = MAX_CACHE_MISS_STREAK,
) -> Path:
    """Score the frozen frame with a cached many-shot prefix and store its ``P(NO_STEER)``.

    Builds one byte-constant exemplar system prompt from ``exemplars`` and scores the
    frame in ``batch_size`` batches against it, so every batch after the first reads
    the prefix from the provider's prompt cache instead of re-sending it. Each batch's
    per-context steer probabilities become ``P(NO_STEER) = 1 - p``; the whole frame's
    probabilities and their fire AUC land through
    :func:`~cc_steer.retrain.evalset.write_probs`, paired-comparable with any trained
    watcher on the same frame.

    Args:
        frame: The frozen eval frame to score.
        exemplars: The demonstration pool the system prefix draws from (watcher train rows).
        version: The registry version label the stored probs are keyed under.
        model: The frontier model to run, for example ``claude-opus-4-6``.
        budget_chars: Character budget for the exemplar prefix.
        batch_size: Frame rows scored per cached call.
        seed: Seed for the exemplar shuffle.
        eval_root: Eval root override; defaults to ``~/.cc-steer/eval``.
        max_cache_miss_streak: Consecutive post-warmup cache misses that abort the pass.

    Returns:
        The path the per-row probabilities were written to.

    Raises:
        ManyShotError: When a batch reply is unparseable or the prompt cache stays cold
            for ``max_cache_miss_streak`` batches in a row.
    """
    system = build_exemplar_system(exemplars, budget_chars=budget_chars, seed=seed)
    probs: dict[str, float] = {}
    miss_streak = 0
    for batch, start in enumerate(range(0, len(frame), batch_size)):
        ids = frame.ids[start : start + batch_size]
        result = await run_claude(batch_prompt(frame.tails[start : start + batch_size]), system=system, model=model)
        for row_id, steer in zip(ids, parse_batch_probs(result.text, len(ids)), strict=True):
            probs[row_id] = 1.0 - steer
        if batch == 0 or result.usage is None:
            continue
        miss_streak = 0 if result.usage.cache_read_input_tokens > 0 else miss_streak + 1
        if miss_streak >= max_cache_miss_streak:
            raise ManyShotError(
                f"prompt cache cold for {miss_streak} batches in a row (0 cache-read tokens); "
                "the byte-constant exemplar prefix is not amortizing — aborting rather than paying per-row resends"
            )
    fire = (1.0 - np.array([probs[row_id] for row_id in frame.ids], dtype=np.float64)).tolist()
    auc = instrument.auc(fire, [int(label) for label in frame.labels])
    return evalset.write_probs(frame, version, probs, auc=auc, render=evalset.RENDER_VERSION, root=eval_root)
