"""Shared train/serve rendering: one contract for export, the exemplar index, and the live watcher.

Every consumer that turns a :class:`~cc_transcript.context.ContextWindow` into
model input — the dataset export, the exemplar embedding index, and the live
cascade — renders through this module, so what the models see at inference is
byte-identical to what they saw in training. Windows come in two anchor shapes
and both normalize through :func:`window_parts`: a steering event's window is
anchored on the USER's steer (the agent action is the last assistant turn of
``before``; the trigger is the steer itself and must never leak into model
input), while a sampled negative's window is anchored on the ASSISTANT turn
being judged (the trigger is the agent action).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_transcript.context import ContextWindow, TurnRef

NO_STEER = "NO_STEER"

DRAFT_CHAR_CAP = 10_000

# The Qwen3-2507 *Instruct* chat template renders every assistant turn with an
# empty ``<think>\n\n</think>`` scaffold (it is distilled from a thinking model
# and the template injects the block unconditionally). A LoRA trained through
# that template learns to emit the prefix, so greedy serving returns
# ``<think>\n\n</think>\n\n<content>`` — stripped at the generation source and
# again before any NO_STEER comparison.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and surrounding whitespace.

    Handles the degenerate ``'<think>\\n\\n</think>\\n\\nNO_STEER'`` collapse
    string and think-wrapped real steers alike; leaves think-free text untouched.
    """
    return _THINK_BLOCK.sub("", text).strip()


def split_of(session_id: str) -> str:
    """The deterministic session-hash group split every dataset view inherits."""
    return "test" if int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 10 == 0 else "train"


class Message(TypedDict):
    role: str
    content: str


def messages(turns: Sequence[TurnRef]) -> list[Message]:
    """One chat message per turn, from the capture-time previews."""
    return [{"role": turn.role, "content": turn.preview} for turn in turns]


def assistant_message(content: str) -> list[Message]:
    return [{"role": "assistant", "content": content}]


def agent_action_of(window: ContextWindow) -> str | None:
    """The last assistant turn preceding the anchor — what the user reacted to."""
    return next((turn.preview for turn in reversed(window.before) if turn.role == "assistant"), None)


def render_edit(file: str, old: str, new: str) -> str:
    return f"{file}\n```old\n{old}\n```\n```new\n{new}\n```"


def context_turns(window: ContextWindow) -> tuple[TurnRef, ...]:
    """The model-visible turns: everything up to the moment being judged.

    ``before`` always qualifies; the trigger joins it only when it is an
    assistant turn (an agent action, never a user reaction). A steering event's
    trigger is the user's steer — the label — and must never leak into input.
    Live-computable by construction: no judge output is ever part of the input.
    """
    if window.trigger is not None and window.trigger.role == "assistant":
        return (*window.before, window.trigger)
    return window.before


def watcher_prompt(window: ContextWindow) -> list[Message]:
    """The generative watcher's prompt: the context turns as chat messages."""
    return messages(context_turns(window))


def gate_text(window: ContextWindow) -> str:
    """The gate classifier's input: the watcher prompt flattened deterministically."""
    return "\n\n".join(f"<{message['role']}>\n{message['content']}" for message in watcher_prompt(window))


def tail_messages(prompt: Sequence[Message], cap: int = DRAFT_CHAR_CAP) -> list[Message]:
    """The most recent whole messages fitting in ``cap`` chars of content.

    The local watcher's input contract, shared verbatim with the lab's training
    materialization: walks backward from the latest turn; a message that would
    overflow the budget ends the walk, except that the latest message alone
    keeps its tail ``cap`` chars (a tool-heavy final turn must never evict the
    whole window).
    """
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    kept: list[Message] = []
    used = 0
    for message in reversed(prompt):
        if used + len(message["content"]) > cap:
            if not kept:
                kept.append({"role": message["role"], "content": message["content"][-cap:]})
            break
        kept.append(message)
        used += len(message["content"])
    return kept[::-1]


def truncated(window: ContextWindow, turns_back: int) -> ContextWindow | None:
    """The same window rewound ``turns_back`` turns: the state a watcher saw earlier.

    Drops the last ``turns_back`` turns of ``before``; None once nothing remains.
    """
    if turns_back == 0:
        return window
    if turns_back >= len(window.before):
        return None
    return dataclasses.replace(window, before=window.before[: len(window.before) - turns_back])
