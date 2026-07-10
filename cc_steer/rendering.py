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


def watcher_prompt(window: ContextWindow, *, render_version: int = 1) -> list[Message]:
    """The generative watcher's prompt: the context turns as chat messages.

    ``render_version`` is the training contract a watcher model was built on,
    carried in its registry metadata: v1 is the raw capture-time previews; v2
    rewrites ``AskUserQuestion(...)`` preview fragments into structural ask
    blocks (:func:`structural_asks`). Promoting a model flips the live
    rendering with it — never change versions independently of the model.
    """
    rendered = messages(context_turns(window))
    if render_version >= 2:
        return structural_ask_messages(rendered)
    return rendered


# The capture-time preview of an AskUserQuestion turn is the tool call's
# clipped single-line repr: ``AskUserQuestion([{'question': ..., 'header': ...,
# 'options': [{'label': ...}, ...]}])``, possibly ending in clip()'s
# ``…(+Nch)`` marker. Values repr with either quote (double when the text
# holds an apostrophe). Field regexes tolerate the clip: whatever survived is
# rendered, whatever was cut is dropped.
_ASK_FRAGMENT = re.compile(r"AskUserQuestion\(.*", re.MULTILINE)


def _field(name: str, *, closed: bool = True) -> re.Pattern[str]:
    single = r"'((?:[^'\\]|\\.)*)" + ("'" if closed else "$")
    double = r'"((?:[^"\\]|\\.)*)' + ('"' if closed else "$")
    return re.compile(rf"'{name}':\s*(?:{single}|{double})")


_ASK_QUESTION = _field("question")
_ASK_HEADER = _field("header")
_ASK_LABEL = _field("label")
_ASK_PARTIAL_QUESTION = _field("question", closed=False)
_CLIP_TAIL = re.compile(r"…\(\+\d+ch\)\)*$")
_RECOMMENDED_SUFFIX = " (Recommended)"


def ask_block(question: str, *, header: str = "", options: Sequence[str] = (), recommended: str = "") -> str:
    """The canonical structural rendering of one assistant ask.

    The single formatter both render paths share — the preview rewrite
    (:func:`structural_asks`) and the export's payload-derived block — so the
    watcher sees one shape for "the assistant asked the user something"
    regardless of which side rendered it. Never pass the user's pick; the
    answer is the training label.
    """
    tag = f"[assistant asked: {header}]" if header else "[assistant asked]"
    lines = [f"{tag} {question}".rstrip()]
    lines.extend(f"- {option}" for option in options)
    if recommended:
        lines.append(f"(recommended: {recommended})")
    return "\n".join(lines)


def structural_asks(content: str) -> str:
    """Rewrite every ``AskUserQuestion(...)`` preview fragment into ask blocks.

    Tolerant of the capture-time clip: fields that survived render, fields
    that were cut are omitted, and a fragment with no recoverable question is
    left untouched.
    """
    return _ASK_FRAGMENT.sub(_rewrite_fragment, content)


def structural_ask_messages(rendered: Sequence[Message]) -> list[Message]:
    return [{"role": message["role"], "content": structural_asks(message["content"])} for message in rendered]


def _rewrite_fragment(match: re.Match[str]) -> str:
    fragment = match.group(0)
    questions = _matches(_ASK_QUESTION, fragment)
    if not questions:
        # The clip cut the (first) question mid-string: salvage what survived.
        salvaged = _matches(_ASK_PARTIAL_QUESTION, _CLIP_TAIL.sub("", fragment))
        if salvaged and salvaged[0].strip():
            return ask_block(f"{salvaged[0]}…")
        return fragment
    headers = _matches(_ASK_HEADER, fragment)
    labels = _matches(_ASK_LABEL, fragment)
    options = [label.removesuffix(_RECOMMENDED_SUFFIX) for label in labels]
    recommended = next(
        (label.removesuffix(_RECOMMENDED_SUFFIX) for label in labels if label.endswith(_RECOMMENDED_SUFFIX)), ""
    )
    blocks = [
        ask_block(
            question,
            header=headers[index] if index < len(headers) else "",
            options=options if len(questions) == 1 else (),
            recommended=recommended if len(questions) == 1 else "",
        )
        for index, question in enumerate(questions)
    ]
    return "\n".join(blocks)


def _matches(pattern: re.Pattern[str], fragment: str) -> list[str]:
    return [_unescape(single or double) for single, double in pattern.findall(fragment)]


def _unescape(text: str) -> str:
    return text.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


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
