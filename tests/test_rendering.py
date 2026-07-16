from __future__ import annotations

from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId

from cc_steer.rendering import (
    DRAFT_CHAR_CAP,
    NO_STEER,
    ask_block,
    context_turns,
    gate_text,
    has_substantive_gate_content,
    strip_think,
    structural_asks,
    tail_messages,
    truncated,
    watcher_prompt,
)
from cc_steer.retrain.evalset import GATE_ROLE_BLOCK

STEER = "no, use the other approach"


def turn(role: str, preview: str, uuid: str) -> TurnRef:
    ref = EventRef(session_id=SessionId("sess-1"), event_uuid=EventUuid(uuid), tool_use_id=None)
    return TurnRef(role=role, refs=(ref,), preview=preview, tool_digests=())  # type: ignore[arg-type]


def window(*, trigger: TurnRef | None, before: tuple[TurnRef, ...]) -> ContextWindow:
    anchor = EventRef(session_id=SessionId("sess-1"), event_uuid=EventUuid("anchor"), tool_use_id=None)
    return ContextWindow(anchor=anchor, before=before, trigger=trigger, after=(), fidelity="full", preview_chars=200)


BEFORE = (turn("user", "please add a test", "u1"), turn("user", "I added the test to the suite", "u2"))
USER_ANCHORED = window(trigger=turn("user", STEER, "t1"), before=BEFORE)
NO_TRIGGER = window(trigger=None, before=BEFORE)
ASSISTANT_ANCHORED = window(trigger=turn("assistant", "I will refactor everything", "t2"), before=BEFORE)


def test_user_steer_trigger_is_excluded_from_context() -> None:
    assert context_turns(USER_ANCHORED) == BEFORE
    assert context_turns(NO_TRIGGER) == BEFORE


def test_assistant_trigger_joins_the_context() -> None:
    turns = context_turns(ASSISTANT_ANCHORED)
    assert turns[-1].preview == "I will refactor everything"
    assert turns[:-1] == BEFORE


def test_watcher_prompt_is_context_messages_only() -> None:
    prompt = watcher_prompt(USER_ANCHORED)
    assert prompt == [
        {"role": "user", "content": "please add a test"},
        {"role": "user", "content": "I added the test to the suite"},
    ]


def test_gate_text_never_leaks_the_user_steer() -> None:
    text = gate_text(USER_ANCHORED)
    assert STEER not in text
    assert "<user>\nI added the test to the suite" in text
    assert text == gate_text(USER_ANCHORED)


EMPTY_REWIND = window(trigger=turn("user", STEER, "t1"), before=(turn("assistant", "", "a0"),))
EMPTY_NEGATIVE = window(trigger=None, before=(turn("assistant", "", "a0"),))


def test_has_substantive_gate_content_true_when_a_context_turn_carries_text() -> None:
    assert has_substantive_gate_content(USER_ANCHORED) is True


def test_has_substantive_gate_content_false_for_a_rewound_past_content_positive() -> None:
    # The user steer is the label (excluded), leaving only the empty leading
    # assistant turn — the exact bare-role-block that crashes freeze-eval.
    assert has_substantive_gate_content(EMPTY_REWIND) is False


def test_has_substantive_gate_content_false_for_an_empty_anchor_negative() -> None:
    assert has_substantive_gate_content(EMPTY_NEGATIVE) is False


def test_has_substantive_gate_content_agrees_with_the_freeze_eval_predicate() -> None:
    # The window-level predicate and the parquet text-regex freeze-eval uses must
    # never disagree, or an empty row slips past one gate into the other.
    for candidate in (USER_ANCHORED, NO_TRIGGER, ASSISTANT_ANCHORED, EMPTY_REWIND, EMPTY_NEGATIVE):
        assert has_substantive_gate_content(candidate) is bool(GATE_ROLE_BLOCK.sub("", gate_text(candidate)).strip())


def test_truncated_rewinds_before_turns() -> None:
    assert truncated(USER_ANCHORED, 0) is USER_ANCHORED
    rewound = truncated(USER_ANCHORED, 1)
    assert rewound is not None
    assert [item.preview for item in rewound.before] == ["please add a test"]
    assert rewound.trigger == USER_ANCHORED.trigger


def test_truncated_stops_when_nothing_remains() -> None:
    assert truncated(USER_ANCHORED, 2) is None
    assert truncated(USER_ANCHORED, 3) is None


def test_no_steer_sentinel_is_stable() -> None:
    assert NO_STEER == "NO_STEER"


def test_no_steer_sentinel_and_draft_cap_are_stable() -> None:
    assert DRAFT_CHAR_CAP == 10_000


def test_tail_messages_keeps_the_most_recent_whole_messages() -> None:
    prompt = [
        {"role": "user", "content": "a" * 60},
        {"role": "assistant", "content": "b" * 30},
        {"role": "user", "content": "c" * 30},
    ]
    assert tail_messages(prompt, 60) == prompt[1:]
    assert tail_messages(prompt, 200) == prompt


def test_tail_messages_boundary_is_inclusive_on_exact_fit() -> None:
    prompt = [{"role": "user", "content": "aaaa"}, {"role": "assistant", "content": "bb"}]
    assert tail_messages(prompt, 6) == prompt
    assert tail_messages(prompt, 5) == prompt[1:]


def test_tail_messages_oversized_final_message_keeps_its_tail() -> None:
    prompt = [{"role": "user", "content": "early"}, {"role": "assistant", "content": "x" * 50}]
    assert tail_messages(prompt, 10) == [{"role": "assistant", "content": "x" * 10}]


def test_tail_messages_rejects_a_nonpositive_cap() -> None:
    import pytest

    with pytest.raises(ValueError, match="cap"):
        tail_messages([], 0)


def test_strip_think_removes_the_template_scaffold() -> None:
    assert strip_think("<think>\n\n</think>\n\nNO_STEER") == "NO_STEER"
    assert strip_think("<think>plan</think>do the thing") == "do the thing"
    assert strip_think("plain steer") == "plain steer"


CLIPPED_ASK = (
    "assistant: on it\n"
    "AskUserQuestion([{'question': 'Which corpus should the clean baseline re-scan cover?', "
    "'header': 'Corpus', 'multiSelect': False, 'options': [{'label': 'Local only'}, "
    "{'label': 'Local + remote mirror (Recommended)'}, {'label': 'Remote onl…(+312ch))"
)


def test_ask_block_is_the_canonical_shape() -> None:
    block = ask_block(
        "Which corpus?", header="Corpus", options=("Local only", "Both"), recommended="Both"
    )
    assert block == "[assistant asked: Corpus] Which corpus?\n- Local only\n- Both\n(recommended: Both)"
    assert ask_block("Bare question") == "[assistant asked] Bare question"


def test_structural_asks_rewrites_a_clipped_fragment() -> None:
    rewritten = structural_asks(CLIPPED_ASK)
    assert "AskUserQuestion(" not in rewritten
    expected_head = "assistant: on it\n[assistant asked: Corpus] Which corpus should the clean baseline re-scan cover?"
    assert rewritten.startswith(expected_head)
    assert "- Local only" in rewritten
    assert "- Local + remote mirror" in rewritten
    assert "(recommended: Local + remote mirror)" in rewritten
    assert "Remote onl" not in rewritten  # the option the clip cut mid-string is dropped


def test_structural_asks_leaves_plain_text_and_unparseable_fragments_alone() -> None:
    assert structural_asks("assistant: no questions here") == "assistant: no questions here"
    assert structural_asks("AskUserQuestion(<clipped before any field>") == "AskUserQuestion(<clipped before any field>"


def test_structural_asks_salvages_a_question_the_clip_cut_mid_string() -> None:
    fragment = "AskUserQuestion([{'question': 'The lineage dashboard already exists. What does updated web…(+1262ch))"
    rewritten = structural_asks(fragment)
    assert rewritten == "[assistant asked] The lineage dashboard already exists. What does updated web…"


def test_structural_asks_handles_multi_question_asks() -> None:
    fragment = (
        "AskUserQuestion([{'question': 'First?', 'header': 'A', 'options': [{'label': 'x'}]}, "
        "{'question': 'Second?', 'header': 'B', 'options': [{'label': 'y'}]}])"
    )
    rewritten = structural_asks(fragment)
    assert "[assistant asked: A] First?" in rewritten
    assert "[assistant asked: B] Second?" in rewritten
    # option-to-question association is ambiguous under the clip; options are omitted
    assert "- x" not in rewritten


def test_watcher_prompt_v2_rewrites_asks_and_v1_stays_raw() -> None:
    window = ContextWindow(
        anchor=EventRef(SessionId("s"), EventUuid("a0")),
        before=(TurnRef(role="assistant", refs=(), preview=CLIPPED_ASK, tool_digests=()),),
        trigger=None,
        after=(),
        fidelity="full",
        preview_chars=200,
    )
    assert watcher_prompt(window)[0]["content"] == CLIPPED_ASK
    assert watcher_prompt(window, render_version=1)[0]["content"] == CLIPPED_ASK
    v2 = watcher_prompt(window, render_version=2)[0]["content"]
    assert "[assistant asked: Corpus]" in v2 and "AskUserQuestion(" not in v2


def test_structural_asks_handles_double_quoted_repr_values() -> None:
    fragment = (
        "AskUserQuestion([{'question': \"What should the task 'diagram' depict?\", 'header': 'Diagram', "
        "'options': [{'label': 'Flow diagram per workf…(+90ch))"
    )
    rewritten = structural_asks(fragment)
    assert rewritten.startswith("[assistant asked: Diagram] What should the task 'diagram' depict?")
    assert "AskUserQuestion(" not in rewritten
