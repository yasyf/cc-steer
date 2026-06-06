from __future__ import annotations

from pathlib import Path

from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.transcripts import ReviewComments, TranscriptMessages
from tests.builders import assistant_text, parse, user_text


def candidates(entries: list[dict[str, object]]) -> list[FeedbackCandidate]:
    return list(TranscriptMessages().candidates_for_file(Path("/t.jsonl"), parse(entries)))


def test_keeps_genuine_user_pushback() -> None:
    cands = candidates([user_text("don't add a fallback, crash instead"), assistant_text("ok")])

    assert len(cands) == 1
    assert cands[0].text == "don't add a fallback, crash instead"
    assert cands[0].source_kind == "transcript_message"


def test_drops_system_reminder_junk() -> None:
    assert candidates([user_text("<system-reminder>injected</system-reminder>")]) == []


def test_drops_command_message_junk() -> None:
    assert candidates([user_text("<command-message>run tests</command-message>")]) == []


def test_drops_scheduled_task_automated_run() -> None:
    assert (
        candidates([user_text('<scheduled-task name="x" file="/s">\nautomated run\n</scheduled-task>')]) == []
    )


def test_drops_empty_user_text() -> None:
    assert candidates([user_text("   ")]) == []


def test_skips_sidechain_and_meta() -> None:
    cands = candidates(
        [user_text("sidechain note", isSidechain=True), user_text("meta note", isMeta=True), user_text("real one")]
    )

    assert [c.text for c in cands] == ["real one"]


def test_interrupt_marker_text_not_junk_filtered_here() -> None:
    cands = candidates([user_text("revise this [Request interrupted by user] please")])

    assert len(cands) == 1
    assert "[Request interrupted by user]" in cands[0].text


def review_candidates(entries: list[dict[str, object]]) -> list[FeedbackCandidate]:
    return list(ReviewComments().candidates_for_file(Path("/t.jsonl"), parse(entries)))


def test_review_comments_explode_superset_inline_message() -> None:
    message = (
        "In src/captain_hook/tools.py:L55: which is it? dont just guess\n"
        "In src/captain_hook/tasks.py:L13: classvar"
    )

    cands = review_candidates([user_text(message), assistant_text("ok")])

    assert [c.text for c in cands] == ["which is it? dont just guess", "classvar"]
    assert all(c.source_kind == "review_comment" for c in cands)
    assert cands[0].payload == {
        "format": "superset-inline",
        "file": "src/captain_hook/tools.py",
        "line_start": 55,
        "line_end": None,
    }
    assert len({c.dedup_key for c in cands}) == 2


def test_review_comments_skip_plain_messages() -> None:
    assert review_candidates([user_text("don't add a fallback, crash instead")]) == []


def test_drops_compact_continuation_summaries() -> None:
    entry = user_text(
        "This session is being continued from a previous conversation that ran out of context.",
        isCompactSummary=True,
        isVisibleInTranscriptOnly=True,
    )

    assert candidates([entry]) == []


def test_drops_teammate_messages() -> None:
    assert candidates([user_text('<teammate-message teammate_id="be-events">done</teammate-message>')]) == []


def test_drops_third_party_agent_prompts() -> None:
    assert candidates([user_text("# Augment Agent\n\nYou are Augment Agent, an AI coding assistant.")]) == []
