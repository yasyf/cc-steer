from __future__ import annotations

from pathlib import Path

from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.transcripts import TranscriptMessages
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
