"""Guardrail: the shared filter must never strip cc-pushback's reason to exist.

If a future cc-transcript change folds interrupt or stop-hook patterns into the
structural-noise group, these assertions fail loudly — that regression would
silently delete every interrupt/stop-hook correction cc-pushback learns from.
"""

from __future__ import annotations

from cc_transcript import (
    INTERRUPT_MARKER_RE,
    PUSHBACK_SPEC,
    STOP_HOOK_RE,
    STRUCTURAL_NOISE_RE,
    keep,
)

from tests.builders import parse, user_text


def user(text: str):  # noqa: ANN201 - test helper
    return parse([user_text(text)])[0]


def test_structural_noise_excludes_interrupt_and_stop_hook() -> None:
    assert STRUCTURAL_NOISE_RE.search("[Request interrupted by user] do it differently") is None
    assert STRUCTURAL_NOISE_RE.search("Stop hook feedback: ruff failed") is None


def test_dedicated_groups_match_their_markers() -> None:
    assert INTERRUPT_MARKER_RE.search("[Request interrupted by user]") is not None
    assert STOP_HOOK_RE.search("Stop hook feedback: x") is not None


def test_pushback_spec_keeps_interrupt_and_stop_hook() -> None:
    assert keep(user("[Request interrupted by user] no, do it this way"), PUSHBACK_SPEC)
    assert keep(user("Stop hook feedback: the build broke, revert that change"), PUSHBACK_SPEC)


def test_pushback_spec_still_drops_structural_noise() -> None:
    assert not keep(user("<system-reminder>injected</system-reminder>"), PUSHBACK_SPEC)
    assert not keep(user("<teammate-message>done</teammate-message>"), PUSHBACK_SPEC)
