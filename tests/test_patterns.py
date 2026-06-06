from __future__ import annotations

import pytest

from cc_pushback.matchers import LlmMatcher
from cc_pushback.patterns import PATTERNS, Pattern, by_name, render_taxonomy


def test_pattern_names_are_unique() -> None:
    names = [pattern.name for pattern in PATTERNS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("pattern", PATTERNS, ids=[pattern.name for pattern in PATTERNS])
def test_non_llm_patterns_have_triggers_and_llm_patterns_do_not(pattern: Pattern) -> None:
    if isinstance(pattern.matcher, LlmMatcher):
        assert pattern.triggers == ()
    elif pattern.name == "denied-edit":
        assert pattern.triggers == ()
    else:
        assert pattern.triggers


@pytest.mark.parametrize("pattern", PATTERNS, ids=[pattern.name for pattern in PATTERNS])
def test_by_name_round_trips(pattern: Pattern) -> None:
    assert by_name(pattern.name) is pattern


def test_by_name_unknown_raises_stop_iteration() -> None:
    with pytest.raises(StopIteration):
        by_name("does-not-exist")


def test_render_taxonomy_contains_every_name_and_rule() -> None:
    rendered = render_taxonomy()
    for pattern in PATTERNS:
        assert f"{pattern.name} — {pattern.rule}" in rendered
    assert rendered.count("\n") == len(PATTERNS) - 1
