from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from collections.abc import Callable

    from cc_pushback.classify import FeedbackEvent

PatternName = NewType("PatternName", str)


def rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RegexMatcher:
    """Matches an event's verbatim text against a compiled pattern.

    Attributes:
        pattern: The regex tested against the feedback text.
    """

    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class StructuralMatcher:
    """Matches on the structure of a persisted event rather than its text.

    Attributes:
        predicate: A test over the event row, e.g. inspecting its payload.
    """

    predicate: Callable[[FeedbackEvent], bool]


@dataclass(frozen=True, slots=True)
class LlmMatcher:
    """A pattern only the language model can decide; the cheap pass never fires it.

    Attributes:
        hint: Optional guidance surfaced to the model for this pattern.
    """

    hint: str = ""


Matcher = RegexMatcher | StructuralMatcher | LlmMatcher


def matches(matcher: Matcher, event: FeedbackEvent) -> bool:
    """Tests ``event`` against ``matcher`` in the cheap, LLM-free pass.

    Args:
        matcher: The matcher to evaluate.
        event: The persisted feedback event to test.

    Returns:
        Whether the matcher fires. :class:`LlmMatcher` always returns ``False``
        here; it is resolved only by the language-model pass.
    """
    match matcher:
        case RegexMatcher(pattern=pattern):
            return pattern.search(event.text) is not None
        case StructuralMatcher(predicate=predicate):
            return predicate(event)
        case LlmMatcher():
            return False
