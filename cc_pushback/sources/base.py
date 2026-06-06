from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Protocol

from cc_pushback.models import DedupKey

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cc_pushback.models import FeedbackCandidate
    from cc_pushback.repo import Repository

__all__ = [
    "ASK_USER_ANSWERED_PREFIX",
    "DENIAL_PREFIX",
    "INTERRUPT_RE",
    "PLAN_APPROVED_PREFIX",
    "USER_SAID_MARKER",
    "USER_SAID_TRAILER",
    "Source",
    "dedup_key",
]

DENIAL_PREFIX = "The user doesn't want to proceed with this tool use. The tool use was rejected"
USER_SAID_MARKER = "To tell you how to proceed, the user said:\n"
USER_SAID_TRAILER = "Note: The user's next message"
PLAN_APPROVED_PREFIX = "User has approved your plan"
ASK_USER_ANSWERED_PREFIX = "Your questions have been answered:"
INTERRUPT_RE = re.compile(r"\[Request interrupted by user(?: for tool use)?\]")


class Source(Protocol):
    """A detector that turns scanned material into feedback candidates."""

    def candidates(self, repo: Repository) -> Iterator[FeedbackCandidate]: ...


def dedup_key(*parts: str) -> DedupKey:
    """Returns the stable dedup key for ``parts``.

    Args:
        parts: The content fragments that uniquely identify a candidate.

    Returns:
        The SHA-256 hex digest of the parts joined by a null byte.
    """
    return DedupKey(hashlib.sha256("\x00".join(parts).encode()).hexdigest())
