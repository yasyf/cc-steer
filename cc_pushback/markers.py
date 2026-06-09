"""Re-exports the transcript marker constants from the mining domain.

Deprecated: import these names from :mod:`cc_transcript.domains.mining`. This shim
keeps cc-pushback's historical import paths working for at least one release.
"""

from __future__ import annotations

from cc_transcript.domains.mining import (
    DENIAL_PREFIX,
    EDIT_TOOLS,
    INTERRUPT_MARKER_RE,
    REENTRY_LOOKBACK,
    USER_SAID_MARKER,
    USER_SAID_TRAILER,
)

__all__ = [
    "DENIAL_PREFIX",
    "EDIT_TOOLS",
    "INTERRUPT_MARKER_RE",
    "REENTRY_LOOKBACK",
    "USER_SAID_MARKER",
    "USER_SAID_TRAILER",
]
