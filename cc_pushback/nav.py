"""Re-exports the transcript navigation helpers from the mining domain.

Deprecated: import these names from :mod:`cc_transcript.domains.mining`. This shim
keeps cc-pushback's historical import paths working for at least one release.
"""

from __future__ import annotations

from cc_transcript.domains.mining import (
    denial_results,
    denied_tool_payload,
    embedded_user_text,
    interrupt_marker,
    is_bare_interrupt_marker,
    last_edit_index,
    marker_in,
    next_user_message,
    tool_uses,
)

__all__ = [
    "denial_results",
    "denied_tool_payload",
    "embedded_user_text",
    "interrupt_marker",
    "is_bare_interrupt_marker",
    "last_edit_index",
    "marker_in",
    "next_user_message",
    "tool_uses",
]
