"""Re-exports the conversational-window primitive from the mining domain.

Deprecated: import these names from :mod:`cc_transcript.domains.mining`. This shim
keeps cc-pushback's historical import paths working for at least one release.
"""

from __future__ import annotations

from cc_transcript.domains.mining import ContextSnapshot, ContextTurn, build_snapshot, trigger_for, turn_for

__all__ = ["ContextSnapshot", "ContextTurn", "build_snapshot", "trigger_for", "turn_for"]
