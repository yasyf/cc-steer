"""cc-pushback's event-filter policy, composed from cc-transcript primitives.

Keeps user turns that carry pushback worth learning from: drops structural noise,
agent-injected banners, trivial acknowledgements, very short control messages, and
sidechain/meta/compacted/empty turns. Unlike the sentiment filter, interrupt and
stop-hook markers are deliberately kept.
"""

from __future__ import annotations

from cc_transcript import (
    RESUME_PHRASE_SET,
    TRIVIAL_ACK_SET,
    USERS,
    FilterSpec,
    build_spec,
    drop_compacted,
    drop_empty,
    drop_junk,
    drop_meta_flag,
    drop_phrases,
    drop_short,
    drop_sidechain,
    keep_only,
)

PUSHBACK_SPEC: FilterSpec = build_spec(
    keep_only("user"),
    drop_sidechain(),
    drop_meta_flag("is_meta"),
    drop_compacted(),
    drop_empty(only_from=USERS),
    drop_junk("structural", "agent_injection"),
    drop_phrases(TRIVIAL_ACK_SET | RESUME_PHRASE_SET),
    drop_short(2),
)
