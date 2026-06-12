"""cc-pushback's concrete code-review formats over the platform's parser infra.

The generic :class:`ReviewComment`/:class:`ReviewFormat` types and the
format-dispatch live in :mod:`cc_transcript.mining`; this module supplies
cc-pushback's policy — the three review formats it recognizes — and injects them
into the platform's :func:`extract_all`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cc_transcript.mining import ReviewComment, ReviewFormat
from cc_transcript.mining import extract_all as platform_extract_all

if TYPE_CHECKING:
    from collections.abc import Iterator

SUPERSET_INLINE_RE = re.compile(
    r"^In ((?=\S*[./]|\S+?:L)\S+?)(?::L(\d+)(?:-(\d+))?)?: (.+)$", re.MULTILINE
)
CONDUCTOR_FINDING_RE = re.compile(
    r"^- file: (?P<file>\S+?):(?P<line>\d+)\s*$"
    r"(?:\n- theme: .+$)?"
    r"(?:\n- claim: (?P<claim>.+)$)?"
    r"(?:\n- suggestion: (?P<suggestion>.+)$)?",
    re.MULTILINE,
)
CONDUCTOR_WORKSTREAM_HEADER_RE = re.compile(
    r"^### (?P<id>[A-Z][\w-]*\d*)\s*\[(?P<kind>[A-Z]+)\]\s*—\s*(?P<title>.+)$",
    re.MULTILINE,
)
WORKSTREAM_BODY_RE = re.compile(r"^(?:FIX|Tests): .+$", re.MULTILINE)


def extract_superset_inline(text: str) -> tuple[ReviewComment, ...]:
    return tuple(
        ReviewComment(
            file=match.group(1),
            line_start=int(match.group(2)) if match.group(2) else None,
            line_end=int(match.group(3)) if match.group(3) else None,
            comment=match.group(4).strip(),
        )
        for match in SUPERSET_INLINE_RE.finditer(text)
    )


def extract_conductor_finding(text: str) -> tuple[ReviewComment, ...]:
    return tuple(
        ReviewComment(
            file=match.group("file"),
            line_start=int(match.group("line")),
            line_end=None,
            comment=" ".join(part.strip() for part in (match.group("claim"), match.group("suggestion")) if part),
        )
        for match in CONDUCTOR_FINDING_RE.finditer(text)
    )


def extract_conductor_workstream(text: str) -> tuple[ReviewComment, ...]:
    headers = list(CONDUCTOR_WORKSTREAM_HEADER_RE.finditer(text))
    return tuple(
        ReviewComment(
            file=None,
            line_start=None,
            line_end=None,
            comment=" ".join(
                [f"{header.group('id')} [{header.group('kind')}] {header.group('title').strip()}"]
                + [line.group(0).strip() for line in WORKSTREAM_BODY_RE.finditer(text[header.end() : end])]
            ),
        )
        for header, end in zip(
            headers,
            [*(h.start() for h in headers[1:]), len(text)],
            strict=True,
        )
    )


def formats() -> tuple[ReviewFormat, ...]:
    return (
        ReviewFormat("superset-inline", SUPERSET_INLINE_RE, extract_superset_inline),
        ReviewFormat("conductor-finding", CONDUCTOR_FINDING_RE, extract_conductor_finding),
        ReviewFormat("conductor-workstream", CONDUCTOR_WORKSTREAM_HEADER_RE, extract_conductor_workstream),
    )


def extract_all(text: str) -> Iterator[tuple[ReviewFormat, ReviewComment]]:
    """Yields every ``(format, comment)`` extracted by any of cc-pushback's formats.

    Args:
        text: The raw review message text.

    Yields:
        One pair per extracted comment, across all formats whose pattern matches.
    """
    return platform_extract_all(text, formats())
