"""cc-pushback's concrete code-review formats over the platform's mining policy.

The generic :class:`ReviewComment`/:class:`StructuredFormat` types and the
format-dispatch live in :mod:`cc_transcript.mining`; this module supplies
cc-pushback's policy — the three review formats it recognizes and the structured
finding formats it lifts — assembled into a :class:`ReviewSpec` for the
review-comment detector. ``conductor-finding`` is a portable
:class:`RegexReviewFormat`; ``superset-inline`` (lookahead) and
``conductor-workstream`` (multi-pass) are :class:`CallableReviewFormat` escape
hatches that run in Python.
"""

from __future__ import annotations

import re

from cc_transcript.mining import (
    CallableReviewFormat,
    RegexReviewFormat,
    ReviewComment,
    ReviewSpec,
    StructuredFormat,
)

SUPERSET_INLINE_RE = re.compile(
    r"^In ((?=\S*[./]|\S+?:L)\S+?)(?::L(\d+)(?:-(\d+))?)?: (.+)$", re.MULTILINE
)
CONDUCTOR_WORKSTREAM_HEADER_RE = re.compile(
    r"^### (?P<id>[A-Z][\w-]*\d*)\s*\[(?P<kind>[A-Z]+)\]\s*—\s*(?P<title>.+)$",
    re.MULTILINE,
)
WORKSTREAM_BODY_RE = re.compile(r"^(?:FIX|Tests): .+$", re.MULTILINE)
CONDUCTOR_FINDING_FMT = RegexReviewFormat(
    name="conductor-finding",
    groups=(
        (
            "conductor-finding",
            r"^- file: (\S+?):(\d+)\s*$"
            r"(?:\n- theme: .+$)?"
            r"(?:\n- claim: (.+)$)?"
            r"(?:\n- suggestion: (.+)$)?",
        ),
    ),
    file_group=1,
    line_start_group=2,
    line_end_group=None,
    comment_groups=(3, 4),
    join=" ",
    multiline=True,
    ignore_case=False,
)


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


def structured_formats() -> tuple[StructuredFormat, ...]:
    return (
        StructuredFormat(
            name="workflow-finding",
            file_keys=("file", "path", "file_path", "location"),
            line_keys=("line", "line_start", "lines"),
            comment_keys=("comment", "message", "description", "evidence", "detail", "problem", "why", "issue"),
            fix_keys=("suggested_fix", "suggestion", "fix"),
            finding_keys=("confirmedHigh", "confirmedCritical"),
        ),
    )


def review_spec() -> ReviewSpec:
    return ReviewSpec(
        regex_formats=(CONDUCTOR_FINDING_FMT,),
        callable_formats=(
            CallableReviewFormat("superset-inline", SUPERSET_INLINE_RE, extract_superset_inline),
            CallableReviewFormat("conductor-workstream", CONDUCTOR_WORKSTREAM_HEADER_RE, extract_conductor_workstream),
        ),
        structured_formats=structured_formats(),
        surfaces=frozenset({"typed", "surfaced"}),
    )
