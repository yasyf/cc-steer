from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["FORMATS", "ReviewComment", "ReviewFormat", "extract_all"]

SUPERSET_INLINE_RE = re.compile(r"^In (\S+?)(?::L(\d+)(?:-(\d+))?)?: (.+)$", re.MULTILINE)
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


@dataclass(frozen=True, slots=True)
class ReviewComment:
    """A single inline review comment parsed from a code-review message.

    Attributes:
        file: The file the comment targets, when cited.
        line_start: The first line the comment targets, when cited.
        line_end: The last line the comment targets, when a range is cited.
        comment: The comment's text.
    """

    file: str | None
    line_start: int | None
    line_end: int | None
    comment: str


@dataclass(frozen=True, slots=True)
class ReviewFormat:
    """A named code-review text format with a detector and extractor.

    Attributes:
        name: The format's identifier.
        pattern: A pattern that matches when the format is present in a text.
        extract: Parses a matching text into its review comments.
    """

    name: str
    pattern: re.Pattern[str]
    extract: Callable[[str], tuple[ReviewComment, ...]]


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


FORMATS: tuple[ReviewFormat, ...] = (
    ReviewFormat("superset-inline", SUPERSET_INLINE_RE, extract_superset_inline),
    ReviewFormat("conductor-finding", CONDUCTOR_FINDING_RE, extract_conductor_finding),
    ReviewFormat("conductor-workstream", CONDUCTOR_WORKSTREAM_HEADER_RE, extract_conductor_workstream),
)


def extract_all(text: str) -> Iterator[tuple[ReviewFormat, ReviewComment]]:
    """Yields every ``(format, comment)`` extracted by any matching format.

    Args:
        text: The raw review message text.

    Yields:
        One pair per extracted comment, across all formats whose pattern matches.
    """
    return (
        (fmt, comment)
        for fmt in FORMATS
        if fmt.pattern.search(text)
        for comment in fmt.extract(text)
    )
