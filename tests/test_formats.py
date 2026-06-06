from __future__ import annotations

import pytest

from cc_pushback.formats import FORMATS, ReviewComment, extract_all

SUPERSET = """Review notes follow.
In src/foo.py:L10-20: do not add a fallback here
In bar.py:L5: ask before assuming
In baz.py: crash on the unexpected"""

FINDING = """- file: src/a.py:42
- theme: error-handling
- claim: this swallows the exception
- suggestion: let it propagate

- file: src/b.py:7
- claim: redundant guard"""

WORKSTREAM = """### W1 [BUG] — race in the queue
Some prose about the bug.
FIX: add a lock around the enqueue
Tests: cover concurrent producers

### W2 [STYLE] — naming
FIX: rename foo to bar"""


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        pytest.param(
            "superset-inline",
            SUPERSET,
            (
                ReviewComment("src/foo.py", 10, 20, "do not add a fallback here"),
                ReviewComment("bar.py", 5, None, "ask before assuming"),
                ReviewComment("baz.py", None, None, "crash on the unexpected"),
            ),
            id="superset-inline-multi",
        ),
        pytest.param(
            "conductor-finding",
            FINDING,
            (
                ReviewComment("src/a.py", 42, None, "this swallows the exception let it propagate"),
                ReviewComment("src/b.py", 7, None, "redundant guard"),
            ),
            id="conductor-finding-multi",
        ),
        pytest.param(
            "conductor-workstream",
            WORKSTREAM,
            (
                ReviewComment(
                    None,
                    None,
                    None,
                    "W1 [BUG] race in the queue FIX: add a lock around the enqueue Tests: cover concurrent producers",
                ),
                ReviewComment(None, None, None, "W2 [STYLE] naming FIX: rename foo to bar"),
            ),
            id="conductor-workstream-multi",
        ),
    ],
)
def test_format_extracts_expected_comments(name: str, text: str, expected: tuple[ReviewComment, ...]) -> None:
    fmt = next(f for f in FORMATS if f.name == name)

    assert fmt.extract(text) == expected


@pytest.mark.parametrize(
    "text",
    ["plain prose with no review markers", "- file: missing line number", "In: no path here"],
    ids=["prose", "no-line", "no-path"],
)
def test_formats_ignore_non_matching_text(text: str) -> None:
    assert list(extract_all(text)) == []


def test_extract_all_yields_format_and_comment_pairs() -> None:
    pairs = list(extract_all(SUPERSET))

    assert {fmt.name for fmt, _ in pairs} == {"superset-inline"}
    assert [comment.comment for _, comment in pairs] == [
        "do not add a fallback here",
        "ask before assuming",
        "crash on the unexpected",
    ]
