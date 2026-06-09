from __future__ import annotations

import pytest

from cc_pushback.formats import extract_all, extract_superset_inline


@pytest.mark.unit
def test_superset_inline_parses_file_and_line_range() -> None:
    [comment] = extract_superset_inline("In src/foo.py:L10-12: drop the guard")
    assert comment.file == "src/foo.py"
    assert comment.line_start == 10
    assert comment.line_end == 12
    assert comment.comment == "drop the guard"


@pytest.mark.unit
def test_superset_inline_single_line_has_no_end() -> None:
    [comment] = extract_superset_inline("In a/b.py:L5: fix this")
    assert (comment.line_start, comment.line_end) == (5, None)


@pytest.mark.unit
def test_conductor_finding_joins_claim_and_suggestion() -> None:
    text = "- file: pkg/mod.py:42\n- theme: correctness\n- claim: leaks a handle\n- suggestion: close it"
    [(fmt, comment)] = list(extract_all(text))
    assert fmt.name == "conductor-finding"
    assert comment.file == "pkg/mod.py"
    assert comment.line_start == 42
    assert comment.comment == "leaks a handle close it"


@pytest.mark.unit
def test_extract_all_yields_one_pair_per_comment() -> None:
    text = "In x/y.py:L1: a\nIn x/z.py:L2: b"
    assert [comment.comment for _, comment in extract_all(text)] == ["a", "b"]


@pytest.mark.unit
def test_extract_all_empty_when_no_format_matches() -> None:
    assert list(extract_all("just a normal sentence with no cites")) == []
