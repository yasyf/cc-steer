from __future__ import annotations

import pytest
from cc_transcript.mining.spec import regex_review_comments

from cc_pushback.formats import CONDUCTOR_FINDING_FMT, extract_superset_inline


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
    [comment] = regex_review_comments(CONDUCTOR_FINDING_FMT, text)
    assert comment.file == "pkg/mod.py"
    assert comment.line_start == 42
    assert comment.comment == "leaks a handle close it"


@pytest.mark.unit
def test_conductor_finding_drops_missing_claim_or_suggestion() -> None:
    [comment] = regex_review_comments(CONDUCTOR_FINDING_FMT, "- file: a.py:7\n- suggestion: extract a helper")
    assert comment.comment == "extract a helper"
