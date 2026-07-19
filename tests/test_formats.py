from __future__ import annotations

import pytest
from cc_transcript.mining import MiningSpec, ReviewSpec, mine

from cc_steer.formats import CONDUCTOR_FINDING_FMT, extract_superset_inline
from tests.builders import parse, user_text


def conductor_signals(text: str) -> list:
    spec = MiningSpec(review=ReviewSpec(regex_formats=(CONDUCTOR_FINDING_FMT,)))
    return [sig for sig in mine(parse([user_text(text)]), spec) if sig.detector == "review_comment"]


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
    [sig] = conductor_signals(text)
    assert sig.evidence["file"] == "pkg/mod.py"
    assert sig.evidence["line_start"] == 42
    assert sig.text == "leaks a handle close it"


@pytest.mark.unit
def test_conductor_finding_drops_missing_claim_or_suggestion() -> None:
    [sig] = conductor_signals("- file: a.py:7\n- suggestion: extract a helper")
    assert sig.text == "extract a helper"
