from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cc_steer.decisions import (
    DecisionRow,
    mine,
    read_decisions,
    selection,
    stats_of,
    write_decisions,
)
from tests.builders import assistant_tool_use, tool_result, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path

SESSION = "11111111-1111-1111-1111-111111111111"

SINGLE_QUESTION = {
    "question": "Scope of cleanup?",
    "header": "Scope",
    "multiSelect": False,
    "options": [{"label": "Just A"}, {"label": "A and B (Recommended)"}, {"label": "Everything"}],
}
MULTI_QUESTION = {
    "question": "Which extras?",
    "header": "Extras",
    "multiSelect": True,
    "options": [{"label": "Docs"}, {"label": "PyPI"}, {"label": "CI"}],
}
CUSTOM_QUESTION = {
    "question": "Proceed now?",
    "header": "Go",
    "multiSelect": False,
    "options": [{"label": "Yes"}, {"label": "No"}],
}
LEGACY_QUESTION = {"question": "Legacy?", "header": "L", "multiSelect": False, "options": [{"label": "X"}]}

OCCURRED_AT = "2026-06-01T12:00:00+00:00"


def fixture_entries() -> list[dict[str, object]]:
    """A session with a two-round ask, a custom-answer ask, and one legacy (pre-typed) ask."""
    return [
        user_text("start", uuid="u0"),
        assistant_tool_use("toolu_1", "AskUserQuestion", {"questions": [SINGLE_QUESTION, MULTI_QUESTION]}, uuid="a1"),
        tool_result(
            "toolu_1",
            "ok",
            uuid="r1",
            toolUseResult={
                "questions": [SINGLE_QUESTION, MULTI_QUESTION],
                "answers": {"Scope of cleanup?": "Everything", "Which extras?": "Docs, CI"},
                "annotations": {},
            },
        ),
        assistant_tool_use("toolu_2", "AskUserQuestion", {"questions": [CUSTOM_QUESTION]}, uuid="a2"),
        tool_result(
            "toolu_2",
            "ok",
            uuid="r2",
            toolUseResult={
                "questions": [CUSTOM_QUESTION],
                "answers": {"Proceed now?": "1, but clean up the imports first"},
                "annotations": {},
            },
        ),
        assistant_tool_use("toolu_3", "AskUserQuestion", {"questions": [LEGACY_QUESTION]}, uuid="a3"),
        tool_result("toolu_3", "User rejected tool use", uuid="r3", toolUseResult="User rejected tool use"),
    ]


@pytest.fixture
def mined(tmp_path: Path):
    write_transcript(tmp_path / "project" / f"{SESSION}.jsonl", fixture_entries())
    return mine(tmp_path)


EXPECTED = {
    "single": DecisionRow(
        id="5245e1a28daac050",
        session_id=SESSION,
        occurred_at=OCCURRED_AT,
        turn_index=0,
        event_uuid="a1",
        tool_use_id="toolu_1",
        question="Scope of cleanup?",
        header="Scope",
        options=("Just A", "A and B (Recommended)", "Everything"),
        multi_select=False,
        answer="Everything",
        chosen_index=(2,),
        is_custom=False,
        split="train",
    ),
    "multi": DecisionRow(
        id="6417d73c234dca84",
        session_id=SESSION,
        occurred_at=OCCURRED_AT,
        turn_index=0,
        event_uuid="a1",
        tool_use_id="toolu_1",
        question="Which extras?",
        header="Extras",
        options=("Docs", "PyPI", "CI"),
        multi_select=True,
        answer="Docs, CI",
        chosen_index=(0, 2),
        is_custom=False,
        split="train",
    ),
    "custom": DecisionRow(
        id="2123412ca7866a4a",
        session_id=SESSION,
        occurred_at=OCCURRED_AT,
        turn_index=0,
        event_uuid="a2",
        tool_use_id="toolu_2",
        question="Proceed now?",
        header="Go",
        options=("Yes", "No"),
        multi_select=False,
        answer="1, but clean up the imports first",
        chosen_index=(),
        is_custom=True,
        split="train",
    ),
}


def test_mine_counts_rounds_and_quarantines_legacy(mined) -> None:
    assert len(mined.rows) == 3
    assert len(mined.quarantined) == 1


@pytest.mark.parametrize("case", ["single", "multi", "custom"], ids=["single", "multi", "custom"])
def test_row_fields(mined, case: str) -> None:
    by_question = {row.question: row for row in mined.rows}
    assert by_question[EXPECTED[case].question] == EXPECTED[case]


def test_legacy_round_is_quarantined_not_mined(mined) -> None:
    quarantine = mined.quarantined[0]
    assert quarantine.result_type == "TextResult"
    assert quarantine.event_uuid == "a3"
    assert quarantine.tool_use_id == "toolu_3"
    assert quarantine.session_id == SESSION
    assert "toolu_3" not in {row.tool_use_id for row in mined.rows}


COMMA_LABEL_OPTIONS = (
    "cc-transcript v7.0.2 (Recommended)",
    "Pure trio: spawnllm, captain-hook, docker-dsl",
    "cc-sentiment (push = release v0.2.127)",
    "dailies v0.1.0",
)
COMMA_LABEL_ANSWER = (
    "cc-transcript v7.0.2 (Recommended), "
    "Pure trio: spawnllm, captain-hook, docker-dsl, "
    "cc-sentiment (push = release v0.2.127)"
)


@pytest.mark.parametrize(
    ("answer", "options", "multi_select", "chosen_index", "is_custom"),
    [
        pytest.param("Everything", ("Just A", "Everything"), False, (1,), False, id="single-exact"),
        pytest.param("type my own", ("Just A", "Everything"), False, (), True, id="single-custom"),
        pytest.param("Docs, CI", ("Docs", "PyPI", "CI"), True, (0, 2), False, id="multi-clean"),
        pytest.param("Docs, also do X", ("Docs", "PyPI"), True, (0,), True, id="multi-custom"),
        pytest.param(COMMA_LABEL_ANSWER, COMMA_LABEL_OPTIONS, True, (0, 1, 2), False, id="multi-comma-in-label"),
        pytest.param(
            f"{COMMA_LABEL_ANSWER}, and also frobnicate",
            COMMA_LABEL_OPTIONS,
            True,
            (0, 1, 2),
            True,
            id="multi-comma-in-label-plus-custom",
        ),
        pytest.param("", ("env", "superset"), True, (), False, id="multi-empty-no-selection"),
        pytest.param(None, ("Docs", "PyPI"), True, (), False, id="unanswered"),
    ],
)
def test_selection(
    answer: str | None,
    options: tuple[str, ...],
    multi_select: bool,
    chosen_index: tuple[int, ...],
    is_custom: bool,
) -> None:
    assert selection(answer, options, multi_select=multi_select) == (chosen_index, is_custom)


def test_write_read_roundtrip_stamps_digest_and_quarantine(mined, tmp_path: Path) -> None:
    out = tmp_path / "decisions.parquet"
    digest = write_decisions(mined, out)
    rows, stored_digest, quarantined = read_decisions(out)
    assert stored_digest == digest
    assert quarantined == 1
    assert {row.id: row for row in rows} == {row.id: row for row in mined.rows}


def test_stats_counts_splits_and_shares(mined) -> None:
    stats = stats_of(list(mined.rows), quarantined=len(mined.quarantined))
    assert stats.total == 3
    assert stats.by_split == {"train": 3}
    assert stats.multi_select == 1
    assert stats.custom == 1
    assert stats.quarantined == 1
    assert stats.to_dict() == {
        "total": 3,
        "by_split": {"train": 3},
        "multi_select": 1,
        "custom": 1,
        "quarantined": 1,
    }
