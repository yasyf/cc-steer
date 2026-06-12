from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from cc_transcript.activity import Edit, SessionActivity
from cc_transcript.evidence import EXTRACTOR_VERSION, CandidatePair, GitFix
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.judge import resolved_model
from cc_transcript.tools import Hunk
from pydantic import ValidationError

from cc_pushback.detectors import detect
from cc_pushback.enrich import (
    ENRICH_VERSION,
    CodeEvidence,
    EditSide,
    build_enrich_prompt,
    correction_source,
    enrich,
)
from cc_pushback.refine import PROMPT_VERSION as REFINE_VERSION
from cc_pushback.refine import RefinedPair, Refinement, refine
from cc_pushback.triage import Verdict, triage
from tests.builders import SESSION, assistant_text, assistant_tool_use, parse, tool_result, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from cc_pushback.store import FeedbackStore

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"
MODEL = resolved_model("medium")
PUSHBACK = "no, this skips validation, this is wrong"

INCORRECT_OLD = "def parse(blob):\n    return blob"
INCORRECT_NEW = "def parse(blob):\n    return json.loads(blob)"
CORRECT_NEW = "def parse(blob):\n    data = json.loads(blob)\n    validate(data)\n    return data"

PROMPT_ROW = {
    "action": "rewrote the parser without validation",
    "complaint": "the change skips validation",
    "complaint_verbatim": PUSHBACK,
}


def coding_entries() -> list[dict[str, Any]]:
    return [
        user_text("tighten the parser"),
        assistant_text("editing the parser"),
        assistant_tool_use(
            "t1", "Edit", {"file_path": "/repo/app.py", "old_string": INCORRECT_OLD, "new_string": INCORRECT_NEW}
        ),
        tool_result("t1", "ok"),
        user_text(PUSHBACK),
        assistant_text("restoring validation"),
        assistant_tool_use(
            "t2", "Edit", {"file_path": "/repo/app.py", "old_string": INCORRECT_NEW, "new_string": CORRECT_NEW}
        ),
        tool_result("t2", "ok"),
    ]


def make_edit(old: str, new: str, *, turn: int, uuid: str) -> Edit:
    return Edit(
        file_path="/repo/app.py",
        hunks=(Hunk(old, new),),
        tool="Edit",
        ref=EventRef(SessionId(SESSION), EventUuid(uuid)),
        turn_index=turn,
        ts=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )


def git_fix(old: str, new: str, *, commit: str = "abc123") -> GitFix:
    return GitFix(
        file_path="/repo/app.py",
        hunks=(Hunk(old, new),),
        commit=commit,
        committed_at=datetime(2026, 6, 2, tzinfo=UTC),
    )


def code_evidence(*, correct: EditSide | None) -> CodeEvidence:
    return CodeEvidence(
        kind="code",
        file_path="/repo/app.py",
        incorrect_edit=EditSide(old=INCORRECT_OLD, new=INCORRECT_NEW),
        correct_edit=correct,
        note="the complaint names the unvalidated parse",
    )


async def seed_refined(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, Any]], *, pushback: str = PUSHBACK
) -> None:
    await store.record_file_scan(FILE, 1.0, detect(parse(entries)))

    async def judge(prompt: str) -> Verdict:
        accepted = f"USER MESSAGE TO CLASSIFY ===\n{pushback}" in prompt
        return Verdict(
            category="incorrect_change" if accepted else "status_update",
            what_claude_did="edited the parser",
            confidence=0.9,
            rationale="r",
        )

    monkeypatch.setattr("cc_pushback.triage.structured_judge", lambda *_, **__: judge)
    await triage(store)

    async def refiner(prompt: str) -> Refinement:
        return Refinement(
            pairs=[
                RefinedPair(
                    action=PROMPT_ROW["action"],
                    complaint_verbatim=pushback,
                    complaint=PROMPT_ROW["complaint"],
                )
            ]
        )

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: refiner)
    report = await refine(store)
    assert report.refined == 1


@pytest.mark.unit
def test_build_enrich_prompt_annotates_git_distance_and_likely_fix() -> None:
    pairs = (
        CandidatePair(
            incorrect=make_edit("x = 1", "x = 2", turn=3, uuid="u-a"),
            correction=git_fix("x = 2", "x = 3"),
            overlap=0.5,
        ),
        CandidatePair(incorrect=make_edit("y = 1", "y = 2", turn=1, uuid="u-b"), correction=None, overlap=0.0),
    )
    prompt = build_enrich_prompt(PROMPT_ROW, pairs, anchor_turn=5)
    assert "--- candidate 1: /repo/app.py (Edit, 2 turn(s) before the pushback) ---" in prompt
    assert "--- candidate 2: /repo/app.py (Edit, 4 turn(s) before the pushback) ---" in prompt
    assert "correction (git abc123, overlap 0.50) [likely fix]:" in prompt
    assert "no correction found" in prompt
    assert "- x = 1" in prompt and "+ x = 2" in prompt
    assert "- x = 2" in prompt and "+ x = 3" in prompt
    assert "[what the assistant did: rewrote the parser without validation]" in prompt
    assert prompt.rstrip().endswith(PUSHBACK)


@pytest.mark.unit
def test_build_enrich_prompt_clips_each_hunk_at_600_chars() -> None:
    long_new = "x" * 700
    pairs = (CandidatePair(incorrect=make_edit("a", long_new, turn=0, uuid="u-a"), correction=None, overlap=0.0),)
    prompt = build_enrich_prompt(PROMPT_ROW, pairs, anchor_turn=1)
    assert f"+ {'x' * 600}…(+100ch)" in prompt
    assert "x" * 601 not in prompt


@pytest.mark.unit
def test_correction_source_distinguishes_git_session_and_none() -> None:
    pairs = (
        CandidatePair(
            incorrect=make_edit(INCORRECT_OLD, INCORRECT_NEW, turn=0, uuid="u-a"),
            correction=git_fix(INCORRECT_NEW, CORRECT_NEW),
            overlap=0.4,
        ),
    )
    git_pick = code_evidence(correct=EditSide(old=INCORRECT_NEW, new=CORRECT_NEW))
    assert correction_source(pairs, git_pick) == "git"
    session_pick = code_evidence(correct=EditSide(old="something else", new=CORRECT_NEW))
    assert correction_source(pairs, session_pick) == "session"
    assert correction_source(pairs, CodeEvidence(kind="no_code", note="n")) is None


@pytest.mark.unit
def test_code_evidence_requires_a_file_and_an_edit() -> None:
    with pytest.raises(ValidationError):
        CodeEvidence(kind="code", note="n")
    assert CodeEvidence(kind="no_code", note="n").incorrect_edit is None


@pytest.mark.integration
async def test_enrich_links_a_real_harvest_then_noop_then_extractor_bump_rederives(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)
    prompts: list[str] = []

    async def linker(prompt: str) -> CodeEvidence:
        prompts.append(prompt)
        return code_evidence(correct=EditSide(old=INCORRECT_NEW, new=CORRECT_NEW))

    monkeypatch.setattr("cc_pushback.enrich.structured_judge", lambda *_, **__: linker)
    report = await enrich(store)
    assert (report.enriched, report.code, report.no_code, report.git, report.failed, report.pending) == (
        1,
        1,
        0,
        0,
        0,
        0,
    )

    assert len(prompts) == 1
    prompt = prompts[0]
    for line in (*INCORRECT_OLD.splitlines(), *INCORRECT_NEW.splitlines()):
        assert f"- {line}" in prompt or f"+ {line}" in prompt
    assert f"+ {CORRECT_NEW.splitlines()[1]}" in prompt
    assert "--- candidate 1: /repo/app.py (Edit, 1 turn(s) before the pushback) ---" in prompt
    assert "correction (same session, 1 turn(s) later, overlap 1.00) [likely fix]:" in prompt
    assert "[the complaint: the change skips validation]" in prompt
    assert prompt.rstrip().endswith(PUSHBACK)

    row = (await store.pairs())[0]
    assert (row["evidence_kind"], row["evidence_file_path"]) == ("code", "/repo/app.py")
    assert row["evidence_source"] == "session"
    assert (row["incorrect_old"], row["incorrect_new"]) == (INCORRECT_OLD, INCORRECT_NEW)
    assert (row["correct_old"], row["correct_new"]) == (INCORRECT_NEW, CORRECT_NEW)
    assert (row["enrich_version"], row["enrich_model"], row["extractor_version"]) == (
        ENRICH_VERSION,
        MODEL,
        EXTRACTOR_VERSION,
    )

    again = await enrich(store)
    assert (again.enriched, again.failed, again.pending) == (0, 0, 0)
    assert len(prompts) == 1

    monkeypatch.setattr("cc_pushback.enrich.EXTRACTOR_VERSION", EXTRACTOR_VERSION + 1)
    rederived = await enrich(store)
    assert (rederived.enriched, rederived.pending) == (1, 0)
    assert len(prompts) == 2
    assert (await store.pairs())[0]["extractor_version"] == EXTRACTOR_VERSION + 1


@pytest.mark.integration
async def test_expired_transcript_persists_a_no_code_sentinel_without_llm(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_refined(store, monkeypatch, coding_entries())  # transcript never written to disk

    async def linker(prompt: str) -> CodeEvidence:
        raise AssertionError("the expired path must not call the LLM")

    monkeypatch.setattr("cc_pushback.enrich.structured_judge", lambda *_, **__: linker)
    report = await enrich(store)
    assert (report.enriched, report.code, report.no_code, report.failed, report.pending) == (1, 0, 1, 0, 0)

    cur = await store.store.conn.execute("SELECT pair_index, evidence_kind, note, source FROM pair_evidence")
    rows = [(row["pair_index"], row["evidence_kind"], row["note"], row["source"]) async for row in cur]
    assert rows == [(-1, "no_code", "transcript expired before enrichment", None)]
    row = (await store.pairs())[0]
    assert (row["evidence_kind"], row["evidence_note"]) == ("no_code", "transcript expired before enrichment")


@pytest.mark.integration
async def test_editless_window_persists_no_code_for_free(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [
        user_text("check the deploy status"),
        assistant_text("the deploy is green"),
        user_text("no, look closer, this is wrong"),
        assistant_text("checking again"),
    ]
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries, pushback="no, look closer, this is wrong")

    async def linker(prompt: str) -> CodeEvidence:
        raise AssertionError("the editless path must not call the LLM")

    monkeypatch.setattr("cc_pushback.enrich.structured_judge", lambda *_, **__: linker)
    report = await enrich(store)
    assert (report.enriched, report.no_code, report.pending) == (1, 1, 0)
    cur = await store.store.conn.execute("SELECT pair_index, note FROM pair_evidence")
    assert [(row["pair_index"], row["note"]) async for row in cur] == [(-1, "no edits in the lookback window")]


@pytest.mark.integration
async def test_git_sourced_correction_counts_and_persists(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)
    fix = git_fix(INCORRECT_NEW, CORRECT_NEW, commit="abc1234def")

    def fake_harvest(
        activity: SessionActivity, anchor: EventRef, *, repo: Path | None = None
    ) -> tuple[CandidatePair, ...]:
        incorrect = next(edit for edit in activity.edits if edit.hunks[0].old == INCORRECT_OLD)
        return (CandidatePair(incorrect=incorrect, correction=fix, overlap=0.5),)

    monkeypatch.setattr("cc_pushback.enrich.harvest_pairs", fake_harvest)
    prompts: list[str] = []

    async def linker(prompt: str) -> CodeEvidence:
        prompts.append(prompt)
        return code_evidence(correct=EditSide(old=INCORRECT_NEW, new=CORRECT_NEW))

    monkeypatch.setattr("cc_pushback.enrich.structured_judge", lambda *_, **__: linker)
    report = await enrich(store)
    assert (report.code, report.git, report.pending) == (1, 1, 0)
    assert "correction (git abc1234def, overlap 0.50) [likely fix]:" in prompts[0]
    assert (await store.pairs())[0]["evidence_source"] == "git"


@pytest.mark.integration
async def test_refine_rerun_at_a_new_version_resurfaces_pairs(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    async def linker(prompt: str) -> CodeEvidence:
        return code_evidence(correct=None)

    monkeypatch.setattr("cc_pushback.enrich.structured_judge", lambda *_, **__: linker)
    assert (await enrich(store)).pending == 0
    settled = await store.unenriched(
        enrich_version=ENRICH_VERSION, enrich_model=MODEL, extractor_version=EXTRACTOR_VERSION
    )
    assert settled == []

    monkeypatch.setattr("cc_pushback.refine.PROMPT_VERSION", REFINE_VERSION + 1)
    await refine(store)
    resurfaced = await store.unenriched(
        enrich_version=ENRICH_VERSION, enrich_model=MODEL, extractor_version=EXTRACTOR_VERSION
    )
    assert [int(str(row["refine_version"])) for row in resurfaced] == [REFINE_VERSION + 1]

    report = await enrich(store)
    assert (report.enriched, report.pending) == (1, 0)
    row = (await store.pairs())[0]
    assert (row["prompt_version"], row["evidence_kind"], row["evidence_source"]) == (REFINE_VERSION + 1, "code", None)
