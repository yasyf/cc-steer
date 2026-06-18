from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from cc_transcript.corrections import CorrectionLog
from cc_transcript.ids import EventUuid, SessionId, tool_digest

from cc_pushback.detectors import detect
from cc_pushback.enrich import enrich
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
PUSHBACK = "no, this skips validation, this is wrong"

INCORRECT_OLD = "def parse(blob):\n    return blob"
INCORRECT_NEW = "def parse(blob):\n    return json.loads(blob)"
CORRECT_NEW = "def parse(blob):\n    data = json.loads(blob)\n    validate(data)\n    return data"

PROMPT_ROW = {
    "action": "rewrote the parser without validation",
    "complaint": "the change skips validation",
    "complaint_verbatim": PUSHBACK,
}


@pytest.fixture(autouse=True)
def deterministic_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe no LLM backend, so the extractor picks the best-overlap candidate."""
    monkeypatch.setattr("cc_pushback.enrich.usable_backend", lambda: None)


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


async def seed_refined(
    store: FeedbackStore,
    monkeypatch: pytest.MonkeyPatch,
    entries: list[dict[str, Any]],
    *,
    pushback: str = PUSHBACK,
    origin: str = FILE,
) -> None:
    await store.record_file_scan(origin, 1.0, detect(parse(entries)))

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


@pytest.mark.integration
async def test_enrich_appends_one_correction_to_the_shared_ledger(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    report = await enrich(store)
    assert (report.enriched, report.corrections, report.skipped, report.failed, report.pending) == (1, 1, 0, 0, 0)

    edit_input = {"file_path": "/repo/app.py", "old_string": INCORRECT_OLD, "new_string": INCORRECT_NEW}
    (row,) = CorrectionLog.open().by_digest(SessionId(SESSION), incorrect_digest=tool_digest("Edit", edit_input))
    assert (row.source, row.incorrect_file) == ("cc-pushback", "/repo/app.py")
    assert (row.incorrect_old, row.incorrect_new) == (INCORRECT_OLD, INCORRECT_NEW)
    assert (row.correction_origin, row.correction_old, row.correction_new) == ("session", INCORRECT_NEW, CORRECT_NEW)
    assert row.overlap == 1.0


@pytest.mark.integration
async def test_enrich_is_idempotent_per_anchor(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    assert (await enrich(store)).corrections == 1
    again = await enrich(store)
    assert (again.enriched, again.corrections, again.pending) == (0, 0, 0)
    assert len(CorrectionLog.open().for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_enrich_stamps_the_repo_into_the_correction_detail(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    await enrich(store)
    (row,) = CorrectionLog.open().for_session(SessionId(SESSION))
    assert row.detail == {"repo": "/repo"}  # the anchor event's cwd
    assert CorrectionLog.open().for_repo("/repo") == (row,)


@pytest.mark.integration
async def test_enrich_resolves_a_transcript_by_origin_path_outside_the_discovery_root(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A mirror-like corpus: the transcript lives outside CLAUDE_PROJECTS_DIR, so
    # session discovery would fail — only the stored origin_path resolves it.
    entries = coding_entries()
    mirror = tmp_path / "mirror" / f"{SESSION}.jsonl"
    write_transcript(mirror, entries)
    await seed_refined(store, monkeypatch, entries, origin=str(mirror))

    report = await enrich(store)
    assert (report.corrections, report.pending) == (1, 0)
    assert len(CorrectionLog.open().for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_expired_transcript_skips_without_a_correction(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_refined(store, monkeypatch, coding_entries())  # transcript never written to disk

    report = await enrich(store)
    # No anchor row lands, so the pair cannot settle; it is skipped, not corrected.
    assert (report.enriched, report.corrections, report.skipped, report.failed) == (1, 0, 1, 0)
    assert report.pending == 1
    assert CorrectionLog.open().for_session(SessionId(SESSION)) == ()


@pytest.mark.integration
async def test_editless_window_skips_without_an_llm_call(
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

    report = await enrich(store)
    assert (report.corrections, report.skipped, report.pending) == (0, 1, 1)
    assert CorrectionLog.open().for_session(SessionId(SESSION)) == ()


@pytest.mark.integration
async def test_pairs_sharing_one_anchor_settle_together(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await store.record_file_scan(FILE, 1.0, detect(parse(entries)))

    async def judge(prompt: str) -> Verdict:
        accepted = f"USER MESSAGE TO CLASSIFY ===\n{PUSHBACK}" in prompt
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
                RefinedPair(action="a0", complaint_verbatim=PUSHBACK, complaint="skips validation"),
                RefinedPair(action="a1", complaint_verbatim=PUSHBACK, complaint="loses the schema"),
            ]
        )

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: refiner)
    assert (await refine(store)).pairs == 2

    assert len(await store.unenriched(CorrectionLog.open())) == 2  # two pairs, one shared anchor
    # Serialize the pass so the per-anchor idempotency is observed in order: the
    # first pair writes the anchor's row, the second sees it and is a clean no-op.
    report = await enrich(store, concurrency=1)
    assert (report.enriched, report.corrections, report.skipped, report.pending) == (2, 1, 1, 0)
    assert len(CorrectionLog.open().for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_a_grounded_anchor_stays_settled_across_a_refine_rerun(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    assert (await enrich(store)).pending == 0
    assert await store.unenriched(CorrectionLog.open()) == []

    # A fresh refine generation reuses the same pushback anchor, which already
    # carries a correction — so the pair stays settled and enrich never re-writes.
    monkeypatch.setattr("cc_pushback.refine.PROMPT_VERSION", REFINE_VERSION + 1)
    await refine(store)
    assert await store.unenriched(CorrectionLog.open()) == []

    report = await enrich(store)
    assert (report.corrections, report.skipped, report.pending) == (0, 0, 0)
    assert len(CorrectionLog.open().for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_unenriched_skips_pairs_whose_anchor_already_has_a_ledger_row(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    log = CorrectionLog.open()
    [pending] = await store.unenriched(log)
    assert (pending["session_id"], pending["event_uuid"]) == (SESSION, EventUuid(str(pending["event_uuid"])))
    await enrich(store)
    assert await store.unenriched(CorrectionLog.open()) == []
