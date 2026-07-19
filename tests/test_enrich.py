from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from cc_transcript.corrections import CorrectionLog
from cc_transcript.ids import EventUuid, SessionId, tool_digest

from cc_steer.detectors import detect
from cc_steer.enrich import EnrichError, enrich, run_enrichments
from cc_steer.refine import PROMPT_VERSION as REFINE_VERSION
from cc_steer.refine import RefinedPair, Refinement, refine
from cc_steer.triage import Verdict, triage
from tests.builders import SESSION, assistant_text, assistant_tool_use, parse, tool_result, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"
STEERING = "no, this skips validation, this is wrong"

INCORRECT_OLD = "def parse(blob):\n    return blob"
INCORRECT_NEW = "def parse(blob):\n    return json.loads(blob)"
CORRECT_NEW = "def parse(blob):\n    data = json.loads(blob)\n    validate(data)\n    return data"

PROMPT_ROW = {
    "action": "rewrote the parser without validation",
    "direction": "the change skips validation",
    "direction_verbatim": STEERING,
}


@pytest.fixture(autouse=True)
def deterministic_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe no LLM backend, so the extractor picks the best-overlap candidate."""
    monkeypatch.setattr("cc_steer.enrich.usable_backend", lambda: None)


def coding_entries() -> list[dict[str, Any]]:
    return [
        user_text("tighten the parser"),
        assistant_text("editing the parser"),
        assistant_tool_use(
            "t1", "Edit", {"file_path": "/repo/app.py", "old_string": INCORRECT_OLD, "new_string": INCORRECT_NEW}
        ),
        tool_result("t1", "ok"),
        user_text(STEERING),
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
    steering: str = STEERING,
    origin: str = FILE,
) -> None:
    await store.record_file_scan(origin, 1.0, detect(parse(entries)))

    async def judge(prompt: str) -> Verdict:
        accepted = f"USER MESSAGE TO CLASSIFY ===\n{steering}" in prompt
        return Verdict(
            category="incorrect_change" if accepted else "status_update",
            what_claude_did="edited the parser",
            confidence=0.9,
            rationale="r",
        )

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: judge)
    await triage(store)

    async def refiner(prompt: str) -> Refinement:
        return Refinement(
            pairs=[
                RefinedPair(
                    action=PROMPT_ROW["action"],
                    direction_verbatim=steering,
                    direction=PROMPT_ROW["direction"],
                )
            ]
        )

    monkeypatch.setattr("cc_steer.refine.structured_judge", lambda *_, **__: refiner)
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
    assert (report.enriched, report.corrections, report.skipped, report.pending) == (1, 1, 0, 0)

    edit_input = {"file_path": "/repo/app.py", "old_string": INCORRECT_OLD, "new_string": INCORRECT_NEW}
    (row,) = await (await CorrectionLog.open()).by_digest(SessionId(SESSION), incorrect_digest=tool_digest("Edit", edit_input))
    assert (row.source, row.incorrect_file) == ("cc-steer", "/repo/app.py")
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
    assert len(await (await CorrectionLog.open()).for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_enrich_stamps_the_repo_into_the_correction_detail(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    await enrich(store)
    (row,) = await (await CorrectionLog.open()).for_session(SessionId(SESSION))
    assert row.detail == {"repo": "/repo"}  # the anchor event's cwd
    assert await (await CorrectionLog.open()).for_repo("/repo") == (row,)


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
    assert len(await (await CorrectionLog.open()).for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_expired_transcript_skips_without_a_correction(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_refined(store, monkeypatch, coding_entries())  # transcript never written to disk

    report = await enrich(store)
    # No anchor row lands, so the pair cannot settle; it is skipped, not corrected.
    assert (report.enriched, report.corrections, report.skipped) == (1, 0, 1)
    assert report.pending == 1
    assert await (await CorrectionLog.open()).for_session(SessionId(SESSION)) == ()


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
    await seed_refined(store, monkeypatch, entries, steering="no, look closer, this is wrong")

    report = await enrich(store)
    assert (report.corrections, report.skipped, report.pending) == (0, 1, 1)
    assert await (await CorrectionLog.open()).for_session(SessionId(SESSION)) == ()


@pytest.mark.unit
async def test_run_enrichments_isolates_one_failed_pair_among_many(monkeypatch: pytest.MonkeyPatch) -> None:
    # Serialized (concurrency=1) so the fixed outcome order is deterministic: a
    # correction, an isolated failure, then two clean skips.
    outcomes = iter([True, EnrichError("provider hiccup"), False, False])

    async def resolve(*_: object, **__: object) -> bool:
        match next(outcomes):
            case EnrichError() as err:
                raise err
            case bool() as landed:
                return landed

    monkeypatch.setattr("cc_steer.enrich.resolve_pair", resolve)
    rows: list[dict[str, object]] = [{} for _ in range(4)]
    corrections, skipped, failed = await run_enrichments(
        rows, tier="medium", concurrency=1, log=await CorrectionLog.open(), backend=None, max_consecutive_failures=3
    )
    # One bad pair is isolated; the rest of the pass still completes.
    assert (corrections, skipped, failed) == (1, 2, 1)


@pytest.mark.unit
async def test_run_enrichments_aborts_after_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    async def resolve(*_: object, **__: object) -> bool:
        nonlocal attempts
        attempts += 1
        raise EnrichError("backend down")

    monkeypatch.setattr("cc_steer.enrich.resolve_pair", resolve)
    rows: list[dict[str, object]] = [{} for _ in range(10)]
    with pytest.raises(EnrichError, match="3 consecutive"):
        await run_enrichments(
            rows, tier="medium", concurrency=1, log=await CorrectionLog.open(), backend=None, max_consecutive_failures=3
        )
    assert attempts == 3  # aborts on the third; the remaining seven rows are never attempted


@pytest.mark.integration
async def test_enrich_isolates_a_failed_pair_and_completes(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_refined(store, monkeypatch, coding_entries())

    async def failing(*_: object, **__: object) -> bool:
        raise EnrichError("provider hiccup")

    monkeypatch.setattr("cc_steer.enrich.resolve_pair", failing)
    report = await enrich(store)
    # The lone pair fails, is isolated (never raised), and stays pending for the next pass.
    assert (report.enriched, report.corrections, report.skipped, report.failed, report.pending) == (0, 0, 0, 1, 1)
    assert await (await CorrectionLog.open()).for_session(SessionId(SESSION)) == ()


@pytest.mark.integration
async def test_a_programming_error_still_aborts_the_pass(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed_refined(store, monkeypatch, coding_entries())

    async def broken(*_: object, **__: object) -> bool:
        raise RuntimeError("corrupt transcript")  # not an EnrichError, so it is never isolated

    monkeypatch.setattr("cc_steer.enrich.resolve_pair", broken)
    with pytest.raises(ExceptionGroup):
        await enrich(store)


@pytest.mark.integration
async def test_pairs_sharing_one_anchor_settle_together(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await store.record_file_scan(FILE, 1.0, detect(parse(entries)))

    async def judge(prompt: str) -> Verdict:
        accepted = f"USER MESSAGE TO CLASSIFY ===\n{STEERING}" in prompt
        return Verdict(
            category="incorrect_change" if accepted else "status_update",
            what_claude_did="edited the parser",
            confidence=0.9,
            rationale="r",
        )

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: judge)
    await triage(store)

    async def refiner(prompt: str) -> Refinement:
        return Refinement(
            pairs=[
                RefinedPair(action="a0", direction_verbatim=STEERING, direction="skips validation"),
                RefinedPair(action="a1", direction_verbatim=STEERING, direction="loses the schema"),
            ]
        )

    monkeypatch.setattr("cc_steer.refine.structured_judge", lambda *_, **__: refiner)
    assert (await refine(store)).pairs == 2

    assert len(await store.unenriched(await CorrectionLog.open())) == 2  # two pairs, one shared anchor
    # Serialize the pass so the per-anchor idempotency is observed in order: the
    # first pair writes the anchor's row, the second sees it and is a clean no-op.
    report = await enrich(store, concurrency=1)
    assert (report.enriched, report.corrections, report.skipped, report.pending) == (2, 1, 1, 0)
    assert len(await (await CorrectionLog.open()).for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_a_grounded_anchor_stays_settled_across_a_refine_rerun(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    assert (await enrich(store)).pending == 0
    assert await store.unenriched(await CorrectionLog.open()) == []

    # A fresh refine generation reuses the same steering anchor, which already
    # carries a correction — so the pair stays settled and enrich never re-writes.
    monkeypatch.setattr("cc_steer.refine.PROMPT_VERSION", REFINE_VERSION + 1)
    await refine(store)
    assert await store.unenriched(await CorrectionLog.open()) == []

    report = await enrich(store)
    assert (report.corrections, report.skipped, report.pending) == (0, 0, 0)
    assert len(await (await CorrectionLog.open()).for_session(SessionId(SESSION))) == 1


@pytest.mark.integration
async def test_unenriched_skips_pairs_whose_anchor_already_has_a_ledger_row(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = coding_entries()
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    await seed_refined(store, monkeypatch, entries)

    log = await CorrectionLog.open()
    [pending] = await store.unenriched(log)
    assert (pending["session_id"], pending["event_uuid"]) == (SESSION, EventUuid(str(pending["event_uuid"])))
    await enrich(store)
    assert await store.unenriched(await CorrectionLog.open()) == []
