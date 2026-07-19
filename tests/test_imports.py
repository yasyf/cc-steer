from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from cc_transcript.context import ContextWindow
from cc_transcript.mining import dedup_key
from pydantic import ValidationError

from cc_steer.imports import (
    IMPORT_SOURCE_KIND,
    SCHEMA_ID,
    SESSION_PREFIX,
    ImportBatch,
    import_batch,
)
from cc_steer.rendering import agent_action_of, context_turns, messages
from cc_steer.store import FeedbackStore
from cc_steer.triage import JUDGE, PROMPT_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio


def item_dict(
    *,
    external_id: str = "dec-1",
    label: str = "rejected",
    repo: str | None = "yasyf/cc-steer",
    context: str = "I edited store.py directly to add the column.",
    verbatim: str = "don't touch store.py — add a new module instead",
) -> dict[str, Any]:
    return {
        "external_id": external_id,
        "occurred_at": "2026-07-19T10:00:00+00:00",
        "repo": repo,
        "context": context,
        "verbatim": verbatim,
        "label": label,
    }


def batch_dict(*, source: str = "cc-factory", items: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"schema": SCHEMA_ID, "source": source, "items": list(items) if items is not None else [item_dict()]}


async def fetchall(store: FeedbackStore, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, object]]:
    return await store.sql(sql, params)


async def table_columns(store: FeedbackStore) -> set[str]:
    return {str(row["name"]) for row in await fetchall(store, "PRAGMA table_info(feedback_events)")}


@pytest.fixture
async def store(tmp_path: Path) -> FeedbackStore:
    async with await FeedbackStore.open(tmp_path / "feedback.db") as opened:
        yield opened


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"schema": "cc-steer/import@2"}, id="wrong-schema-tag"),
        pytest.param({"source": 5}, id="non-string-source"),
        pytest.param({"bogus": 1}, id="extra-top-level-key"),
    ],
)
def test_schema_rejects_malformed_batch(mutation: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ImportBatch.model_validate(batch_dict() | mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"label": "maybe"}, id="unknown-label"),
        pytest.param({"occurred_at": "not-a-date"}, id="bad-datetime"),
        pytest.param({"surprise": True}, id="extra-item-key"),
    ],
)
def test_schema_rejects_malformed_item(mutation: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ImportBatch.model_validate(batch_dict(items=[item_dict() | mutation]))


def test_schema_accepts_null_repo() -> None:
    parsed = ImportBatch.model_validate(batch_dict(items=[item_dict(repo=None)]))
    assert parsed.items[0].repo is None
    assert parsed.schema_ == SCHEMA_ID


async def test_import_inserts_candidate_rows(store: FeedbackStore) -> None:
    result = await import_batch(
        ImportBatch.model_validate(batch_dict(items=[item_dict(external_id="dec-1", label="accepted")])),
        db=store,
    )

    assert not result.dry_run
    assert len(result.new) == 1
    assert result.duplicates == ()

    rows = await fetchall(store, "SELECT * FROM feedback_events")
    assert len(rows) == 1
    row = rows[0]
    assert row["dedup_key"] == dedup_key("cc-factory", "dec-1")
    assert row["source_kind"] == IMPORT_SOURCE_KIND
    assert row["text"] == "don't touch store.py — add a new module instead"
    assert row["import_source"] == "cc-factory"
    assert row["import_batch"] == result.batch
    assert str(row["session_id"]).startswith(SESSION_PREFIX)
    assert row["quarantined_reason"] is None

    payload = json.loads(str(row["payload_json"]))
    assert payload["label"] == "accepted"
    assert payload["provenance"] == "imported"
    assert payload["import_source"] == "cc-factory"
    assert payload["external_id"] == "dec-1"


async def test_imported_row_carries_context_window(store: FeedbackStore) -> None:
    await import_batch(
        ImportBatch.model_validate(
            batch_dict(items=[item_dict(context="ran the whole suite", verbatim="run only the failing test")])
        ),
        db=store,
    )
    row = (await fetchall(store, "SELECT context_json FROM feedback_events"))[0]
    window = ContextWindow.from_json(str(row["context_json"]))

    assert window.fidelity == "summary"
    assert agent_action_of(window) == "ran the whole suite"
    assert window.trigger is not None
    assert window.trigger.role == "user"
    assert window.trigger.preview == "run only the failing test"
    # The steer is the label and must stay out of the model-visible turns.
    assert messages(context_turns(window)) == [{"role": "assistant", "content": "ran the whole suite"}]


async def test_imported_row_is_selected_for_triage(store: FeedbackStore) -> None:
    await import_batch(ImportBatch.model_validate(batch_dict()), db=store)
    unjudged = await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION)
    assert [row["dedup_key"] for row in unjudged] == [dedup_key("cc-factory", "dec-1")]


async def test_imported_rows_survive_context_rebuild(store: FeedbackStore, tmp_path: Path) -> None:
    from cc_steer.context_rebuild import rebuild_contexts

    await import_batch(ImportBatch.model_validate(batch_dict()), db=store)
    empty_root = tmp_path / "no-transcripts"
    empty_root.mkdir()
    report = await rebuild_contexts(store, (empty_root,))

    key = dedup_key("cc-factory", "dec-1")
    row = (await fetchall(store, "SELECT quarantined_reason FROM feedback_events WHERE dedup_key = ?", (key,)))[0]
    assert row["quarantined_reason"] is None
    assert (report.found, report.quarantined) == (0, 0)
    unjudged = await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION)
    assert [row["dedup_key"] for row in unjudged] == [key]


async def test_import_writes_no_verdicts_or_pairs(store: FeedbackStore) -> None:
    await import_batch(ImportBatch.model_validate(batch_dict()), db=store)
    assert (await fetchall(store, "SELECT COUNT(*) AS n FROM triage"))[0]["n"] == 0
    assert (await fetchall(store, "SELECT COUNT(*) AS n FROM refinement"))[0]["n"] == 0


async def test_reimport_same_batch_is_noop(store: FeedbackStore) -> None:
    batch = ImportBatch.model_validate(
        batch_dict(items=[item_dict(external_id="dec-1"), item_dict(external_id="dec-2", verbatim="be terse")])
    )
    first = await import_batch(batch, db=store)
    assert len(first.new) == 2

    second = await import_batch(batch, db=store)
    assert second.new == ()
    assert len(second.duplicates) == 2
    assert first.batch == second.batch
    assert (await fetchall(store, "SELECT COUNT(*) AS n FROM feedback_events"))[0]["n"] == 2


async def test_intra_batch_duplicate_external_id_lands_once(store: FeedbackStore) -> None:
    result = await import_batch(
        ImportBatch.model_validate(
            batch_dict(items=[item_dict(external_id="dup"), item_dict(external_id="dup", verbatim="different steer")])
        ),
        db=store,
    )
    assert [outcome.status for outcome in result.outcomes] == ["new", "duplicate"]
    assert (await fetchall(store, "SELECT COUNT(*) AS n FROM feedback_events"))[0]["n"] == 1


async def test_same_external_id_different_source_are_distinct(store: FeedbackStore) -> None:
    await import_batch(ImportBatch.model_validate(batch_dict(source="cc-factory")), db=store)
    await import_batch(ImportBatch.model_validate(batch_dict(source="other-tool")), db=store)
    rows = await fetchall(store, "SELECT dedup_key, import_source FROM feedback_events ORDER BY import_source")
    assert [row["import_source"] for row in rows] == ["cc-factory", "other-tool"]
    assert rows[0]["dedup_key"] != rows[1]["dedup_key"]


async def test_dry_run_writes_nothing(store: FeedbackStore) -> None:
    result = await import_batch(ImportBatch.model_validate(batch_dict()), db=store, dry_run=True)

    assert result.dry_run
    assert len(result.new) == 1
    assert "import_source" not in await table_columns(store)
    assert (await fetchall(store, "SELECT COUNT(*) AS n FROM feedback_events"))[0]["n"] == 0

    landed = await import_batch(ImportBatch.model_validate(batch_dict()), db=store)
    assert len(landed.new) == 1
    assert (await fetchall(store, "SELECT COUNT(*) AS n FROM feedback_events"))[0]["n"] == 1


async def test_import_adds_provenance_columns(store: FeedbackStore) -> None:
    assert "import_source" not in await table_columns(store)
    await import_batch(ImportBatch.model_validate(batch_dict()), db=store)
    columns = await table_columns(store)
    assert {"import_source", "import_batch"} <= columns


async def test_import_from_path(store: FeedbackStore, tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(batch_dict(items=[item_dict(external_id="from-disk")])))
    result = await import_batch(path, db=store)
    assert len(result.new) == 1
    row = (await fetchall(store, "SELECT dedup_key FROM feedback_events"))[0]
    assert row["dedup_key"] == dedup_key("cc-factory", "from-disk")


async def test_summary_line_reflects_disposition(store: FeedbackStore) -> None:
    batch = ImportBatch.model_validate(batch_dict(items=[item_dict(external_id="a"), item_dict(external_id="b")]))
    await import_batch(batch, db=store)
    reimport = await import_batch(batch, db=store)
    assert reimport.summary_line() == (
        f"imported 0 new, skipped 2 duplicate from cc-factory (batch {reimport.batch[:12]})"
    )
