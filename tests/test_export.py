from __future__ import annotations

import json
from typing import TYPE_CHECKING

import datasets
import huggingface_hub
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.corrections import Correction, CorrectionLog
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import DedupKey

from cc_steer.export import export, split_of
from cc_steer.refine import RefinedPair, Refinement
from cc_steer.triage import AUDIT_VERSION, JUDGE, PROMPT_VERSION, Verdict

if TYPE_CHECKING:
    from pathlib import Path

    from cc_transcript.context import Role

    from cc_steer.store import FeedbackStore

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

TRAIN_SESSION = "sess-0"
TEST_SESSION = "sess-14"
ORIGIN = "/h-Code-proj/s.jsonl"
SIGNAL = {"confidence": 0.5, "reasons": ["user_message"], "durable": True}


def turn(role: Role, preview: str) -> TurnRef:
    return TurnRef(role=role, refs=(), preview=preview, tool_digests=())


def window(
    session: str,
    uuid: str,
    *,
    before: tuple[TurnRef, ...] = (),
    trigger: TurnRef | None = None,
    after: tuple[TurnRef, ...] = (),
) -> str:
    return ContextWindow(
        anchor=EventRef(SessionId(session), EventUuid(uuid)),
        before=before,
        trigger=trigger,
        after=after,
        fidelity="full",
        preview_chars=200,
    ).to_json()


def verdict(category: str, *, what: str) -> Verdict:
    return Verdict.model_validate({"category": category, "what_claude_did": what, "confidence": 0.9, "rationale": "r"})


def correction(uuid: str, *, ts_ms: int, source: str = "cc-steer", grounded: bool = True) -> Correction:
    return Correction(
        ts_ms=ts_ms,
        session_id=SessionId(TRAIN_SESSION),
        source=source,
        anchor_uuid=EventUuid(uuid),
        incorrect_digest=f"d{ts_ms}",
        incorrect_file="/repo/a.py",
        incorrect_old="bad",
        incorrect_new="worse",
        correction_origin="git" if grounded else None,
        correction_file="/repo/a.py" if grounded else None,
        correction_old="worse" if grounded else None,
        correction_new="good" if grounded else None,
        correction_commit="abc123" if grounded else None,
        overlap=0.9 if grounded else 0.0,
    )


async def seed(store: FeedbackStore) -> None:
    events = [
        (
            "k1",
            "transcript_message",
            TRAIN_SESSION,
            "u1",
            "no dont vendor it",
            json.dumps({"signal": SIGNAL}),
            window(
                TRAIN_SESSION,
                "u1",
                before=(turn("user", "please add the feature"), turn("assistant", "I vendored the lib")),
                trigger=turn("user", "no dont vendor it"),
                after=(turn("assistant", "removed the vendored copy"),),
            ),
        ),
        (
            "k2",
            "transcript_message",
            TRAIN_SESSION,
            "u2",
            "thanks, looks good",
            json.dumps({"signal": SIGNAL}),
            window(TRAIN_SESSION, "u2", before=(turn("assistant", "shipped the fix"),)),
        ),
        (
            "k3",
            "review_comment",
            TEST_SESSION,
            "u3",
            "no comments",
            json.dumps(
                {"format": "superset-inline", "file": "src/x.py", "line_start": 5, "line_end": None}
                | {"signal": SIGNAL}
            ),
            window(TEST_SESSION, "u3", before=(turn("user", "review the diff"),)),
        ),
        (
            "k4",
            "plan_review",
            TRAIN_SESSION,
            "u1",
            "no dont vendor it",
            json.dumps({"signal": SIGNAL}),
            window(TRAIN_SESSION, "u1"),
        ),
    ]
    for i, (key, kind, session, uuid, text, payload, ctx) in enumerate(events):
        await store.store.conn.execute(
            "INSERT INTO feedback_events (dedup_key, source_kind, session_id, event_uuid, "
            "occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                kind,
                session,
                uuid,
                f"2026-01-0{i + 1}T00:00:00",
                text,
                payload,
                ctx,
                "2.0.1",
                "2026-01-01T00:00:00",
                ORIGIN,
            ),
        )
    await store.record_verdict(
        DedupKey("k1"),
        verdict("wrong_approach", what="vendored the dependency"),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        model="sonnet",
        fidelity="full",
    )
    await store.record_verdict(
        DedupKey("k1"),
        verdict("wrong_approach", what="vendored the dependency"),
        role="auditor",
        prompt_version=AUDIT_VERSION,
        model="opus",
        fidelity="full",
    )
    await store.record_verdict(
        DedupKey("k2"),
        verdict("status_update", what="shipped the fix"),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        model="sonnet",
        fidelity="full",
    )
    await store.record_verdict(
        DedupKey("k3"),
        verdict("style_violation", what="added comments to the diff"),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        model="sonnet",
        fidelity="summary",
    )
    await store.record_verdict(
        DedupKey("k4"),
        verdict("operational_directive", what="proposed the plan again"),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        model="sonnet",
        fidelity="full",
    )
    stale = Refinement(
        pairs=[
            RefinedPair(action="stale a", direction_verbatim="stale", direction="stale"),
            RefinedPair(action="stale b", direction_verbatim="stale", direction="stale"),
        ]
    )
    await store.record_refinement(DedupKey("k1"), stale, prompt_version=1, model="sonnet")
    latest = Refinement(
        pairs=[
            RefinedPair(
                action="vendored the lib",
                direction_verbatim="no dont vendor it",
                direction="do not vendor dependencies",
            ),
        ]
    )
    await store.record_refinement(DedupKey("k1"), latest, prompt_version=2, model="sonnet")
    log = CorrectionLog.open()
    log.append(correction("u1", ts_ms=1))
    log.append(correction("u1", ts_ms=2, grounded=False))
    log.append(correction("u1", ts_ms=3, source="captain-hook"))


def rows(out: Path, config: str, split: str) -> list[dict[str, object]]:
    import pyarrow.parquet

    return pyarrow.parquet.read_table(out / config / f"{split}.parquet").to_pylist()


@pytest.fixture
async def out(store: FeedbackStore, tmp_path: Path) -> Path:
    await seed(store)
    report = await export(store, out=tmp_path / "dataset")
    assert report.counts == {
        "traces": {"train": 3, "test": 1},
        "sft": {"train": 1, "test": 1},
        "dpo": {"train": 1, "test": 0},
        "kto": {"train": 3, "test": 1},
    }
    assert report.pushed is False
    return report.out


@pytest.mark.unit
def test_split_of_is_deterministic_on_the_session_hash() -> None:
    assert split_of(TEST_SESSION) == split_of(TEST_SESSION) == "test"
    assert split_of(TRAIN_SESSION) == split_of(TRAIN_SESSION) == "train"


async def test_traces_grounded_row_carries_full_lineage(out: Path) -> None:
    trace = {row["id"]: row for row in rows(out, "traces", "train")}["k1"]
    assert trace == {
        "id": "k1",
        "session_id": TRAIN_SESSION,
        "event_uuid": "u1",
        "project": "proj",
        "occurred_at": "2026-01-01T00:00:00",
        "cc_version": "2.0.1",
        "source_kind": "transcript_message",
        "context": [
            {"role": "user", "content": "please add the feature"},
            {"role": "assistant", "content": "I vendored the lib"},
        ],
        "agent_action": "I vendored the lib",
        "what_claude_did": "vendored the dependency",
        "user_message": "no dont vendor it",
        "aftermath": [
            {"role": "user", "content": "no dont vendor it"},
            {"role": "assistant", "content": "removed the vendored copy"},
        ],
        "is_steering": True,
        "category": "wrong_approach",
        "confidence": 0.9,
        "judge_rationale": "r",
        "judge_model": "sonnet",
        "fidelity": "full",
        "auditor_category": "wrong_approach",
        "pairs": [
            {
                "pair_index": 0,
                "action": "vendored the lib",
                "direction_verbatim": "no dont vendor it",
                "direction": "do not vendor dependencies",
            }
        ],
        "evidence": [
            {
                "digest": "d1",
                "file": "/repo/a.py",
                "faulted_old": "bad",
                "faulted_new": "worse",
                "origin": "git",
                "correction_file": "/repo/a.py",
                "correcting_old": "worse",
                "correcting_new": "good",
                "commit": "abc123",
                "overlap": 0.9,
            },
            {
                "digest": "d2",
                "file": "/repo/a.py",
                "faulted_old": "bad",
                "faulted_new": "worse",
                "origin": None,
                "correction_file": None,
                "correcting_old": None,
                "correcting_new": None,
                "commit": None,
                "overlap": 0.0,
            },
        ],
        "split": "train",
        "meta": trace["meta"],
    }
    assert json.loads(str(trace["meta"])) == {
        "signal": SIGNAL,
        "prompt_version": PROMPT_VERSION,
        "audit_version": AUDIT_VERSION,
        "origin_path": ORIGIN,
    }


async def test_traces_noise_row_has_empty_pairs_and_evidence(out: Path) -> None:
    trace = {row["id"]: row for row in rows(out, "traces", "train")}["k2"]
    assert trace["is_steering"] is False and trace["category"] == "status_update"
    assert trace["pairs"] == [] and trace["evidence"] == []
    assert trace["auditor_category"] is None


async def test_traces_review_comment_meta_carries_file_line_and_format(out: Path) -> None:
    trace = rows(out, "traces", "test")[0]
    assert trace["id"] == "k3" and trace["fidelity"] == "summary"
    assert json.loads(str(trace["meta"])) == {
        "signal": SIGNAL,
        "file": "src/x.py",
        "line_start": 5,
        "line_end": None,
        "format": "superset-inline",
        "prompt_version": PROMPT_VERSION,
        "audit_version": AUDIT_VERSION,
        "origin_path": ORIGIN,
    }


async def test_sft_contract_speaks_the_verbatim_steering(out: Path) -> None:
    row = rows(out, "sft", "train")[0]
    assert set(row) == {"prompt", "completion", "id", "category"}
    assert row["id"] == "k1" and row["category"] == "wrong_approach"
    assert row["prompt"] == [
        {"role": "user", "content": "please add the feature"},
        {"role": "assistant", "content": "I vendored the lib"},
        {"role": "assistant", "content": "I vendored the lib"},
    ]
    assert row["completion"] == [{"role": "assistant", "content": "no dont vendor it"}]


async def test_sft_falls_back_to_what_claude_did_without_an_agent_action(out: Path) -> None:
    row = rows(out, "sft", "test")[0]
    assert row["id"] == "k3"
    assert row["prompt"] == [
        {"role": "user", "content": "review the diff"},
        {"role": "assistant", "content": "added comments to the diff"},
    ]
    assert row["completion"] == [{"role": "assistant", "content": "no comments"}]


async def test_dpo_contract_renders_both_sides_of_grounded_evidence(out: Path) -> None:
    dpo = rows(out, "dpo", "train")
    assert len(dpo) == 1  # only the evidence entry with both sides; the ungrounded one drops
    row = dpo[0]
    assert set(row) == {"prompt", "chosen", "rejected", "id", "category"}
    assert row["id"] == "k1" and row["category"] == "wrong_approach"
    assert row["prompt"] == [
        {"role": "user", "content": "please add the feature"},
        {"role": "assistant", "content": "I vendored the lib"},
    ]
    assert row["chosen"] == [{"role": "assistant", "content": "/repo/a.py\n```old\nworse\n```\n```new\ngood\n```"}]
    assert row["rejected"] == [{"role": "assistant", "content": "/repo/a.py\n```old\nbad\n```\n```new\nworse\n```"}]


async def test_dpo_dedupes_ledger_evidence_shared_by_dual_detected_events(out: Path) -> None:
    traces = {row["id"]: row for row in rows(out, "traces", "train")}
    assert traces["k4"]["event_uuid"] == traces["k1"]["event_uuid"]
    assert traces["k4"]["evidence"] == traces["k1"]["evidence"]
    assert [row["id"] for row in rows(out, "dpo", "train")] == ["k1"]


async def test_kto_contract_labels_no_steering_as_desirable(out: Path) -> None:
    kto = {row["id"]: row for row in rows(out, "kto", "train")}
    assert set(kto["k1"]) == {"prompt", "completion", "label", "id", "category"}
    assert kto["k1"]["label"] is False  # the user pushed back: undesirable
    assert kto["k1"]["completion"] == [{"role": "assistant", "content": "I vendored the lib"}]
    assert kto["k2"]["label"] is True
    assert kto["k2"]["completion"] == [{"role": "assistant", "content": "shipped the fix"}]
    fallback = rows(out, "kto", "test")[0]
    assert fallback["completion"] == [{"role": "assistant", "content": "added comments to the diff"}]


async def test_derived_configs_inherit_the_parent_split(out: Path) -> None:
    assert {row["id"] for row in rows(out, "traces", "test")} == {"k3"}
    assert {row["id"] for row in rows(out, "sft", "test")} == {"k3"}
    assert {row["id"] for row in rows(out, "kto", "test")} == {"k3"}
    assert rows(out, "dpo", "test") == []
    for config in ("sft", "kto", "dpo"):
        assert "k3" not in {row["id"] for row in rows(out, config, "train")}


async def test_dataset_card_documents_configs_categories_and_splits(out: Path) -> None:
    card = (out / "README.md").read_text()
    assert "- config_name: traces\n  default: true" in card
    for config in ("sft", "dpo", "kto"):
        assert f"- config_name: {config}" in card
    assert "path: traces/train*.parquet" in card
    assert "size_categories:\n- 1K<n<10K" in card
    for category in ("wrong_approach", "style_violation", "operational_directive", "other"):
        assert f"`{category}`" in card
    assert "int(sha256(session_id), 16) % 10 == 0" in card
    assert "3 train / 1 test" in card
    assert f"judge v{PROMPT_VERSION}" in card and f"auditor v{AUDIT_VERSION}" in card
    assert "2 steering vs 2 noise (50% steering)" in card


async def test_export_survives_a_corpus_with_zero_judged_events(store: FeedbackStore, tmp_path: Path) -> None:
    report = await export(store, out=tmp_path / "dataset")
    assert report.counts == {config: {"train": 0, "test": 0} for config in ("traces", "sft", "dpo", "kto")}
    assert report.pushed is False
    for config in ("traces", "sft", "dpo", "kto"):
        assert rows(report.out, config, "train") == [] and rows(report.out, config, "test") == []
    card = (report.out / "README.md").read_text()
    assert "0 train / 0 test" in card
    assert "0 steering vs 0 noise at" in card


async def test_export_push_uploads_every_config_and_the_card(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pushes: list[dict[str, object]] = []
    uploads: list[dict[str, object]] = []
    monkeypatch.setattr(
        datasets.DatasetDict,
        "push_to_hub",
        lambda self, repo_id, **kwargs: pushes.append({"repo_id": repo_id} | kwargs),
    )
    monkeypatch.setattr(huggingface_hub.HfApi, "upload_file", lambda self, **kwargs: uploads.append(kwargs))
    await seed(store)
    report = await export(store, out=tmp_path / "dataset", push_to="u/r")
    assert report.pushed is True
    assert len(pushes) == 4
    assert {push["config_name"] for push in pushes} == {"traces", "sft", "dpo", "kto"}
    assert all(push["repo_id"] == "u/r" and push["private"] is True for push in pushes)
    assert uploads == [
        {
            "path_or_fileobj": report.out / "README.md",
            "path_in_repo": "README.md",
            "repo_id": "u/r",
            "repo_type": "dataset",
        }
    ]


async def test_export_push_failure_propagates_after_local_write(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(*_: object, **__: object) -> None:
        raise RuntimeError("hub is down")

    monkeypatch.setattr(datasets.DatasetDict, "push_to_hub", down)
    await seed(store)
    dataset_dir = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="hub is down"):
        await export(store, out=dataset_dir, push_to="u/r")
    for config in ("traces", "sft", "dpo", "kto"):
        assert {path.name for path in (dataset_dir / config).glob("*.parquet")} == {"train.parquet", "test.parquet"}
    assert (dataset_dir / "README.md").is_file()
