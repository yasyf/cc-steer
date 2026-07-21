from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import datasets
import huggingface_hub
import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.corrections import Correction, CorrectionLog
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import DedupKey

import cc_steer.export as export_mod
from cc_steer.export import (
    LIVE_EMPTY_WINDOW_REASON,
    TRAJECTORY_BUDGET,
    TRAJECTORY_UNMAPPED_REASON,
    EmptyWatcherPrompt,
    ask_message_of,
    dpo_row,
    evidence_entry,
    export,
    gate_row,
    kto_row,
    live_gate_row,
    live_watcher_row,
    resolve_origin_path,
    session_trajectory,
    sft_row,
    split_of,
    trajectory_rows,
    watcher_negative,
    watcher_positive,
)
from cc_steer.refine import RefinedPair, Refinement
from cc_steer.retrain.data import HF_PUSH_NAME, hf_revision
from cc_steer.triage import AUDIT_VERSION, JUDGE, PROMPT_VERSION, Verdict
from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.live import LiveConfig, MailboxDelivery
from tests.test_delivery import make_proposal

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
            window(TRAIN_SESSION, "u1", before=(turn("assistant", "proposed the plan again"),)),
        ),
    ]
    for i, (key, kind, session, uuid, text, payload, ctx) in enumerate(events):
        await store.execute(
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
    log = await CorrectionLog.open()
    await log.append(correction("u1", ts_ms=1))
    await log.append(correction("u1", ts_ms=2, grounded=False))
    await log.append(correction("u1", ts_ms=3, source="captain-hook"))


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
        "gate": {"train": 0, "test": 0},
        "watcher": {"train": 1, "test": 1},
        "trajectory": {"train": 0, "test": 0},  # seeded sessions' transcripts are not on disk
    }
    assert report.pushed is False
    assert report.hf_revision is None
    assert not (report.out / HF_PUSH_NAME).exists()
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
    configs = ("traces", "sft", "dpo", "kto", "gate", "watcher", "trajectory")
    assert report.counts == {config: {"train": 0, "test": 0} for config in configs}
    assert report.pushed is False
    for config in configs:
        assert rows(report.out, config, "train") == [] and rows(report.out, config, "test") == []
    card = (report.out / "README.md").read_text()
    assert "0 train / 0 test" in card
    assert "0 steering vs 0 noise at" in card


async def test_export_without_push_invalidates_previous_hf_revision(store: FeedbackStore, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / HF_PUSH_NAME).write_text(json.dumps({"hf_revision": "sha-old", "repo_id": "u/r"}))
    await seed(store)

    report = await export(store, out=dataset_dir)

    assert report.hf_revision is None
    assert hf_revision(dataset_dir=dataset_dir) is None


async def test_export_push_uploads_every_config_and_the_card(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pushes: list[dict[str, object]] = []
    uploads: list[dict[str, object]] = []

    def push(self: datasets.DatasetDict, repo_id: str, **kwargs: object) -> SimpleNamespace:
        pushes.append({"repo_id": repo_id} | kwargs)
        return SimpleNamespace(oid=f"sha-{kwargs['config_name']}")

    monkeypatch.setattr(datasets.DatasetDict, "push_to_hub", push)
    monkeypatch.setattr(huggingface_hub.HfApi, "upload_file", lambda self, **kwargs: uploads.append(kwargs))
    await seed(store)
    report = await export(store, out=tmp_path / "dataset", push_to="u/r")
    assert report.pushed is True
    assert report.hf_revision == "sha-trajectory"
    assert len(pushes) == 7
    assert {push["config_name"] for push in pushes} == {
        "traces",
        "sft",
        "dpo",
        "kto",
        "gate",
        "watcher",
        "trajectory",
    }
    assert all(push["repo_id"] == "u/r" and push["private"] is True for push in pushes)
    assert uploads == [
        {
            "path_or_fileobj": report.out / "README.md",
            "path_in_repo": "README.md",
            "repo_id": "u/r",
            "repo_type": "dataset",
        }
    ]
    sidecar = json.loads((report.out / HF_PUSH_NAME).read_text())
    ts = datetime.fromisoformat(sidecar.pop("ts"))
    assert ts.tzinfo is not None and ts.utcoffset() == timedelta(0)
    assert sidecar == {"hf_revision": "sha-trajectory", "repo_id": "u/r"}


async def test_export_push_failure_propagates_after_local_write(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(*_: object, **__: object) -> None:
        raise RuntimeError("hub is down")

    monkeypatch.setattr(datasets.DatasetDict, "push_to_hub", down)
    await seed(store)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / HF_PUSH_NAME).write_text(json.dumps({"hf_revision": "sha-old", "repo_id": "u/r"}))
    with pytest.raises(RuntimeError, match="hub is down"):
        await export(store, out=dataset_dir, push_to="u/r")
    for config in ("traces", "sft", "dpo", "kto", "gate", "watcher", "trajectory"):
        assert {path.name for path in (dataset_dir / config).glob("*.parquet")} == {"train.parquet", "test.parquet"}
    assert (dataset_dir / "README.md").is_file()
    assert hf_revision(dataset_dir=dataset_dir) is None


async def test_export_adds_live_reaction_rows_to_watcher_and_gate(store: FeedbackStore, tmp_path: Path) -> None:
    shadow_db = tmp_path / "shadow.db"
    async with await ShadowDelivery.open(shadow_db) as delivery:
        await delivery.deliver(make_proposal(session_id=TRAIN_SESSION, anchor_uuid="a1", steer="run the linter"))
    async with await MailboxDelivery.open(shadow_db, config=LiveConfig.shadow()) as mailbox:
        await mailbox.record_reaction(proposal_id=1, delivery_id=None, kind="accepted", source="cli_verb")
    report = await export(store, out=tmp_path / "dataset", shadow_db=shadow_db)
    [live] = [row for row in rows(report.out, "watcher", "train") if row["source_kind"] == "live_reaction"]
    assert live["label"] is True and live["label_confidence"] == 0.9
    assert live["completion"] == [{"role": "assistant", "content": "run the linter"}]
    assert live["prompt"][0]["role"] == "user"
    [gate] = [row for row in rows(report.out, "gate", "train") if row["source_kind"] == "live_reaction"]
    assert (gate["label"], gate["label_confidence"]) == (True, 0.9)
    assert gate["text"] == "<user>\nplease do step\n\n<assistant>\ndid step"


async def test_export_quarantines_legacy_live_reaction_without_window_render(
    store: FeedbackStore, tmp_path: Path
) -> None:
    shadow_db = tmp_path / "shadow.db"
    async with await ShadowDelivery.open(shadow_db) as delivery:
        await delivery.deliver(make_proposal(session_id=TRAIN_SESSION, anchor_uuid="a1", steer="run the linter"))
        await delivery.conn.execute("UPDATE proposals SET window_render = NULL")
    async with await MailboxDelivery.open(shadow_db, config=LiveConfig.shadow()) as mailbox:
        await mailbox.record_reaction(proposal_id=1, delivery_id=None, kind="accepted", source="cli_verb")
    report = await export(store, out=tmp_path / "dataset", shadow_db=shadow_db)
    assert report.quarantined == {LIVE_EMPTY_WINDOW_REASON: 1}
    assert rows(report.out, "watcher", "train") == []
    assert rows(report.out, "gate", "train") == []


@pytest.mark.parametrize("render", [None, "", "<assistant>\n   "], ids=["null", "empty", "role-markup-only"])
def test_live_rows_reject_empty_rendered_prompts(render: object) -> None:
    reaction = {
        "kind": "accepted",
        "session_id": TRAIN_SESSION,
        "feedback_dedup_key": None,
        "steer": "run the linter",
        "proposal_id": 7,
        "window_render": render,
    }
    with pytest.raises(EmptyWatcherPrompt) as watcher:
        live_watcher_row(reaction, {})
    assert (watcher.value.view, watcher.value.dedup_key) == ("watcher", "live:7")
    with pytest.raises(EmptyWatcherPrompt) as gate:
        live_gate_row(reaction)
    assert (gate.value.view, gate.value.dedup_key) == ("gate", "live:7")


def test_live_gate_row_rejects_role_marker_render_the_watcher_tolerates() -> None:
    # <system>-shaped render: a valid user message for the watcher, empty for the gate.
    reaction = {
        "kind": "accepted",
        "session_id": TRAIN_SESSION,
        "feedback_dedup_key": None,
        "steer": "run the linter",
        "proposal_id": 7,
        "window_render": "\n\n<system>\n",
    }
    assert live_watcher_row(reaction, {}) is not None
    with pytest.raises(EmptyWatcherPrompt) as gate:
        live_gate_row(reaction)
    assert (gate.value.view, gate.value.dedup_key) == ("gate", "live:7")


def trace_for(
    *,
    source_kind: str = "question_answer",
    meta: dict[str, object] | None = None,
    context: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "id": "k-qa",
        "session_id": TRAIN_SESSION,
        "event_uuid": "u9",
        "project": "p",
        "occurred_at": "2026-01-01T00:00:00",
        "cc_version": "2.0.1",
        "source_kind": source_kind,
        "context": context if context is not None else [{"role": "assistant", "content": ""}],
        "agent_action": None,
        "what_claude_did": "asked a question",
        "user_message": "Per-scope files + ATTACH (Recommended)",
        "aftermath": [],
        "is_steering": True,
        "category": "direction",
        "confidence": 0.9,
        "judge_rationale": "r",
        "judge_model": "opus",
        "fidelity": "full",
        "auditor_category": None,
        "pairs": [{"action": "a", "direction": "use per-scope files with ATTACH", "quote": "q"}],
        "evidence": [],
        "split": "train",
        "meta": json.dumps(
            {"signal": SIGNAL}
            | (
                meta
                if meta is not None
                else {
                    "question": "How should the db attach?",
                    "header": "Storage",
                    "recommended_pick": "Per-scope files + ATTACH",
                }
            )
        ),
    }


def empty_prompt_trace() -> dict[str, object]:
    return trace_for(meta={}, context=[]) | {"what_claude_did": ""}


class TestViewPrompts:
    def test_kto_rejects_an_empty_emitted_prompt_with_row_identity(self) -> None:
        with pytest.raises(EmptyWatcherPrompt) as raised:
            kto_row(empty_prompt_trace())
        assert (raised.value.view, raised.value.dedup_key) == ("kto", "k-qa")

    def test_sft_rejects_an_empty_final_assembled_prompt_with_row_identity(self) -> None:
        with pytest.raises(EmptyWatcherPrompt) as raised:
            sft_row(empty_prompt_trace())
        assert (raised.value.view, raised.value.dedup_key) == ("sft", "k-qa")

    def test_sft_accepts_a_substantive_action_in_the_final_assembled_prompt(self) -> None:
        row = sft_row(trace_for(meta={}, context=[]))
        assert row["prompt"] == [{"role": "assistant", "content": "asked a question"}]

    def test_dpo_rejects_an_empty_emitted_prompt_with_row_identity(self) -> None:
        with pytest.raises(EmptyWatcherPrompt) as raised:
            dpo_row(empty_prompt_trace(), evidence_entry(correction("u9", ts_ms=9)))
        assert (raised.value.view, raised.value.dedup_key) == ("dpo", "k-qa")


class TestWatcherV2:
    def test_qa_positive_appends_the_payload_ask_block(self) -> None:
        row = watcher_positive(trace_for())
        assert row["source_kind"] == "question_answer"
        assert row["session_id"] == TRAIN_SESSION
        ask = row["prompt"][-1]
        assert ask["role"] == "assistant"
        assert ask["content"].startswith("[assistant asked: Storage] How should the db attach?")
        assert "(recommended: Per-scope files + ATTACH)" in ask["content"]

    def test_the_users_pick_never_reaches_the_prompt(self) -> None:
        pick = "THE SECRET PICK"
        trace = trace_for(meta={"question": "Q?", "picked_labels": [pick], "option_pick": pick})
        row = watcher_positive(trace)
        assert all(pick not in m["content"] for m in row["prompt"])

    def test_positive_rewrites_clipped_ask_previews_in_context(self) -> None:
        preview = "AskUserQuestion([{'question': 'Pick one?', 'header': 'H', 'options': [{'label': 'a…(+50ch))"
        context = [{"role": "assistant", "content": preview}]
        row = watcher_positive(trace_for(source_kind="transcript_message", meta={}, context=context))
        assert row["prompt"][0]["content"] == "[assistant asked: H] Pick one?"
        # the context already carries the ask, so nothing is appended
        assert len(row["prompt"]) == 1

    def test_qa_without_payload_question_fails_with_row_identity(self) -> None:
        assert ask_message_of(trace_for(meta={})) is None
        with pytest.raises(EmptyWatcherPrompt) as raised:
            watcher_positive(trace_for(meta={}))
        assert raised.value.dedup_key == "k-qa"
        assert raised.value.session_id == TRAIN_SESSION
        assert raised.value.source_kind == "question_answer"

    def test_negative_carries_source_kind_and_session_id(self) -> None:
        sample = {
            "sample_key": "s1",
            "kind": "random_negative",
            "offset_turns": 0,
            "category": "",
            "source_kind": "",
            "session_id": TRAIN_SESSION,
            "window_json": window(TRAIN_SESSION, "u5", before=(turn("assistant", "did a thing"),)),
        }
        row = watcher_negative(sample)
        assert row is not None
        assert (row["source_kind"], row["session_id"]) == ("", TRAIN_SESSION)
        assert row["label"] is False


def gate_input(
    sample_key: str, *, before: tuple[TurnRef, ...], trigger: TurnRef | None = None, kind: str = "positive_window"
) -> dict[str, object]:
    return {
        "sample_key": sample_key,
        "kind": kind,
        "offset_turns": 0,
        "category": "",
        "source_kind": "transcript_message",
        "session_id": TRAIN_SESSION,
        "window_json": window(TRAIN_SESSION, "u9", before=before, trigger=trigger),
    }


def test_gate_row_emits_a_substantive_window() -> None:
    row = gate_row(gate_input("s1", before=(turn("user", "add a test"), turn("assistant", "added it"))))
    assert row is not None
    assert (row["id"], row["label"]) == ("s1", True)
    assert "added it" in row["text"]


def test_gate_row_raises_on_a_window_with_no_substantive_content() -> None:
    with pytest.raises(EmptyWatcherPrompt) as raised:
        gate_row(gate_input("s2", before=(turn("assistant", ""),), trigger=turn("user", "no, do it differently")))
    assert (raised.value.view, raised.value.dedup_key, raised.value.session_id) == ("gate", "s2", TRAIN_SESSION)


def trajectory_activity() -> object:
    from cc_transcript.activity import SessionActivity

    from tests import builders

    entries = [
        builders.user_text("do the thing", uuid="uA"),
        builders.assistant_tool_use(
            "t1", "Edit", {"file_path": "/repo/a.py", "old_string": "x", "new_string": "y"}, uuid="aA"
        ),
        builders.tool_result("t1", "ok"),
        builders.user_text("no, revert that", uuid="uB"),
        builders.assistant_text("reverted the change", uuid="aB"),
        builders.user_text("looks good", uuid="uC"),
        builders.assistant_text("thanks", uuid="aC"),
    ]
    return SessionActivity.from_events(SessionId("sess-traj"), builders.parse(entries))


@pytest.mark.unit
async def test_session_trajectory_maps_anchors_and_counts_the_compaction_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    activity = trajectory_activity()

    async def fake_load(session_id: object, origin_path: object) -> object:
        return activity

    monkeypatch.setattr(export_mod, "load_activity", fake_load)
    anchors = [
        {"event_uuid": "uB", "category": "wrong_approach"},  # lands on the revert turn
        {"event_uuid": "ghost-uuid", "category": "direction"},  # compacted away — in no turn
    ]
    rows, unmapped = await session_trajectory(
        "sess-traj", anchors, origin_path="/mirror/s.jsonl", budget=TRAJECTORY_BUDGET
    )

    assert unmapped == 1  # the ghost anchor is counted, never silently dropped
    steered = [row for row in rows if row["steer_label"]]
    assert len(steered) == 1
    assert steered[0]["steer_category"] == "wrong_approach"
    assert steered[0]["steer_event_uuid"] == "uB"
    assert "revert" in steered[0]["prompt"] and "reverted the change" in steered[0]["assistant_digest"]
    edited = [row for row in rows if row["n_edits"] == 1]
    assert len(edited) == 1 and edited[0]["tool_summary"] == "Edit"
    assert {row["split"] for row in rows} == {split_of("sess-traj")}


@pytest.mark.unit
async def test_session_trajectory_yields_nothing_for_an_expired_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    from cc_transcript.discovery import TranscriptExpiredError

    async def expired(session_id: object, origin_path: object) -> object:
        raise TranscriptExpiredError(SessionId("sess-traj"))

    monkeypatch.setattr(export_mod, "load_activity", expired)
    rows, unmapped = await session_trajectory(
        "sess-traj", [{"event_uuid": "uB", "category": "wrong_approach"}], origin_path=None, budget=TRAJECTORY_BUDGET
    )
    assert rows == [] and unmapped == 0


@pytest.mark.unit
async def test_trajectory_rows_bounds_io_to_steering_sessions_and_totals_the_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cc_transcript.discovery import TranscriptExpiredError

    activity = trajectory_activity()

    async def route(session_id: object, origin_path: object) -> object:
        if str(session_id) == "sess-traj":
            return activity
        raise TranscriptExpiredError(SessionId(str(session_id)))

    monkeypatch.setattr(export_mod, "load_activity", route)
    traces = [
        {
            "is_steering": True,
            "session_id": "sess-traj",
            "event_uuid": "uB",
            "category": "wrong_approach",
            "meta": json.dumps({"origin_path": "/mirror/s.jsonl"}),
        },
        {
            "is_steering": True,
            "session_id": "sess-traj",
            "event_uuid": "ghost-uuid",
            "category": "direction",
            "meta": json.dumps({"origin_path": "/mirror/s.jsonl"}),
        },
        {  # a noise trace never triggers transcript I/O
            "is_steering": False,
            "session_id": "sess-noise",
            "event_uuid": "n1",
            "category": "status_update",
            "meta": json.dumps({"origin_path": "/mirror/other.jsonl"}),
        },
    ]
    by_split, unmapped = await trajectory_rows(traces)
    all_rows = [row for rows in by_split.values() for row in rows]
    assert unmapped == 1
    assert {row["session_id"] for row in all_rows} == {"sess-traj"}
    assert sum(row["steer_label"] for row in all_rows) == 1


@pytest.mark.unit
def test_trajectory_unmapped_reason_is_a_stable_key() -> None:
    assert TRAJECTORY_UNMAPPED_REASON == "trajectory_anchor_unmapped"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("origin_path", "expected"),
    [
        pytest.param(
            "/Users/yasyf/.cc-pushback/mirrors/yasyf/-Users-yasyf-Code-cc-notes/s.jsonl",
            "/Users/yasyf/.cc-steer/mirrors/yasyf/-Users-yasyf-Code-cc-notes/s.jsonl",
            id="legacy-mirror-rewritten",
        ),
        pytest.param(
            "/Users/yasyf/.cc-steer/mirrors/yasyf/-Users-yasyf/s.jsonl",
            "/Users/yasyf/.cc-steer/mirrors/yasyf/-Users-yasyf/s.jsonl",
            id="current-mirror-untouched",
        ),
        pytest.param(
            "/Users/yasyf/.claude/projects/-Users-yasyf-Code-yclaw/s.jsonl",
            "/Users/yasyf/.claude/projects/-Users-yasyf-Code-yclaw/s.jsonl",
            id="default-projects-untouched",
        ),
        pytest.param(None, None, id="none-passes-through"),
    ],
)
def test_resolve_origin_path_rewrites_only_the_legacy_mirror(origin_path: str | None, expected: str | None) -> None:
    assert resolve_origin_path(origin_path) == expected


@pytest.mark.unit
async def test_trajectory_rows_resolves_legacy_origin_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    activity = trajectory_activity()
    seen: list[object] = []

    async def capture(session_id: object, origin_path: object) -> object:
        seen.append(origin_path)
        return activity

    monkeypatch.setattr(export_mod, "load_activity", capture)
    await trajectory_rows(
        [
            {
                "is_steering": True,
                "session_id": "sess-traj",
                "event_uuid": "uB",
                "category": "wrong_approach",
                "meta": json.dumps({"origin_path": "/Users/yasyf/.cc-pushback/mirrors/yasyf/-Users-yasyf/s.jsonl"}),
            }
        ]
    )
    assert seen == ["/Users/yasyf/.cc-steer/mirrors/yasyf/-Users-yasyf/s.jsonl"]
