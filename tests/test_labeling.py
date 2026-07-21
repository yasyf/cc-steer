from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import anyio
import pytest
from athome.research.golden import GoldenGateViolation, Stratum, build_packet, write_packet

from cc_steer import labeling
from cc_steer.labeling import blocks, ingest, models
from cc_steer.labeling.models import CATEGORIES
from cc_steer.retrain import judged

if TYPE_CHECKING:
    from pathlib import Path

GOLDEN_WINDOWS: tuple[tuple[str, str], ...] = (
    ("warranted", "assistant proposes a risky rewrite of the auth flow"),
    ("negative", "assistant is on a reasonable track, nothing to act on"),
)


async def mint_golden(eval_dir: Path) -> None:
    directory = judged.golden_dir(root=eval_dir)
    rows = [
        {"row_id": f"g{i}", "stratum": stratum, "window": window} for i, (stratum, window) in enumerate(GOLDEN_WINDOWS)
    ]
    strata = tuple(
        Stratum(name=name, size=sum(row["stratum"] == name for row in rows))
        for name in dict.fromkeys(row["stratum"] for row in rows)
    )
    packet = build_packet(
        rows,
        strata=strata,
        stratum_of=lambda row: str(row["stratum"]),
        window_of=lambda row: str(row["window"]),
        row_id=lambda row: str(row["row_id"]),
        seed=7,
        dataset_digest="dd",
        question="Was steering warranted?",
        header="# Packet\n",
    )
    await write_packet(packet, anyio.Path(directory))
    (directory / judged.FIRES_NAME).write_text(
        "".join(json.dumps({"row_id": row.row_id, "context": row.window}) + "\n" for row in packet.rows)
    )


def golden_item() -> models.LabelItem:
    return models.LabelItem(
        item_id="golden-0001",
        source="golden",
        row_id="g0",
        window=models.FireWindow(text="risky rewrite"),
        row_number=1,
        stratum="warranted",
    )


def adjudication_item() -> models.LabelItem:
    return models.LabelItem(
        item_id="adj-0000",
        source="adjudication",
        row_id="d1",
        window=models.FireWindow(text="ctx d1", detector="loop", session_id="s9", turn=42),
        disputed_category="direction",
        candidate_labels={"medium": "yes", "auditor": "no"},
    )


def warrant(item_id: str, verdict: str) -> dict:
    return {"type": "decision.created", "blockId": f"{item_id}-warrant", "verdict": verdict}


class TestBuildQueue:
    @pytest.mark.anyio
    async def test_golden_and_adjudication_rows(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        disagreements = tmp_path / "disagreements.jsonl"
        disagreements.write_text(
            json.dumps(
                {
                    "row_id": "d1",
                    "context": "assistant picks option B unprompted",
                    "category": "unwanted_action",
                    "labels": {"medium": "yes", "auditor": "no"},
                    "detector": "loop",
                    "session_id": "s9",
                    "turn": 42,
                }
            )
            + "\n"
        )
        queue = labeling.build_queue(disagreements_path=disagreements, root=eval_dir)
        assert len(queue.golden) == 2
        assert len(queue.adjudication) == 1
        assert [item.item_id for item in queue.golden] == ["golden-0001", "golden-0002"]
        assert {item.window.text for item in queue.golden} == {window for _, window in GOLDEN_WINDOWS}
        assert {item.stratum for item in queue.golden} == {"warranted", "negative"}
        adj = queue.adjudication[0]
        assert adj.item_id == "adj-0000"
        assert adj.disputed_category == "unwanted_action"
        assert adj.candidate_labels == {"medium": "yes", "auditor": "no"}
        assert (adj.window.detector, adj.window.session_id, adj.window.turn) == ("loop", "s9", 42)

    @pytest.mark.anyio
    async def test_golden_only_when_no_disagreements_path(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        queue = labeling.build_queue(root=eval_dir)
        assert len(queue.golden) == 2
        assert queue.adjudication == ()


class TestBlockSpec:
    def test_document_sections_and_stats(self) -> None:
        queue = models.LabelQueue(items=(golden_item(), adjudication_item()))
        doc = blocks.render_document(queue)
        assert doc["version"] == 1
        assert doc["submit"]["label"]
        assert [stat["value"] for stat in doc["stats"]] == ["1", "1"]
        ids = [block["id"] for block in doc["blocks"]]
        assert ids == ["sec-golden", "golden-0001", "sec-adjudication", "adj-0000"]

    def test_golden_card_controls(self) -> None:
        card = blocks.item_card(golden_item())
        children = {child["id"]: child for child in card["children"]}
        assert list(children) == [
            "golden-0001-window",
            "golden-0001-warrant",
            "golden-0001-category",
            "golden-0001-flag",
            "golden-0001-note",
        ]
        assert children["golden-0001-window"]["type"] == "code"
        assert children["golden-0001-window"]["code"] == "risky rewrite"
        assert children["golden-0001-warrant"]["type"] == "approval"
        category = children["golden-0001-category"]
        assert category["type"] == "choice" and category["multi"] is False
        assert [option["id"] for option in category["options"]] == list(CATEGORIES)
        flag = children["golden-0001-flag"]
        assert flag["multi"] is True and {option["id"] for option in flag["options"]} == {"hard", "ambiguous"}
        assert children["golden-0001-note"]["type"] == "input" and children["golden-0001-note"]["multiline"] is True

    def test_adjudication_card_carries_dispute_record(self) -> None:
        card = blocks.item_card(adjudication_item())
        assert card["flagged"] is True
        dispute = next(child for child in card["children"] if child["id"] == "adj-0000-dispute")
        assert dispute["type"] == "record"
        labels = {fact["label"]: fact["value"] for fact in dispute["facts"]}
        assert labels == {"disputed category": "direction", "medium": "yes", "auditor": "no"}


class TestReduce:
    def test_last_write_wins_and_multi_flags(self) -> None:
        queue = models.LabelQueue(items=(adjudication_item(),))
        reduced = ingest.reduce_journal(
            queue,
            [
                {"type": "choice.selected", "blockId": "adj-0000-category", "optionIds": ["direction"]},
                {"type": "choice.selected", "blockId": "adj-0000-category", "optionIds": ["premature"]},
                {"type": "choice.selected", "blockId": "adj-0000-flag", "optionIds": ["hard", "ambiguous"]},
                {"type": "input.submitted", "blockId": "adj-0000-note", "text": ""},
            ],
        )
        label = reduced["adj-0000"]
        assert label.category == "premature"
        assert label.flags == ("hard", "ambiguous")
        assert label.note is None
        assert label.warrant is None and label.complete is False

    def test_non_control_events_do_not_create_rows(self) -> None:
        queue = models.LabelQueue(items=(golden_item(),))
        reduced = ingest.reduce_journal(
            queue,
            [
                {"type": "submit", "revision": 1},
                {"type": "channel.changed", "connected": True},
                {"type": "feedback.created", "blockId": "golden-0001-warrant", "id": "x", "text": "hmm"},
                {"type": "decision.created", "blockId": "missing-block-warrant", "verdict": "approved"},
            ],
        )
        assert reduced == {}

    def test_rejects_unknown_verdict(self) -> None:
        queue = models.LabelQueue(items=(golden_item(),))
        with pytest.raises(ingest.LabelingError):
            ingest.reduce_journal(queue, [warrant("golden-0001", "maybe")])


class TestIngestArtifacts:
    @pytest.mark.anyio
    async def test_golden_labels_round_trip_through_load_golden(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        queue = labeling.build_queue(root=eval_dir)
        verdict = {"warranted": "approved", "negative": "rejected"}
        for item in queue.golden:
            labeling.record_event(warrant(item.item_id, verdict[item.stratum or ""]), queue, root=eval_dir)
            labeling.record_event(
                {"type": "choice.selected", "blockId": f"{item.item_id}-category", "optionIds": ["wrong_approach"]},
                queue,
                root=eval_dir,
            )
            labeling.record_event(
                {"type": "input.submitted", "blockId": f"{item.item_id}-note", "text": "seen"}, queue, root=eval_dir
            )
        golden = await judged.load_golden(judged.golden_dir(root=eval_dir))
        assert golden.human == {item.row_id: item.stratum == "warranted" for item in queue.golden}

    @pytest.mark.anyio
    async def test_human_gold_manifest_sha_and_lines(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        queue = labeling.build_queue(root=eval_dir)
        for item in queue.golden:
            labeling.record_event(warrant(item.item_id, "approved"), queue, root=eval_dir)
        hg = labeling.human_gold_dir(root=eval_dir)
        lines = [json.loads(line) for line in (hg / "labels.jsonl").read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(line["warrant"] is True and line["source"] == "golden" for line in lines)
        manifest = json.loads((hg / "MANIFEST.json").read_text())
        assert manifest["n"] == 2
        assert manifest["labels.jsonl"] == hashlib.sha256((hg / "labels.jsonl").read_bytes()).hexdigest()

    @pytest.mark.anyio
    async def test_reingest_and_replay_are_idempotent(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        queue = labeling.build_queue(root=eval_dir)
        for item in queue.golden:
            labeling.record_event(warrant(item.item_id, "approved"), queue, root=eval_dir)
        labels_path = judged.golden_dir(root=eval_dir) / judged.LABELS_NAME
        human_path = labeling.human_gold_dir(root=eval_dir) / "labels.jsonl"
        golden_before, human_before = labels_path.read_bytes(), human_path.read_bytes()
        labeling.rebuild(queue, root=eval_dir)
        assert labels_path.read_bytes() == golden_before
        assert human_path.read_bytes() == human_before
        labeling.record_event(warrant(queue.golden[0].item_id, "approved"), queue, root=eval_dir)
        assert labels_path.read_bytes() == golden_before
        assert human_path.read_bytes() == human_before

    @pytest.mark.anyio
    async def test_golden_labels_absent_until_complete_and_cleared_removes_them(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        queue = labeling.build_queue(root=eval_dir)
        labels_path = judged.golden_dir(root=eval_dir) / judged.LABELS_NAME
        labeling.record_event(warrant(queue.golden[0].item_id, "approved"), queue, root=eval_dir)
        assert not labels_path.exists()
        labeling.record_event(warrant(queue.golden[1].item_id, "rejected"), queue, root=eval_dir)
        assert labels_path.exists()
        labeling.record_event(warrant(queue.golden[0].item_id, "cleared"), queue, root=eval_dir)
        assert not labels_path.exists()
        with pytest.raises(GoldenGateViolation):
            await judged.load_golden(judged.golden_dir(root=eval_dir))


class TestMain:
    @pytest.mark.anyio
    async def test_dry_run_prints_document(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        eval_dir = tmp_path / "eval"
        await mint_golden(eval_dir)
        from cc_steer.labeling.__main__ import main

        main(["--root", str(eval_dir)])
        doc = json.loads(capsys.readouterr().out)
        assert doc["version"] == 1
        assert len([block for block in doc["blocks"] if block["type"] == "card"]) == 2
