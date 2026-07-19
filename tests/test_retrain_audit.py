from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import spawnllm
from athome.research.common import canonical_json
from athome.research.golden import GoldenGateViolation
from athome.research.judge import Pairwise

from cc_steer import registry
from cc_steer.retrain import audit, evalset, golden, judged

if TYPE_CHECKING:
    from pathlib import Path

MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string())])
BUDGET = 0.5
WILSON_1_15 = (0.011866588606194828, 0.29817077552423993)  # Wilson score interval for 1/15 at z=1.96
WILSON_2_15 = (0.0373604698913593, 0.3788249920651624)


def watcher_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.table(
        {
            "prompt": pa.array([[{"role": "user", "content": row["content"]}] for row in rows], type=pa.list_(MESSAGE)),
            "completion": pa.array(
                [[{"role": "assistant", "content": "steer" if row["label"] else "NO_STEER"}] for row in rows],
                type=pa.list_(MESSAGE),
            ),
            "verbatim": ["v" if row["label"] else "" for row in rows],
            "label": [row["label"] for row in rows],
            "id": [row["id"] for row in rows],
            "category": [row["category"] for row in rows],
            "source_kind": [row["source_kind"] for row in rows],
            "session_id": [f"s{i}" for i in range(len(rows))],
            "split": ["test"] * len(rows),
        }
    )


def warranted(row_id: str, content: str) -> dict[str, object]:
    return {"id": row_id, "label": True, "category": "wrong_approach", "source_kind": "", "content": content, "prob": 0.4}


def negative(row_id: str, content: str) -> dict[str, object]:
    return {"id": row_id, "label": False, "category": "", "source_kind": "", "content": content, "prob": 0.7}


def audit_rows() -> list[dict[str, object]]:
    # Fire scores 0.6 (warranted) vs 0.3 (negative) => auc_frame 1.0; OVERTURN is one FP, PROMOTE one FN.
    return (
        [warranted(f"w{i}", f"WARRANT=yes context w{i}") for i in range(14)]
        + [warranted("wov", "WARRANT=yes OVERTURN context")]
        + [negative(f"n{i}", f"WARRANT=no context n{i}") for i in range(14)]
        + [negative("npr", "WARRANT=no PROMOTE context")]
    )


def setup_authored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> tuple[Path, str]:
    dataset = tmp_path / "dataset"
    (dataset / "watcher").mkdir(parents=True)
    pq.write_table(watcher_table(rows), dataset / "watcher" / "test.parquet")
    eval_dir = tmp_path / "eval"
    evalset.freeze_eval("watcher", dataset_dir=dataset, root=eval_dir)
    frame = evalset.EvalFrame.load(root=eval_dir)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    incumbent = registry.register(
        golden.WATCHER_COMPONENT,
        {"adapter.bin": b"stub"},
        {"base_model": "mlx", "render_version": 2, "thresholds": {"budget": BUDGET}, "dataset_digest": "d"},
    )
    registry.promote(golden.WATCHER_COMPONENT, incumbent.version)
    evalset.write_probs(frame, incumbent.version, {row["id"]: float(row["prob"]) for row in rows}, auc=0.5, root=eval_dir)
    return eval_dir, incumbent.version


def fill_labels(eval_dir: Path) -> None:
    directory = judged.golden_dir(root=eval_dir)
    stratum = {row["row_id"]: row["stratum"] for row in json.loads((directory / "manifest.json").read_text())["rows"]}
    (directory / golden.LABELS_NAME).write_text(
        json.dumps(
            [
                {"row": entry["row"], "row_id": entry["row_id"], "label": "yes" if stratum[entry["row_id"]] == golden.WARRANTED else "no"}
                for entry in json.loads((directory / "labels_template.json").read_text())
            ]
        )
    )


def _slots(prompt: str) -> tuple[str, str]:
    _, rest = prompt.split("--- A ---\n", 1)
    a, b = rest.split("\n\n--- B ---\n", 1)
    return a, b.rstrip("\n")


def decide(prompt: str) -> str:
    # Garbage loses, identical slots tie, STEER wins on WARRANT=yes; OVERTURN and PROMOTE flip the call.
    a, b = _slots(prompt)
    match (judged.GARBAGE_TEXT in a, judged.GARBAGE_TEXT in b):
        case (True, False):
            return "B"
        case (False, True):
            return "A"
    if a == b:
        return "tie"
    steer = "A" if a.startswith(f"[{judged.FIRE_ACTION}]") else "B"
    warrant = ("WARRANT=yes" in prompt or "PROMOTE" in prompt) and "OVERTURN" not in prompt
    return steer if warrant else ("B" if steer == "A" else "A")


class FakeExtract:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, response_model: type[Pairwise], **_kw: Any) -> Pairwise:
        self.prompts.append(prompt)
        return Pairwise(winner=decide(prompt))


class TestWilsonInterval:
    def test_matches_hand_computed_values(self) -> None:
        assert audit.wilson_interval(1, 15) == pytest.approx(WILSON_1_15)
        assert audit.wilson_interval(2, 15) == pytest.approx(WILSON_2_15)

    def test_empty_stratum_is_zero(self) -> None:
        assert audit.wilson_interval(0, 0) == (0.0, 0.0)


class TestRunWarrantAudit:
    @pytest.mark.anyio
    async def test_end_to_end_corrects_labels_and_journals(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, version = setup_authored(tmp_path, monkeypatch, audit_rows())
        await golden.author_packet(root=eval_dir)
        fill_labels(eval_dir)
        fake = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", fake)

        line = await audit.run_warrant_audit(root=eval_dir)

        assert line.startswith(f"{audit.JOURNAL_COMPONENT}: {version}")
        assert fake.prompts  # panel, controls, and every candidate vote bought through the mocked backend

        sidecar = json.loads((eval_dir / audit.AUDIT_DIRNAME / audit.SIDECAR_NAME).read_text())
        assert sidecar["strata"]["warranted"]["fp"] == 1
        assert sidecar["strata"]["negative"]["fn"] == 1
        assert sidecar["strata"]["warranted"]["fp_ci"] == pytest.approx(list(WILSON_1_15))
        assert sidecar["strata"]["negative"]["fn_ci"] == pytest.approx(list(WILSON_1_15))
        assert sidecar["paired_auc"]["frame"] == 1.0
        assert sidecar["paired_auc"]["corrected"] == pytest.approx(14 / 15)
        assert sidecar["meta"]["seeds"] == [1729, 2718]
        assert sidecar["meta"]["incumbent_version"] == version
        assert sidecar["meta"]["judge"]["provider"]
        assert sidecar["meta"]["watcher_eval_sha256"] == json.loads((eval_dir / evalset.MANIFEST_NAME).read_text())[evalset.WATCHER_EVAL_NAME]
        assert sidecar["meta"]["self_sha256"] == hashlib.sha256(canonical_json(sidecar["rows"])).hexdigest()

        entry = json.loads((tmp_path / "retrain" / "journal.jsonl").read_text().splitlines()[-1])
        assert entry["component"] == audit.JOURNAL_COMPONENT
        assert entry["version"] == version
        assert entry["metrics"]["fp"] == 1.0
        assert entry["metrics"]["fn"] == 1.0
        assert entry["metrics"]["auc_frame"] == 1.0
        assert entry["metrics"]["auc_corrected"] == pytest.approx(14 / 15)
        assert entry["metrics"]["n_flagged"] == 0.0

    @pytest.mark.anyio
    async def test_unlabeled_packet_raises_before_any_spend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, _version = setup_authored(tmp_path, monkeypatch, audit_rows())
        await golden.author_packet(root=eval_dir)  # packet + fires, but no human labels.json
        fake = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", fake)
        with pytest.raises(GoldenGateViolation, match="refusing to fabricate labels"):
            await audit.run_warrant_audit(root=eval_dir)
        assert fake.prompts == []

    @pytest.mark.anyio
    async def test_drifted_audit_provenance_raises_before_any_spend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_dir, _version = setup_authored(tmp_path, monkeypatch, audit_rows())
        await golden.author_packet(root=eval_dir)
        fill_labels(eval_dir)
        manifest_path = judged.golden_dir(root=eval_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["audit"]["stratum_counts"]["warranted"] = 99  # provenance no longer matches the recomputed sample
        manifest_path.write_text(json.dumps(manifest))
        fake = FakeExtract()
        monkeypatch.setattr(spawnllm, "extract", fake)
        with pytest.raises(audit.WarrantAuditError, match="does not match the sample recomputed"):
            await audit.run_warrant_audit(root=eval_dir)
        assert fake.prompts == []
