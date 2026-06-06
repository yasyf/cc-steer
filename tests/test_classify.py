from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

import cc_pushback.classify as classify_mod
import cc_pushback.cli as cli_mod
from cc_pushback.classify import ClassifyResponse
from cc_pushback.cli import main
from tests.builders import assistant_tool_use, denial_result, user_text, write_transcript

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_pushback.llm import LlmBackend, PromptMessage, TModel

SEED = [
    user_text("don't add a fallback, crash instead"),
    assistant_tool_use("e1", "Edit", {"file_path": "/a.py"}),
    denial_result("e1", "rename it to feedback_db, and ask me first next time"),
    user_text("run these tests in parallel"),
]


def db_with_seed(tmp_path: Path, runner: CliRunner) -> Path:
    projects = tmp_path / "projects"
    projects.mkdir()
    write_transcript(projects / "a.jsonl", SEED)
    db = tmp_path / "feedback.db"
    result = runner.invoke(main, ["scan", "--transcripts", str(projects), "--no-github", "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


def rows(db: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(r) for r in conn.execute("SELECT * FROM pattern_matches ORDER BY feedback_id, backend, pattern_name")
        ]
    finally:
        conn.close()


def fake_classify(*responses: ClassifyResponse) -> object:
    async def stub(
        backend: LlmBackend, prompts: Sequence[PromptMessage], response_model: type[ClassifyResponse], *, model: TModel
    ) -> list[ClassifyResponse]:
        assert len(prompts) == len(responses), f"expected {len(responses)} prompts, got {len(prompts)}"
        return list(responses)

    return stub


def classify(runner: CliRunner, db: Path, *extra: str) -> str:
    result = runner.invoke(main, ["classify", "--db", str(db), *extra])
    assert result.exit_code == 0, result.output
    return result.output


def test_no_llm_writes_only_matcher_rows(tmp_path: Path) -> None:
    runner = CliRunner()
    db = db_with_seed(tmp_path, runner)

    output = classify(runner, db, "--no-llm")

    assert "events: 3" in output
    assert "matcher rows: 4" in output
    assert "llm rows: 0" in output
    written = rows(db)
    assert {r["backend"] for r in written} == {"matcher"}
    assert {(r["feedback_id"], r["pattern_name"]) for r in written} == {
        (1, "no-defensive-coding"),
        (2, "parallelize-work"),
        (3, "ask-before-assuming"),
        (3, "denied-edit"),
    }
    assert all(r["taxonomy_version"] == "v1" and r["prompt_version"] == "v1" for r in written)

    rerun = classify(runner, db, "--no-llm")
    assert "matcher rows: 4" in rerun
    assert "new: 0" in rerun
    assert len(rows(db)) == 4


def test_llm_pass_writes_named_and_novel_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    db = db_with_seed(tmp_path, runner)
    monkeypatch.setattr(
        classify_mod,
        "classify_batch",
        fake_classify(
            ClassifyResponse(
                pattern_names=("no-defensive-coding",),
                severity="major",
                what_claude_did="added a fallback",
                rule="crash instead",
            ),
            ClassifyResponse(
                pattern_names=(),
                novel_pattern="prefer-pytest-fixtures",
                severity="nit",
                what_claude_did="wrote a bespoke fixture",
                rule="reuse the shared fixture",
            ),
            ClassifyResponse(
                pattern_names=("parallelize-work",),
                severity="minor",
                what_claude_did="ran tests serially",
                rule="run tests in parallel",
            ),
        ),
    )

    output = classify(runner, db, "--backend", "claude")

    assert "llm rows: 3" in output
    assert "novel proposals: prefer-pytest-fixtures" in output
    llm_rows = [r for r in rows(db) if r["backend"] == "claude"]
    assert len(llm_rows) == 3
    by_name = {(r["feedback_id"], r["pattern_name"]): r for r in llm_rows}
    assert by_name[(1, "no-defensive-coding")]["severity"] == "major"
    assert by_name[(1, "no-defensive-coding")]["model"] == "haiku"
    assert by_name[(1, "no-defensive-coding")]["novel"] == 0
    novel = by_name[(2, "prefer-pytest-fixtures")]
    assert novel["novel"] == 1
    assert novel["rule"] == "reuse the shared fixture"


def test_backend_column_reflects_chosen_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    db = db_with_seed(tmp_path, runner)
    monkeypatch.setattr(
        classify_mod,
        "classify_batch",
        fake_classify(
            ClassifyResponse(pattern_names=("minimal-scope",), severity="major", what_claude_did="x", rule="y"),
            ClassifyResponse(pattern_names=(), severity="nit", what_claude_did="x", rule="y"),
            ClassifyResponse(pattern_names=(), severity="nit", what_claude_did="x", rule="y"),
        ),
    )

    classify(runner, db, "--backend", "codex", "--model", "large")

    codex_rows = [r for r in rows(db) if r["backend"] == "codex"]
    assert len(codex_rows) == 3
    assert all(r["model"] == "gpt-5.5" for r in codex_rows)
    assert {r["pattern_name"] for r in codex_rows} == {"minimal-scope", "none"}
    assert {r["backend"] for r in rows(db)} == {"matcher", "codex"}


def test_classify_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    db = db_with_seed(tmp_path, runner)
    canned = fake_classify(
        *(ClassifyResponse(pattern_names=(), severity="nit", what_claude_did="x", rule="y") for _ in range(3))
    )
    monkeypatch.setattr(classify_mod, "classify_batch", canned)

    first = classify(runner, db, "--backend", "claude")
    assert "events: 3" in first
    assert "llm rows: 3" in first

    monkeypatch.setattr(classify_mod, "classify_batch", fake_classify())
    second = classify(runner, db, "--backend", "claude")
    assert "events: 0" in second
    assert "new: 0" in second


def test_taxonomy_bump_reopens_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    db = db_with_seed(tmp_path, runner)
    canned = fake_classify(
        *(ClassifyResponse(pattern_names=(), severity="nit", what_claude_did="x", rule="y") for _ in range(3))
    )
    monkeypatch.setattr(classify_mod, "classify_batch", canned)

    classify(runner, db, "--backend", "claude")
    monkeypatch.setattr(classify_mod, "classify_batch", fake_classify())
    assert "events: 0" in classify(runner, db, "--backend", "claude")

    monkeypatch.setattr(classify_mod, "TAXONOMY_VERSION", "v2")
    monkeypatch.setattr(cli_mod, "TAXONOMY_VERSION", "v2")
    monkeypatch.setattr(classify_mod, "classify_batch", canned)

    reopened = classify(runner, db, "--backend", "claude")
    assert "events: 3" in reopened
    assert {r["taxonomy_version"] for r in rows(db)} == {"v1", "v2"}


def test_limit_restricts_loaded_events(tmp_path: Path) -> None:
    runner = CliRunner()
    db = db_with_seed(tmp_path, runner)

    output = classify(runner, db, "--no-llm", "--limit", "2")

    assert "events: 2" in output
    assert {r["feedback_id"] for r in rows(db)} == {1, 2}
