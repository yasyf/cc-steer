from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from cc_pushback.llm import (
    CONCURRENCY,
    ClaudeBackend,
    CodexBackend,
    PromptMessage,
    call_cli,
    call_llm,
    classify_batch,
    schema_path_for,
)


class Label(BaseModel):
    severity: str
    rule: str


@pytest.mark.parametrize(
    ("backend", "schema_path", "agent", "expected"),
    [
        pytest.param(
            CodexBackend(),
            None,
            False,
            [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                "gpt-5.3-codex-spark",
                "-c",
                "features.codex_hooks=false",
                "-c",
                "features.mcp_servers=false",
            ],
            id="codex-no-schema",
        ),
        pytest.param(
            CodexBackend(),
            "/tmp/s.json",
            False,
            [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                "gpt-5.3-codex-spark",
                "-c",
                "features.codex_hooks=false",
                "-c",
                "features.mcp_servers=false",
                "--output-schema",
                "/tmp/s.json",
            ],
            id="codex-with-schema",
        ),
        pytest.param(
            ClaudeBackend(),
            None,
            False,
            [
                "claude",
                "-p",
                "--no-session-persistence",
                "--model",
                "haiku",
                "--system-prompt",
                "",
                "--setting-sources",
                "",
                "--strict-mcp-config",
            ],
            id="claude-no-schema",
        ),
        pytest.param(
            ClaudeBackend(),
            "{}",
            False,
            [
                "claude",
                "-p",
                "--no-session-persistence",
                "--model",
                "haiku",
                "--system-prompt",
                "",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--json-schema",
                "{}",
                "--output-format",
                "json",
            ],
            id="claude-with-schema",
        ),
    ],
)
def test_build_command_golden_argv(
    backend: CodexBackend | ClaudeBackend, schema_path: str | None, agent: bool, expected: list[str]
) -> None:
    assert backend.build_command(backend.models["small"], schema_path, agent) == expected


def test_call_llm_parses_claude_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps([{"type": "result", "structured_output": {"severity": "major", "rule": "crash instead"}}])

    def fake(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", fake)

    result = call_llm(ClaudeBackend(), PromptMessage().system("hi"), Label)

    assert result == Label(severity="major", rule="crash instead")


def test_call_llm_parses_codex_raw_json_and_writes_schema_tempfile(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    def fake(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        schema_file = Path(args[args.index("--output-schema") + 1])
        assert schema_file.is_file()
        seen["schema"] = json.loads(schema_file.read_text())
        return subprocess.CompletedProcess(
            args, 0, stdout='{"severity": "nit", "rule": "use a comprehension"}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake)

    result = call_llm(CodexBackend(), PromptMessage().system("hi"), Label)

    assert result == Label(severity="nit", rule="use a comprehension")
    assert seen["schema"]["additionalProperties"] is False


def test_call_cli_nonzero_exit_raises_with_diagnostic_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 3, stdout="out-tail", stderr="err-tail")

    monkeypatch.setattr(subprocess, "run", fake)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        call_cli(["claude", "-p"])

    assert excinfo.value.returncode == 3
    assert excinfo.value.__notes__ == [
        "argv: ['claude', '-p']",
        "exit_code: 3",
        "stderr: err-tail",
        "stdout: out-tail",
    ]


def test_classify_batch_preserves_order_and_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = [PromptMessage().ask(f"q{i}") for i in range(12)]
    live = 0
    peak = 0

    def stub(backend: object, prompt: PromptMessage, response_model: type[Label], *, model: str) -> Label:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            return Label(severity="nit", rule=prompt.ask_text)
        finally:
            live -= 1

    monkeypatch.setattr("cc_pushback.llm.runner.call_llm", stub)

    results = [cast(Label, r) for r in asyncio.run(classify_batch(CodexBackend(), prompts, Label))]

    assert [r.rule for r in results] == [f"q{i}" for i in range(12)]
    assert peak <= CONCURRENCY


def test_schema_path_for_claude_returns_schema_string() -> None:
    schema = json.loads(schema_path_for(ClaudeBackend(), Label))

    assert schema["additionalProperties"] is False
    assert schema["properties"]["severity"]["type"] == "string"


def test_str_renders_system_contexts_and_ask() -> None:
    prompt = (
        PromptMessage()
        .system("be terse")
        .context("taxonomy", "no-fallback — crash instead")
        .context("event", "claude added a fallback")
        .ask("classify it")
    )

    assert str(prompt) == (
        "be terse\n\n"
        "<taxonomy>\nno-fallback — crash instead\n</taxonomy>\n\n"
        "<event>\nclaude added a fallback\n</event>\n\n"
        "classify it"
    )


@pytest.mark.parametrize("content", [None, "", "   ", "\n\n"], ids=["none", "empty", "spaces", "blank-lines"])
def test_context_skips_empty_content(content: str | None) -> None:
    assert PromptMessage().context("event", content).contexts == ()


def test_from_template_missing_var_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="template variable 'name' not supplied"):
        PromptMessage.from_template("hello {name}")


def test_load_reads_classify_prompt() -> None:
    prompt = PromptMessage.load("classify")

    assert prompt.system_text.startswith("You are mining one piece of developer pushback")
    assert "Be literal." in prompt.system_text
