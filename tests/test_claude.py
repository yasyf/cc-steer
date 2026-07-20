from __future__ import annotations

import json
import subprocess

import pytest
from spawnllm import BackendCallError, Error, Output, Response, Result, RunSpec

from cc_steer.claude import ClaudeUsage, claude_available, run_claude, usage_of


def envelope(*, cost: float | None) -> str:
    event: dict[str, object] = {
        "type": "result",
        "is_error": False,
        "result": "hello",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 3,
        },
    }
    if cost is not None:
        event["total_cost_usd"] = cost
    return json.dumps(event)


def test_claude_available_reflects_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.claude.shutil.which", lambda _: "/usr/bin/claude")
    assert claude_available() is True
    monkeypatch.setattr("cc_steer.claude.shutil.which", lambda _: None)
    assert claude_available() is False


@pytest.mark.anyio
async def test_run_claude_builds_spec_and_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, RunSpec] = {}

    async def fake_run(spec: RunSpec, **_: object) -> Response:
        captured["spec"] = spec
        return Response(spec=spec, output=Output(raw="hello"), result=Result(raw="hello"))

    monkeypatch.setattr("cc_steer.claude.run", fake_run)
    result = await run_claude("PROMPT", system="SYS", model="claude-x")
    assert result.text == "hello"
    assert result.usage is None
    spec = captured["spec"]
    assert spec.prompt == "PROMPT"
    assert spec.model == "claude-x"
    config = spec.provider_configs["claude"]
    assert config.system_prompt == "SYS"
    assert config.max_turns == 1
    assert config.tools == ""
    assert config.disable_slash_commands is True
    assert config.output_format == "json"


@pytest.mark.anyio
async def test_run_claude_reads_usage_from_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(spec: RunSpec, **_: object) -> Response:
        return Response(spec=spec, output=Output(raw=envelope(cost=0.0123)), result=Result(raw="hello"))

    monkeypatch.setattr("cc_steer.claude.run", fake_run)
    result = await run_claude("p", system="s", model="m")
    assert result.text == "hello"
    assert result.usage == ClaudeUsage(
        input_tokens=10, output_tokens=20, cache_read_input_tokens=5, cache_creation_input_tokens=3, cost_usd=0.0123
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            envelope(cost=0.0123),
            ClaudeUsage(
                input_tokens=10,
                output_tokens=20,
                cache_read_input_tokens=5,
                cache_creation_input_tokens=3,
                cost_usd=0.0123,
            ),
            id="with-cost",
        ),
        pytest.param(
            envelope(cost=None),
            ClaudeUsage(
                input_tokens=10,
                output_tokens=20,
                cache_read_input_tokens=5,
                cache_creation_input_tokens=3,
                cost_usd=None,
            ),
            id="cost-absent",
        ),
        pytest.param("plain non-json text", None, id="no-envelope"),
    ],
)
def test_usage_of(raw: str, expected: ClaudeUsage | None) -> None:
    assert usage_of(raw) == expected


@pytest.mark.anyio
async def test_run_claude_raises_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(spec: RunSpec, **_: object) -> Response:
        return Response(
            spec=spec,
            output=Output(raw=""),
            error=Error(msg="claude reported an error", ex=BackendCallError("claude reported an error")),
        )

    monkeypatch.setattr("cc_steer.claude.run", fake_run)
    with pytest.raises(subprocess.SubprocessError, match="claude reported an error"):
        await run_claude("p", system="s", model="m")


@pytest.mark.anyio
async def test_run_claude_surfaces_timeout_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(spec: RunSpec, **_: object) -> Response:
        return Response(
            spec=spec,
            output=Output(raw=""),
            error=Error(msg="claude timed out after 180s", ex=TimeoutError()),
        )

    monkeypatch.setattr("cc_steer.claude.run", fake_run)
    with pytest.raises(subprocess.SubprocessError, match="timed out"):
        await run_claude("p", system="s", model="m")
