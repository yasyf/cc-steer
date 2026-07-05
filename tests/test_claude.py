from __future__ import annotations

import subprocess

import pytest
from spawnllm import BackendCallError, Error, Output, Response, Result, RunSpec

from cc_steer.claude import claude_available, run_claude


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
    assert await run_claude("PROMPT", system="SYS", model="claude-x") == "hello"
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
