"""A thin shell-out to the ``claude`` CLI for a single headless completion.

Argv construction and envelope parsing come from the shared ``spawnllm`` library;
the spawn stays local (``anyio.run_process``). It uses the user's existing Claude
Code auth (no API key), so the package stays offline unless ``claude`` is
actually on the path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import anyio
from spawnllm import ClaudeCliBackend, parse_result_envelope, parse_structured_output, resolve_schema_path, schema_for

if TYPE_CHECKING:
    from pydantic import BaseModel
    from spawnllm import TModel

CLAUDE_TIMEOUT = 180


def claude_available() -> bool:
    """Returns whether the ``claude`` CLI is on ``PATH``."""
    return shutil.which("claude") is not None


def resolved_model(tier: TModel) -> str:
    """Returns the concrete Claude model name for an abstract tier."""
    return ClaudeCliBackend.models[tier]


async def run_claude(prompt: str, *, system: str, model: str) -> str:
    """Runs one headless ``claude`` turn and returns its text result.

    Args:
        prompt: The user message to send.
        system: The system prompt.
        model: The model to run, for example ``claude-sonnet-4-6``.

    Returns:
        The assistant's text response — the ``result`` field of the JSON output.

    Raises:
        subprocess.SubprocessError: If ``claude`` exits non-zero, times out, or
            reports an error in its JSON envelope.
    """
    argv = ClaudeCliBackend.cc_sentiment(system_prompt=system).build_argv(prompt, model=model)
    try:
        with anyio.fail_after(CLAUDE_TIMEOUT):
            result = await anyio.run_process(argv, check=True)
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(argv, CLAUDE_TIMEOUT) from exc
    return parse_result_envelope(result.stdout, argv=argv, stderr=result.stderr)


async def run_claude_structured[M: BaseModel](prompt: str, *, response_model: type[M], tier: TModel = "medium") -> M:
    """Runs one headless ``claude`` turn and parses its structured output.

    The prompt is delivered over stdin and the response is forced into
    ``response_model``'s JSON schema via the CLI's ``--json-schema`` flag. The
    structured path runs with an empty system prompt, so all instructions must
    live in ``prompt``.

    Args:
        prompt: The full prompt, instructions included.
        response_model: The pydantic model the response must validate against.
        tier: The abstract model tier to run, resolved by the Claude backend.

    Returns:
        The validated ``response_model`` instance.

    Raises:
        subprocess.SubprocessError: If ``claude`` exits non-zero or times out.
        pydantic.ValidationError: If the response does not match the schema.
    """
    backend = ClaudeCliBackend()
    argv = backend.build_command(
        backend.models[tier], resolve_schema_path(backend, schema_for(response_model)), agent=False
    )
    try:
        with anyio.fail_after(CLAUDE_TIMEOUT):
            result = await anyio.run_process(argv, input=prompt.encode(), check=True, env=os.environ | backend.env())
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(argv, CLAUDE_TIMEOUT) from exc
    return parse_structured_output(result.stdout.decode(), response_model)
