"""A thin shell-out to the ``claude`` CLI for a single headless completion.

Argv construction and envelope parsing come from the shared ``spawnllm`` library;
the spawn stays local (``anyio.run_process``). It uses the user's existing Claude
Code auth (no API key), so the package stays offline unless ``claude`` is
actually on the path.
"""

from __future__ import annotations

import shutil
import subprocess

import anyio
from spawnllm import ClaudeCliBackend, parse_result_envelope

CLAUDE_TIMEOUT = 180


def claude_available() -> bool:
    """Returns whether the ``claude`` CLI is on ``PATH``."""
    return shutil.which("claude") is not None


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
