"""A thin shell-out to the ``claude`` CLI for a single headless completion.

Lifted from cc-sentiment's engine, right-sized to one synchronous call: it uses the
user's existing Claude Code auth (no API key) and depends only on the standard
library, so the package stays pure and offline unless ``claude`` is actually on the
path.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import anyio

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
    argv = [
        "claude", "-p", prompt,
        "--model", model,
        "--system-prompt", system,
        "--output-format", "json",
        "--max-turns", "1",
        "--tools", "",
        "--disable-slash-commands",
    ]
    try:
        with anyio.fail_after(CLAUDE_TIMEOUT):
            result = await anyio.run_process(argv, check=True)
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(argv, CLAUDE_TIMEOUT) from exc
    data = json.loads(result.stdout)
    if data["is_error"]:
        raise subprocess.CalledProcessError(0, argv, output=result.stdout, stderr=result.stderr)
    return data["result"]
