"""A thin shell-out to the ``claude`` CLI for a single headless text completion.

The run is driven by the shared ``spawnllm`` library: :func:`spawnllm.run`
spawns ``claude``, retries transient envelopes, and returns a
:class:`spawnllm.Response` whose ``result`` is the unwrapped ``{is_error,
result}`` JSON text and whose ``error`` carries any provider failure. It uses the
user's existing Claude Code auth (no API key), so the package stays offline unless
``claude`` is actually on the path. The structured path lives in
:mod:`cc_transcript.judge` (``run_structured``/``structured_judge``).
"""

from __future__ import annotations

import shutil
import subprocess

from spawnllm import ClaudeConfig, RunSpec, run

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
    spec = RunSpec(
        prompt=prompt,
        model=model,
        timeout=CLAUDE_TIMEOUT,
        provider_configs={
            "claude": ClaudeConfig(
                system_prompt=system,
                max_turns=1,
                tools="",
                disable_slash_commands=True,
                output_format="json",
            )
        },
    )
    try:
        resp = await run(spec)
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(cmd="claude", timeout=CLAUDE_TIMEOUT) from exc
    if resp.error is not None:
        raise subprocess.SubprocessError(resp.error)
    return resp.result
