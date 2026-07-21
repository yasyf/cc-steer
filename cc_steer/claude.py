"""A thin shell-out to the ``claude`` CLI for a single headless text completion.

The run is driven by the shared ``spawnllm`` library: :func:`spawnllm.run`
spawns ``claude``, retries transient envelopes, and returns a
:class:`spawnllm.Response` that carries the spec, the raw output, and exactly one
of ``result``/``error``. On success ``resp.result.raw`` is the unwrapped
``{is_error, result}`` JSON text and ``resp.output.raw`` is the full JSON
envelope the token accounting is read from; on any failure — nonzero exit, error
envelope, or timeout — ``resp.error`` carries the message and underlying
exception, and :func:`spawnllm.run` never raises. It uses the user's existing
Claude Code auth (no API key), so the package stays offline unless ``claude`` is
actually on the path.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from spawnllm import ClaudeCliBackend, ClaudeConfig, Error, Response, Result, RunSpec, run

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import BaseModel
    from spawnllm import TModel

CLAUDE_TIMEOUT = 180
CLAUDE_BACKEND = ClaudeCliBackend()


@dataclass(frozen=True, slots=True)
class ClaudeUsage:
    """Token accounting for one headless ``claude`` turn.

    Attributes:
        input_tokens: The uncached input tokens the turn consumed.
        output_tokens: The output tokens the turn produced.
        cache_read_input_tokens: The input tokens served from the prompt cache.
        cache_creation_input_tokens: The input tokens written to the prompt cache.
        cost_usd: The turn's billed cost in US dollars, or None when the envelope
            omits it.
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float | None

    @classmethod
    def of(cls, cost_usd: float | None, usage: Mapping[str, object]) -> ClaudeUsage:
        return cls(
            input_tokens=int(str(usage["input_tokens"])),
            output_tokens=int(str(usage["output_tokens"])),
            cache_read_input_tokens=int(str(usage["cache_read_input_tokens"])),
            cache_creation_input_tokens=int(str(usage["cache_creation_input_tokens"])),
            cost_usd=cost_usd,
        )


@dataclass(frozen=True, slots=True)
class ClaudeResult:
    """The text and token accounting of one headless ``claude`` turn.

    Attributes:
        text: The assistant's text response — the ``result`` field of the JSON output.
        usage: The turn's token accounting, or None when the JSON envelope carried
            no ``usage`` object (a mocked or non-JSON output).
    """

    text: str
    usage: ClaudeUsage | None


def claude_available() -> bool:
    """Returns whether the ``claude`` CLI is on ``PATH``."""
    return shutil.which("claude") is not None


async def run_claude(prompt: str, *, system: str, model: str) -> ClaudeResult:
    """Runs one headless ``claude`` turn and returns its text and token accounting.

    Args:
        prompt: The user message to send.
        system: The system prompt.
        model: The model to run, for example ``claude-sonnet-4-6``.

    Returns:
        The assistant's text response paired with the turn's token usage.

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
    match resp := await run(spec):
        case Response(error=Error(msg=msg)):
            raise subprocess.SubprocessError(msg)
        case Response(result=Result(raw=raw)):
            return ClaudeResult(text=raw, usage=usage_of(resp.output.raw))
        case _:
            raise AssertionError(resp)


def usage_of(raw: str) -> ClaudeUsage | None:
    """Reads the ``(cost, usage)`` accounting from a ``claude`` JSON envelope, or None when absent."""
    match CLAUDE_BACKEND.accounting(raw):
        case _, None:
            return None
        case cost, usage:
            return ClaudeUsage.of(cost, usage)


def cached_judge[M: BaseModel](
    response_model: type[M], *, tier: TModel, system: str, timeout: int = CLAUDE_TIMEOUT
) -> Callable[[str], Awaitable[M]]:
    """Returns a prompt-to-verdict callable that hoists the static instruction block
    into the ``claude`` CLI's cache-controlled ``--system-prompt``.

    A drop-in for :func:`cc_transcript.judge.structured_judge` — one structured
    completion per prompt, :class:`~cc_transcript.judge.verdicts.JudgeError` on any
    provider or validation failure — that sends the constant ``system`` block once as
    the system prompt (cache-read on every subsequent row within the CLI's cache TTL)
    instead of resending it in each row's user prompt. Only the per-row user prompt
    varies between calls, so the shared prefix is billed once rather than per row.
    Each turn runs the same stripped harness as :func:`run_claude` — no tools, no
    slash commands — so the Claude Code tool baseline never enters the context.

    Args:
        response_model: The Pydantic model the structured output is validated against.
        tier: The judge's abstract model tier.
        system: The constant instruction block, sent as the cached system prompt.
        timeout: Seconds to wait before the backend process is killed.
    """
    from cc_transcript.judge import default_backend
    from cc_transcript.judge.verdicts import JudgeError

    backend = default_backend()

    async def judge(prompt: str) -> M:
        spec = RunSpec(
            prompt=prompt,
            model=backend.models[tier],
            response_model=response_model,
            timeout=timeout,
            provider_configs={
                # No max_turns: --json-schema's tool_use round makes the turn count
                # vary, so any cap intermittently dies with error_max_turns.
                "claude": ClaudeConfig(system_prompt=system, tools="", disable_slash_commands=True)
            },
        )
        match await run(spec, backend=backend):
            case Response(error=Error(msg=msg)):
                raise JudgeError(msg)
            case Response(result=Result(parsed=parsed)):
                return cast(response_model, parsed)
            case other:
                raise AssertionError(other)

    return judge
