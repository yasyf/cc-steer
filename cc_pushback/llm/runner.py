from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from cc_pushback.llm.backends import CodexBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_pushback.llm.backends import LlmBackend, TModel
    from cc_pushback.llm.prompt import PromptMessage

CONCURRENCY = 4
DEFAULT_TIMEOUT = 180


def call_cli(
    args: list[str],
    *,
    input: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        args,
        input=input,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ | (env or {}),
    )
    if result.returncode != 0:
        err = subprocess.CalledProcessError(result.returncode, args, output=result.stdout, stderr=result.stderr)
        err.add_note(f"argv: {args}")
        err.add_note(f"exit_code: {result.returncode}")
        err.add_note(f"stderr: {result.stderr[-4096:]}")
        err.add_note(f"stdout: {result.stdout[-4096:]}")
        raise err
    return result.stdout


def schema_path_for(backend: LlmBackend, model: type[BaseModel]) -> str:
    """Materialize a JSON schema for ``model`` in the form ``backend`` expects.

    Args:
        backend: The backend that will consume the schema.
        model: The pydantic model whose schema constrains structured output.

    Returns:
        A filesystem path for codex, or the schema string itself for claude.
    """
    schema = json.dumps(model.model_json_schema() | {"additionalProperties": False})
    match backend:
        case CodexBackend():
            fd, path = tempfile.mkstemp(suffix=".json")
            os.write(fd, schema.encode())
            os.close(fd)
            return path
        case _:
            return schema


def call_llm(
    backend: LlmBackend,
    prompt: PromptMessage,
    response_model: type[BaseModel],
    *,
    model: TModel = "small",
    timeout: int = DEFAULT_TIMEOUT,
) -> BaseModel:
    """Run one structured LLM invocation through ``backend`` and validate the result.

    Args:
        backend: The CLI backend to invoke.
        prompt: The prompt rendered to stdin via ``str(prompt)``.
        response_model: The pydantic model the response is validated against.
        model: Abstract model size to map onto the backend's model name.
        timeout: Seconds before the subprocess is killed.

    Returns:
        A validated instance of ``response_model``.
    """
    command = backend.build_command(backend.models[model], schema_path_for(backend, response_model), agent=False)
    raw = call_cli(command, input=str(prompt), timeout=timeout, env=backend.env())
    return cast(BaseModel, backend.parse_response(raw, response_model))


async def classify_batch(
    backend: LlmBackend,
    prompts: Sequence[PromptMessage],
    response_model: type[BaseModel],
    *,
    model: TModel = "small",
) -> list[BaseModel]:
    """Classify ``prompts`` concurrently, preserving input order.

    Runs up to :data:`CONCURRENCY` invocations at once, each off the event loop
    via :func:`asyncio.to_thread`.

    Args:
        backend: The CLI backend to invoke for every prompt.
        prompts: The prompts to classify, one invocation each.
        response_model: The pydantic model each response is validated against.
        model: Abstract model size to map onto the backend's model name.

    Returns:
        The validated results in the same order as ``prompts``.
    """
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def classify(prompt: PromptMessage) -> BaseModel:
        async with semaphore:
            return await asyncio.to_thread(call_llm, backend, prompt, response_model, model=model)

    return list(await asyncio.gather(*(classify(prompt) for prompt in prompts)))
