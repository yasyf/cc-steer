from __future__ import annotations

import os
import subprocess

__all__ = ["call_cli"]


def call_cli(
    args: list[str],
    *,
    input: str | None = None,
    timeout: int = 30,
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
