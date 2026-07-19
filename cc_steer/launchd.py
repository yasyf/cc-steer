"""macOS launchd scheduling: the nightly pipeline agent, the weekly retrain agent, and the always-on watch daemon.

The pipeline agent fires ``cc-steer pipeline run --auto-weekly`` at the
configured hour through a login-ish shell so PATH-provided tooling resolves;
``--auto-weekly`` folds the weekly audit into the Sunday run so one agent
covers both cadences. The retrain agent fires ``cc-steer retrain`` for the gate
lane then the watcher lane every Sunday morning, refreshing each promoted model
when its training data moved. The watch agent runs ``cc-steer watch``
continuously under ``KeepAlive`` — a fail-fast crash respawns rather than
staying dead, and a model promotion takes effect on the next
``launchctl kickstart``. All run through the same ``sh -lc`` + ccp-env-guard
command shape.
"""

from __future__ import annotations

import importlib.metadata
import plistlib
import shlex
import shutil
import subprocess
from pathlib import Path

from cc_steer.hooks import DEFAULT_PREFIX

LABEL = "com.cc-steer.pipeline"
RETRAIN_LABEL = "com.cc-steer.retrain"
WATCH_LABEL = "com.cc-steer.watch"
LOG_DIR = Path.home() / ".cc-steer" / "logs"
SUNDAY = 0  # launchd's StartCalendarInterval weekday numbering
RETRAIN_EXTRAS = "retrain"
WATCH_EXTRAS = "gate,mlx"


def ccp_env_guard() -> str:
    """The sh preamble that spend-routes through ccp when present, never evaluating its errors.

    Runs ``ccp env`` only when ccp is on PATH, captures its stdout, discards stderr, and evals the
    captured environment only when ccp exited 0. A failing ``ccp env`` — error text on stdout, a
    nonzero exit — is dropped rather than executed, and the real command still runs; the earlier
    ``eval "$(ccp env)"`` evaluated that stdout regardless of the exit code.
    """
    return 'if command -v ccp >/dev/null 2>&1; then __ccp_env="$(ccp env 2>/dev/null)" && eval "$__ccp_env"; fi;'


def pinned_spec(extras: str) -> str:
    """The version-pinned uvx dist spec ``cc-steer[extras]==<installed>``; ``extras`` empty drops the bracket.

    The pin is the installed version at plist-generation time, so a launched agent resolves the
    exact wheel (and its extras) this build ships — not whatever uvx would pick per launch, the
    unpinned resolution that served an mlx-less env and drove the watcher crash storm.
    """
    version = importlib.metadata.version("cc-steer")
    if version == "0.0.0":
        raise RuntimeError(
            "install-launchd needs a released cc-steer install: the working-tree sentinel "
            "0.0.0 would pin an unresolvable spec and the KeepAlive agent would crash-loop"
        )
    return f"cc-steer{f'[{extras}]' if extras else ''}=={version}"


def agent_path() -> Path:
    """Where the pipeline LaunchAgent plist lives for the current user."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def retrain_agent_path() -> Path:
    """Where the retrain LaunchAgent plist lives for the current user."""
    return Path.home() / "Library" / "LaunchAgents" / f"{RETRAIN_LABEL}.plist"


def watch_agent_path() -> Path:
    """Where the watch LaunchAgent plist lives for the current user."""
    return Path.home() / "Library" / "LaunchAgents" / f"{WATCH_LABEL}.plist"


def pipeline_command(prefix: str, journal_repo: Path | None) -> str:
    """The shell command the pipeline agent runs, spend-routing through ccp when present."""
    resolved = prefix
    if prefix.split(maxsplit=1)[0] == "uvx" and (uvx := shutil.which("uvx")):
        resolved = prefix.replace("uvx", uvx, 1)
    journal = f" --journal-repo {shlex.quote(str(journal_repo))}" if journal_repo else ""
    return f"{ccp_env_guard()} exec {resolved} pipeline run --auto-weekly{journal}"


def retrain_command(prefix: str) -> str:
    """The shell command the retrain agent runs: the gate lane then the watcher lane, each independent.

    A gate-lane failure must not skip the watcher lane, but launchd must still see it —
    both lanes always run, and the exit status is the OR of theirs.
    """
    resolved = prefix
    if prefix.split(maxsplit=1)[0] == "uvx" and (uvx := shutil.which("uvx")):
        resolved = prefix.replace("uvx", uvx, 1)
    lanes = f"s=0; {resolved} retrain --component gate || s=1; {resolved} retrain --component watcher || s=1; exit $s"
    return f"{ccp_env_guard()} {lanes}"


def watch_command(prefix: str) -> str:
    """The shell command the watch daemon runs: the two-stage watcher, delivering per live.toml.

    No ``--gate-threshold`` override: the daemon serves each promoted gate at the
    threshold its retrain fitted, read from the registry metadata.
    """
    resolved = prefix
    if prefix.split(maxsplit=1)[0] == "uvx" and (uvx := shutil.which("uvx")):
        resolved = prefix.replace("uvx", uvx, 1)
    body = f"exec {resolved} watch --gate lexical --drafter mlx"
    return f"{ccp_env_guard()} {body}"


def render(prefix: str, journal_repo: Path | None, *, hour: int = 3) -> bytes:
    """The plist for the nightly pipeline agent, logging under ``~/.cc-steer/logs``."""
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": ["/bin/sh", "-lc", pipeline_command(prefix, journal_repo)],
            "StartCalendarInterval": {"Hour": hour, "Minute": 0},
            "StandardOutPath": str(LOG_DIR / "pipeline.log"),
            "StandardErrorPath": str(LOG_DIR / "pipeline.log"),
        }
    )


def render_retrain(prefix: str, journal_repo: Path | None, *, hour: int = 4) -> bytes:
    """The plist for the weekly retrain agent: Sundays at ``hour``.

    ``journal_repo`` becomes the agent's working directory so the retrain journal's cc-notes
    mirror resolves that repo; without one the unattended agent has no cwd repo and it no-ops.
    """
    plist: dict[str, object] = {
        "Label": RETRAIN_LABEL,
        "ProgramArguments": ["/bin/sh", "-lc", retrain_command(prefix)],
        "StartCalendarInterval": {"Weekday": SUNDAY, "Hour": hour, "Minute": 0},
        "StandardOutPath": str(LOG_DIR / "retrain.log"),
        "StandardErrorPath": str(LOG_DIR / "retrain.log"),
    }
    if journal_repo is not None:
        plist["WorkingDirectory"] = str(journal_repo)
    return plistlib.dumps(plist)


def render_watch(prefix: str) -> bytes:
    """The plist for the always-on watch daemon: KeepAlive respawns a fail-fast crash."""
    return plistlib.dumps(
        {
            "Label": WATCH_LABEL,
            "ProgramArguments": ["/bin/sh", "-lc", watch_command(prefix)],
            "KeepAlive": True,
            "RunAtLoad": True,
            "StandardOutPath": str(LOG_DIR / "watch.log"),
            "StandardErrorPath": str(LOG_DIR / "watch.log"),
        }
    )


def pipeline_prefix(prefix: str) -> str:
    """The pipeline-lane prefix: the bare default version-pinned to this build's wheel, else untouched.

    A custom prefix is the operator's responsibility; the default resolves ``cc-steer==<installed>``
    so the agent runs the exact build that scheduled it (mirrors :func:`retrain_prefix`).
    """
    return f"uvx --from '{pinned_spec('')}' cc-steer" if prefix == DEFAULT_PREFIX else prefix


def install(prefix: str, journal_repo: Path | None, *, hour: int = 3) -> Path:
    """Writes and (re)loads the pipeline agent; returns the plist path."""
    return _install(agent_path(), render(pipeline_prefix(prefix), journal_repo, hour=hour))


def retrain_prefix(prefix: str) -> str:
    """The retrain-lane prefix: the bare default rewritten to the version-pinned ``retrain`` extra, else untouched.

    The base ``uvx cc-steer`` dist cannot import the watcher lane's Tinker/mlx-lm deps, so the default
    resolves ``cc-steer[retrain]==<installed>`` — the pin is the build that scheduled the agent; a custom
    prefix is the operator's responsibility (mirrors :func:`cc_steer.hooks.live_runner`).
    """
    return f"uvx --from '{pinned_spec(RETRAIN_EXTRAS)}' cc-steer" if prefix == DEFAULT_PREFIX else prefix


def install_retrain(prefix: str, journal_repo: Path | None, *, hour: int = 4) -> Path:
    """Writes and (re)loads the weekly retrain agent; returns the plist path."""
    return _install(retrain_agent_path(), render_retrain(retrain_prefix(prefix), journal_repo, hour=hour))


def watch_prefix(prefix: str) -> str:
    """The watch-daemon prefix: the bare default rewritten to the version-pinned ``gate,mlx`` extras, else untouched.

    The base ``uvx cc-steer`` dist cannot import the lexical gate's scikit-learn or the mlx
    drafter's mlx-lm deps, so the default resolves ``cc-steer[gate,mlx]==<installed>`` — the pin is
    the build that scheduled the agent, the fix for the unpinned resolution that served an mlx-less
    env; a custom prefix is the operator's responsibility (mirrors :func:`retrain_prefix`).
    """
    return f"uvx --from '{pinned_spec(WATCH_EXTRAS)}' cc-steer" if prefix == DEFAULT_PREFIX else prefix


def install_watch(prefix: str) -> Path:
    """Writes and (re)loads the always-on watch daemon; returns the plist path."""
    return _install(watch_agent_path(), render_watch(watch_prefix(prefix)))


def kickstart_watch() -> bool:
    """Kicks the always-on watch agent so a fresh promotion loads; True when the kick succeeds.

    The watch agent may legitimately not be installed, so a nonzero return is reported by the
    caller, never raised.
    """
    return (
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{_uid()}/{WATCH_LABEL}"], capture_output=True, check=False
        ).returncode
        == 0
    )


def uninstall() -> bool:
    """Unloads and removes the pipeline agent; True when one was installed."""
    return _uninstall(agent_path())


def uninstall_retrain() -> bool:
    """Unloads and removes the retrain agent; True when one was installed."""
    return _uninstall(retrain_agent_path())


def uninstall_watch() -> bool:
    """Unloads and removes the watch daemon; True when one was installed."""
    return _uninstall(watch_agent_path())


def _install(path: Path, plist: bytes) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plist)
    subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(path)], capture_output=True, check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{_uid()}", str(path)], capture_output=True, check=True)
    return path


def _uninstall(path: Path) -> bool:
    if not path.exists():
        return False
    subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(path)], capture_output=True, check=False)
    path.unlink()
    return True


def _uid() -> str:
    return subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
