"""Live steering: the mode config, the delivery mailbox, the kill switch, and the span scrubber.

Shadow mode records proposals and never touches a session (:mod:`cc_steer.watcher.delivery`).
This module is the step up: a :class:`LiveConfig` read from ``~/.cc-steer/live.toml`` names the
mode (``shadow`` | ``mirror`` | ``live_allow`` | ``live_all``), a :class:`MailboxDelivery` queues
every fired proposal into a ``deliveries`` table beside the shadow ledger, and the
``UserPromptSubmit`` hook (:mod:`cc_steer.livehook`) pops the freshest unexpired steer and either
records the would-be delivery (mirror) or surfaces it (live). A missing config means shadow — the
current world — and an invalid one crashes the daemon loud while the fail-open hook treats it as a
kill. Two kill switches (``~/.cc-steer/live.off`` and ``$CC_STEER_LIVE_OFF``) stop delivery in both
the daemon and the hook.

The steer is surfaced inside a ``<cc-steer-proposal>`` span so the model sees it, and
:func:`scrub_events` strips that span from transcript text before mining, so the watcher never
learns from its own suggestion and the whole user turn survives the junk filter's whole-turn drop.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import aiosqlite
from cc_transcript.mining.store import now
from cc_transcript.models import UserEvent

from cc_steer.watcher.delivery import SHADOW_DDL

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from cc_transcript.models import TranscriptEvent

    from cc_steer.watcher.delivery import SteerDelivery
    from cc_steer.watcher.types import SteerProposal

type LiveMode = Literal["shadow", "mirror", "live_allow", "live_all"]

LIVE_MODES: frozenset[str] = frozenset({"shadow", "mirror", "live_allow", "live_all"})
LIVE_MODE_VALUES: tuple[LiveMode, ...] = ("shadow", "mirror", "live_allow", "live_all")

LIVE_CONFIG_ENV = "CC_STEER_LIVE_CONFIG"
LIVE_OFF_ENV = "CC_STEER_LIVE_OFF"
SHADOW_DB_ENV = "CC_STEER_SHADOW_DB"

PROPOSAL_TAG = "cc-steer-proposal"
PROPOSAL_SPAN_RE = re.compile(rf"<{PROPOSAL_TAG}\b[^>]*>.*?</{PROPOSAL_TAG}>\s*", re.DOTALL)

DEFAULT_TTL_MINUTES = 60

State = Literal["queued", "delivered", "holdout", "mirror", "expired", "suppressed_budget"]

DELIVERIES_DDL = """
CREATE TABLE IF NOT EXISTS deliveries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  project TEXT,
  ts TEXT NOT NULL,
  mode TEXT NOT NULL,
  ttl TEXT NOT NULL,
  holdout INTEGER NOT NULL,
  state TEXT NOT NULL,
  decided_at TEXT,
  UNIQUE(proposal_id)
);
"""

INSERT_DELIVERY = """
INSERT OR IGNORE INTO deliveries (proposal_id, session_id, project, ts, mode, ttl, holdout, state)
VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')
"""


def cc_steer_dir() -> Path:
    return Path.home() / ".cc-steer"


def live_config_path() -> Path:
    """The live-mode config path: ``$CC_STEER_LIVE_CONFIG`` or ``~/.cc-steer/live.toml``."""
    return Path(env) if (env := os.environ.get(LIVE_CONFIG_ENV)) else cc_steer_dir() / "live.toml"


def live_off_path() -> Path:
    """The kill-switch flag path, ``~/.cc-steer/live.off``."""
    return cc_steer_dir() / "live.off"


def shadow_db_path() -> Path:
    """The ledger path shared by the shadow proposals and the delivery mailbox."""
    from cc_steer.watcher.delivery import ShadowDelivery

    return Path(env) if (env := os.environ.get(SHADOW_DB_ENV)) else ShadowDelivery.default_path()


def is_killed() -> bool:
    """Whether live delivery is switched off, by env var or the flag file."""
    return bool(os.environ.get(LIVE_OFF_ENV)) or live_off_path().exists()


@dataclass(frozen=True, slots=True)
class LiveConfig:
    """The live-steering policy read from ``~/.cc-steer/live.toml``.

    A missing file is :meth:`shadow` — the current world, nothing delivered. An invalid file raises,
    so the daemon crashes loud; the hook catches that and treats it as a kill (fail-open).

    Attributes:
        mode: ``shadow`` records nothing new; ``mirror`` queues would-be deliveries without emitting;
            ``live_allow`` emits only in ``allow_projects``; ``live_all`` emits everywhere.
        allow_projects: The project directories that receive live steers under ``live_allow``.
        cooldown_turns: Turns a session cools down after a proposal (the cascade knob, surfaced).
        max_per_session: Proposals a session may accumulate before it stops being evaluated.
        max_live_per_day: The machine-global ceiling on emitted live steers per day.
        steer_ttl_minutes: How long a queued steer stays deliverable before it expires.
        holdout_frac: The deterministic fraction of would-be live deliveries held out unemitted.
    """

    mode: LiveMode = "shadow"
    allow_projects: tuple[str, ...] = ()
    cooldown_turns: int = 5
    max_per_session: int = 5
    max_live_per_day: int = 20
    steer_ttl_minutes: int = DEFAULT_TTL_MINUTES
    holdout_frac: float = 0.5

    @classmethod
    def shadow(cls) -> Self:
        """The default policy when no config exists: shadow mode, nothing delivered."""
        return cls(mode="shadow")

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Reads the policy from ``path`` (default :func:`live_config_path`); missing means shadow.

        Raises:
            ValueError: The file is malformed or names an unknown mode.
        """
        target = path or live_config_path()
        if not target.exists():
            return cls.shadow()
        data = tomllib.loads(target.read_text())
        if (mode := data.get("mode", "shadow")) not in LIVE_MODES:
            raise ValueError(f"unknown live mode {mode!r}; expected one of {sorted(LIVE_MODES)}")
        return cls(
            mode=mode,
            allow_projects=tuple(os.path.realpath(project) for project in data.get("allow_projects", ())),
            cooldown_turns=int(data.get("cooldown_turns", 5)),
            max_per_session=int(data.get("max_per_session", 5)),
            max_live_per_day=int(data.get("max_live_per_day", 20)),
            steer_ttl_minutes=int(data.get("steer_ttl_minutes", DEFAULT_TTL_MINUTES)),
            holdout_frac=float(data.get("holdout_frac", 0.5)),
        )

    def allows(self, project: str) -> bool:
        """Whether ``project`` is covered by ``allow_projects`` (exact dir or a descendant)."""
        return any(project == allowed or project.startswith(f"{allowed.rstrip('/')}/") for allowed in self.allow_projects)

    def to_toml(self) -> str:
        """The config as ``live.toml`` text — the round-trip inverse of :meth:`load`."""
        projects = ", ".join(f'"{project}"' for project in self.allow_projects)
        return (
            f'mode = "{self.mode}"\n'
            f"allow_projects = [{projects}]\n"
            f"cooldown_turns = {self.cooldown_turns}\n"
            f"max_per_session = {self.max_per_session}\n"
            f"max_live_per_day = {self.max_live_per_day}\n"
            f"steer_ttl_minutes = {self.steer_ttl_minutes}\n"
            f"holdout_frac = {self.holdout_frac}\n"
        )

    def write(self, path: Path | None = None) -> Path:
        """Writes the config to ``path`` (default :func:`live_config_path`); returns where it landed."""
        (target := path or live_config_path()).parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_toml())
        return target


def holdout(proposal_id: int, frac: float) -> bool:
    """Whether a proposal is held out, deterministically from its id — same verdict every pop, no RNG."""
    return int(hashlib.sha256(str(proposal_id).encode()).hexdigest(), 16) % 10_000 < frac * 10_000


def expiry(ts: str, minutes: int) -> str:
    return (datetime.fromisoformat(ts) + timedelta(minutes=minutes)).isoformat()


def is_expired(ttl: str, *, at: datetime | None = None) -> bool:
    return (at or datetime.now(UTC)) >= datetime.fromisoformat(ttl)


def today_prefix(at: datetime | None = None) -> str:
    return (at or datetime.now(UTC)).date().isoformat()


def format_additional_context(proposal_id: int, steer: str) -> str:
    """The ``UserPromptSubmit`` additionalContext: the steer in a scannable span plus the surfaced-UX instruction."""
    return (
        f"<{PROPOSAL_TAG} id={proposal_id}>\n{steer}\n</{PROPOSAL_TAG}>\n\n"
        f"A background watcher trained on this user's own steering suggests the message above for the current "
        f"moment. Begin your reply with the line `watcher suggests: <one-line paraphrase> (proposal {proposal_id})`, "
        f"then follow that steer — unless the user's message overrides it, in which case do what they asked and "
        f"note that you set the suggestion aside."
    )


def scrub_text(text: str) -> str:
    """Strips every ``<cc-steer-proposal>`` span, leaving the user's authored text."""
    return PROPOSAL_SPAN_RE.sub("", text)


def scrub_event(event: TranscriptEvent) -> TranscriptEvent:
    """Returns ``event`` with the injected steer span stripped, when it is a user turn carrying one."""
    if isinstance(event, UserEvent) and (scrubbed := scrub_text(event.text)) != event.text:
        return dataclasses.replace(event, text=scrubbed)
    return event


def scrub_events(events: Sequence[TranscriptEvent]) -> list[TranscriptEvent]:
    """Returns ``events`` with the injected steer span removed from any user turn that carries one.

    Runs before mining so a surfaced steer never trains the watcher on its own suggestion, and so the
    junk filter's whole-turn drop never junks the user's authored reply along with the span.
    """
    return [scrub_event(event) for event in events]


class MailboxDelivery:
    """Queues every fired proposal into the ``deliveries`` mailbox beside the shadow ledger.

    The daemon writes ``queued`` rows here; the ``UserPromptSubmit`` hook pops the freshest unexpired
    one and resolves it. The kill switch stops the daemon from queuing (delivery off), and idempotency
    by ``proposal_id`` means a replayed proposal never double-queues.

    Example:
        >>> async with await MailboxDelivery.open(config=LiveConfig.load()) as mailbox:
        ...     await mailbox.deliver(proposal)
    """

    def __init__(self, conn: aiosqlite.Connection, config: LiveConfig) -> None:
        self.conn = conn
        self.config = config

    @classmethod
    async def open(cls, path: Path | None = None, *, config: LiveConfig) -> Self:
        """Opens (creating if needed) the shared ledger and ensures both tables exist."""
        target = path or shadow_db_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(target), isolation_level=None)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=2000")
        await conn.executescript(SHADOW_DDL + DELIVERIES_DDL)
        return cls(conn, config)

    async def close(self) -> None:
        await self.conn.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def deliver(self, proposal: SteerProposal) -> None:
        """Queues one proposal as a would-be delivery; a no-op when delivery is killed or the steer is empty."""
        if is_killed() or proposal.steer is None:
            return
        cur = await self.conn.execute(
            "SELECT id FROM proposals WHERE session_id = ? AND anchor_uuid = ?",
            (proposal.session_id, proposal.anchor_uuid),
        )
        if (row := await cur.fetchone()) is None:
            return
        proposal_id = int(row["id"])
        await self.conn.execute(
            INSERT_DELIVERY,
            (
                proposal_id,
                proposal.session_id,
                proposal.project,
                proposal.ts,
                self.config.mode,
                expiry(proposal.ts, self.config.steer_ttl_minutes),
                int(holdout(proposal_id, self.config.holdout_frac)),
            ),
        )

    async def recent(self, limit: int = 20) -> list[dict[str, object]]:
        """The most recent deliveries joined to their proposal's steer text, newest first — the inbox surface."""
        cur = await self.conn.execute(
            "SELECT d.*, p.steer, p.turn_index FROM deliveries d JOIN proposals p ON p.id = d.proposal_id "
            "ORDER BY d.id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) async for row in cur]

    async def counts(self, *, at: datetime | None = None) -> dict[str, int]:
        """Delivery counts per terminal state plus today's emitted total — the ``live status`` numbers."""
        by_state = {
            str(row["state"]): int(row["n"])
            async for row in await self.conn.execute("SELECT state, COUNT(*) AS n FROM deliveries GROUP BY state")
        }
        [delivered_today] = [
            int(row["n"])
            async for row in await self.conn.execute(
                "SELECT COUNT(*) AS n FROM deliveries WHERE state = 'delivered' AND substr(decided_at, 1, 10) = ?",
                (today_prefix(at),),
            )
        ]
        return by_state | {"delivered_today": delivered_today}


class TeeDelivery:
    """Fans one proposal out to several deliveries in order — shadow first, then the mailbox."""

    def __init__(self, deliveries: Sequence[SteerDelivery]) -> None:
        self.deliveries = tuple(deliveries)

    async def deliver(self, proposal: SteerProposal) -> None:
        for delivery in self.deliveries:
            await delivery.deliver(proposal)
