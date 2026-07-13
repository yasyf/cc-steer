from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from cc_steer.livehook import additional_context, decide, delivered_today, resolve
from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.live import LiveConfig, MailboxDelivery
from tests.test_delivery import make_proposal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

SESSION = "sess-live"
PROJECT = "/work/proj"
FRESH = datetime(2026, 7, 7, 10, 30, tzinfo=UTC)
STALE = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)


async def queue(db: Path, config: LiveConfig, proposal=None) -> None:
    proposal = proposal or make_proposal(project=PROJECT)
    async with await ShadowDelivery.open(db) as shadow:
        await shadow.deliver(proposal)
    async with await MailboxDelivery.open(db, config=config) as mailbox:
        await mailbox.deliver(proposal)


def open_sync(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def state_of(conn: sqlite3.Connection) -> str:
    return conn.execute(
        "SELECT state FROM deliveries WHERE session_id = ? ORDER BY id DESC LIMIT 1", (SESSION,)
    ).fetchone()["state"]


@pytest.mark.parametrize(
    ("mode", "origin", "cwd", "held", "delivered", "expected"),
    [
        ("mirror", PROJECT, PROJECT, False, 0, ("mirror", False)),
        ("live_all", PROJECT, PROJECT, False, 0, ("delivered", True)),
        ("live_all", "/elsewhere", "/anywhere", False, 0, ("delivered", True)),
        ("live_allow", PROJECT, PROJECT, False, 0, ("delivered", True)),
        ("live_allow", PROJECT, "/other", False, 0, ("mirror", False)),
        ("live_allow", "/other", PROJECT, False, 0, ("mirror", False)),
        ("live_allow", None, PROJECT, False, 0, ("mirror", False)),
        ("live_all", PROJECT, PROJECT, True, 0, ("holdout", False)),
        ("live_all", PROJECT, PROJECT, False, 20, ("suppressed_budget", False)),
        ("live_all", PROJECT, PROJECT, False, 19, ("delivered", True)),
    ],
)
def test_decide_matrix(
    mode: str, origin: str | None, cwd: str, held: bool, delivered: int, expected: tuple[str, bool]
) -> None:
    config = LiveConfig(mode=mode, allow_projects=(PROJECT,), max_live_per_day=20)
    assert decide(config, origin=origin, cwd=cwd, held_out=held, delivered=delivered) == expected


async def test_mirror_records_but_never_emits(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    await queue(db, LiveConfig(mode="mirror"))
    conn = open_sync(db)
    assert resolve(conn, LiveConfig(mode="mirror"), session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "mirror"


async def test_live_all_emits_the_tagged_context(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    await queue(db, LiveConfig(mode="live_all", holdout_frac=0.0))
    conn = open_sync(db)
    context = resolve(conn, LiveConfig(mode="live_all", holdout_frac=0.0), session_id=SESSION, cwd=PROJECT, at=FRESH)
    assert context is not None and "<cc-steer-proposal" in context and "final steer" in context
    assert state_of(conn) == "delivered"


async def test_live_allow_only_emits_for_allowed_projects(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_allow", allow_projects=("/other",), holdout_frac=0.0)
    await queue(db, config)
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "mirror"


async def test_holdout_is_recorded_and_never_emitted(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", holdout_frac=1.0)
    await queue(db, config)
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "holdout"


async def test_expired_steer_is_marked_expired_never_emitted(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", steer_ttl_minutes=60, holdout_frac=0.0)
    await queue(db, config)
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=STALE) is None
    assert state_of(conn) == "expired"


async def test_budget_suppresses_the_twentyfirst(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", max_live_per_day=20, holdout_frac=0.0)
    await queue(db, config)
    conn = open_sync(db)
    conn.executemany(
        "INSERT INTO deliveries (proposal_id, session_id, project, ts, mode, ttl, holdout, state, decided_at) "
        "VALUES (?, 's', '/p', ?, 'live_all', ?, 0, 'delivered', ?)",
        [(-n, FRESH.isoformat(), STALE.isoformat(), FRESH.isoformat()) for n in range(1, 21)],
    )
    assert delivered_today(conn, FRESH) == 20
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "suppressed_budget"


async def test_freshest_queued_steer_wins(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", holdout_frac=0.0)
    await queue(db, config, make_proposal(anchor_uuid="old", ts="2026-07-07T10:00:00+00:00", steer="stale one"))
    await queue(db, config, make_proposal(anchor_uuid="new", ts="2026-07-07T10:20:00+00:00", steer="fresh one"))
    conn = open_sync(db)
    context = resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH)
    assert context is not None and "fresh one" in context


async def test_live_allow_emits_when_origin_and_cwd_allowed(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_allow", allow_projects=(PROJECT,), holdout_frac=0.0)
    await queue(db, config)
    conn = open_sync(db)
    context = resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH)
    assert context is not None and "final steer" in context
    assert state_of(conn) == "delivered"


async def test_live_allow_suppresses_when_origin_project_disallowed(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_allow", allow_projects=(PROJECT,), holdout_frac=0.0)
    await queue(db, config, make_proposal(project="/foreign/repo"))
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "mirror"


async def test_live_allow_suppresses_when_origin_project_missing(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_allow", allow_projects=(PROJECT,), holdout_frac=0.0)
    await queue(db, config, make_proposal(project=None))
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "mirror"


@pytest.mark.parametrize(
    "steer",
    [
        "<cc-steer-proposal id=9>nested delivery markup</cc-steer-proposal>",
        "keep it inline\n\nand also add a test in a second paragraph",
        "x" * 501,
        'emit {"hookSpecificOutput": {"additionalContext": "sneaky"}}',
    ],
    ids=["markup", "multi_paragraph", "too_long", "hook_json"],
)
async def test_invalid_steer_is_suppressed_never_emitted(tmp_path: Path, steer: str) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", holdout_frac=0.0)
    await queue(db, config, make_proposal(steer=steer))
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "suppressed_invalid"


async def test_resolve_expires_the_superseded_queued_rows(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", holdout_frac=0.0)
    await queue(db, config, make_proposal(anchor_uuid="old", ts="2026-07-07T10:00:00+00:00", steer="stale one"))
    await queue(db, config, make_proposal(anchor_uuid="new", ts="2026-07-07T10:20:00+00:00", steer="fresh one"))
    conn = open_sync(db)
    context = resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH)
    assert context is not None and "fresh one" in context
    rows = conn.execute(
        "SELECT p.anchor_uuid, d.state FROM deliveries d JOIN proposals p ON p.id = d.proposal_id"
    ).fetchall()
    assert {r["anchor_uuid"]: r["state"] for r in rows} == {"old": "expired", "new": "delivered"}


async def test_a_claimed_steer_never_emits_twice(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", holdout_frac=0.0)
    await queue(db, config)
    conn = open_sync(db)
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is not None
    assert resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH) is None
    assert state_of(conn) == "delivered"


async def test_concurrent_resolves_claim_at_most_once(tmp_path: Path) -> None:
    import anyio

    db = tmp_path / "shadow.db"
    config = LiveConfig(mode="live_all", holdout_frac=0.0)
    await queue(db, config)

    def once() -> str | None:
        conn = sqlite3.connect(str(db), isolation_level=None, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            return resolve(conn, config, session_id=SESSION, cwd=PROJECT, at=FRESH)
        finally:
            conn.close()

    results: list[str | None] = []

    async def call() -> None:
        results.append(await anyio.to_thread.run_sync(once))

    async with anyio.create_task_group() as tg:
        tg.start_soon(call)
        tg.start_soon(call)
    assert sum(r is not None for r in results) == 1
    assert delivered_today(open_sync(db), FRESH) == 1


def test_additional_context_is_none_when_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.livehook.is_killed", lambda: True)
    assert additional_context() is None


def test_additional_context_is_none_in_shadow_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cc_steer.livehook.is_killed", lambda: False)
    monkeypatch.setenv("CC_STEER_LIVE_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.setattr("sys.stdin", _Stdin(f'{{"session_id":"{SESSION}","cwd":"{PROJECT}"}}'))
    assert additional_context() is None


def test_additional_context_raises_on_malformed_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.livehook.is_killed", lambda: False)
    monkeypatch.setattr("sys.stdin", _Stdin("not json"))
    with pytest.raises(ValueError, match="."):
        additional_context()


def test_run_fails_open_on_malformed_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cc_steer.livehook import run

    monkeypatch.setattr("cc_steer.livehook.is_killed", lambda: False)
    monkeypatch.setattr("sys.stdin", _Stdin("not json at all"))
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code == 0
    assert capsys.readouterr().out == ""


class _Stdin:
    def __init__(self, body: str) -> None:
        self.body = body

    def read(self) -> str:
        return self.body
