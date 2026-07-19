from __future__ import annotations

import types
from typing import TYPE_CHECKING

import httpx
import pytest

from cc_steer.dashboard import build_app
from cc_steer.report import Sample, Summary, corpus_stats
from cc_steer.serve import COOKIE, Allow, Deny, authorize, guard_app, is_loopback

if TYPE_CHECKING:
    from cc_steer.store import FeedbackStore

pytestmark = pytest.mark.anyio

TOKEN = "run-secret-9d2f"
LOOPBACK = ("127.0.0.1", 40001)
REMOTE = ("100.64.0.7", 51001)


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        pytest.param("127.0.0.1", True, id="ipv4-loopback"),
        pytest.param("::1", True, id="ipv6-loopback"),
        pytest.param("0.0.0.0", False, id="all-interfaces"),
        pytest.param("100.64.0.1", False, id="tailscale"),
        pytest.param("192.168.1.10", False, id="lan"),
    ],
)
def test_is_loopback(host: str, loopback: bool) -> None:
    assert is_loopback(host) is loopback


async def guarded(store: FeedbackStore, *, addr: tuple[str, int]) -> httpx.AsyncClient:
    summary = Summary(
        stats=corpus_stats([Sample.from_row(row) for row in await store.candidates()]),
        highlights=(),
        narrative="t",
    )
    app = await build_app(store, summary=summary)
    guard_app(app, TOKEN)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=addr), base_url="http://dash")


async def test_loopback_client_needs_no_token(store: FeedbackStore) -> None:
    async with await guarded(store, addr=LOOPBACK) as http:
        resp = await http.get("/api/candidates")
    assert resp.status_code == 200
    assert resp.json() == {"candidates": []}


async def test_loopback_client_reaches_static_assets(store: FeedbackStore) -> None:
    async with await guarded(store, addr=LOOPBACK) as http:
        resp = await http.get("/static/base.css")
    assert resp.status_code == 200


async def test_remote_client_without_token_is_forbidden(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        resp = await http.get("/api/candidates")
    assert resp.status_code == 403
    assert resp.json() == {"detail": "token required"}


async def test_remote_static_asset_is_forbidden_without_token(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        resp = await http.get("/static/base.css")
    assert resp.status_code == 403


async def test_remote_bearer_token_passes_through_to_the_real_api(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        resp = await http.get("/api/candidates", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert resp.json() == {"candidates": []}


async def test_remote_wrong_bearer_token_is_forbidden(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        resp = await http.get("/api/candidates", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 403


async def test_remote_query_token_mints_a_cookie_carrying_later_requests(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        first = await http.get("/api/candidates", params={"token": TOKEN})
        second = await http.get("/api/candidates")
    assert first.status_code == 200
    assert first.cookies.get(COOKIE) == TOKEN
    assert "httponly" in first.headers["set-cookie"].lower()
    assert second.status_code == 200


async def test_remote_cookie_token_passes(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        http.cookies.set(COOKIE, TOKEN)
        resp = await http.get("/api/candidates")
    assert resp.status_code == 200


async def test_remote_wrong_cookie_is_forbidden(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        http.cookies.set(COOKIE, "wrong")
        resp = await http.get("/api/candidates")
    assert resp.status_code == 403


async def test_remote_valid_cookie_survives_a_stale_bearer_header(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        http.cookies.set(COOKIE, TOKEN)
        resp = await http.get("/api/candidates", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 200


async def test_remote_non_ascii_query_token_is_forbidden_not_errored(store: FeedbackStore) -> None:
    async with await guarded(store, addr=REMOTE) as http:
        resp = await http.get("/api/candidates", params={"token": "café"})
    assert resp.status_code == 403


def fake_request(
    *,
    host: str = REMOTE[0],
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    query: dict[str, str] | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host),
        headers=headers or {},
        cookies=cookies or {},
        query_params=query or {},
    )


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(fake_request(headers={"authorization": "Bearer café"}), id="non-ascii-bearer"),
        pytest.param(fake_request(cookies={COOKIE: "café"}), id="non-ascii-cookie"),
        pytest.param(fake_request(query={"token": "café"}), id="non-ascii-query"),
    ],
)
def test_authorize_denies_non_ascii_credentials_without_raising(request_: types.SimpleNamespace) -> None:
    assert isinstance(authorize(request_, TOKEN), Deny)


def test_authorize_allows_a_valid_cookie_despite_a_wrong_bearer_header() -> None:
    request_ = fake_request(headers={"authorization": "Bearer wrong"}, cookies={COOKIE: TOKEN})
    assert authorize(request_, TOKEN) == Allow(set_cookie=False)


def test_authorize_treats_a_clientless_request_as_remote() -> None:
    anonymous = fake_request()
    anonymous.client = None
    assert isinstance(authorize(anonymous, TOKEN), Deny)
    bearing = fake_request(headers={"authorization": f"Bearer {TOKEN}"})
    bearing.client = None
    assert authorize(bearing, TOKEN) == Allow(set_cookie=False)
