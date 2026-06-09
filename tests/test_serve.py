from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cc_pushback.serve import build_app


@pytest.mark.anyio
async def test_build_app_serves_page_to_any_get() -> None:
    page = b"<html>hi</html>"
    async with TestClient(TestServer(build_app(page))) as client:
        resp = await client.get("/anything")
        assert resp.status == 200
        assert await resp.read() == page
        assert resp.headers["Content-Type"] == "text/html; charset=utf-8"
        root = await client.get("/")
        assert await root.read() == page
