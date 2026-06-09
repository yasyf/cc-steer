"""Serve a rendered page from memory over a transient async HTTP server."""

from __future__ import annotations

import socket
import webbrowser

import anyio
import click
from aiohttp import web

BIND_HOST = "0.0.0.0"


def build_app(page: bytes) -> web.Application:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=page, content_type="text/html", charset="utf-8")

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    return app


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


async def serve(page: bytes, *, port: int, open_browser: bool) -> None:
    """Serves ``page`` on all interfaces until interrupted, printing its URLs.

    Binds ``0.0.0.0`` so the page is reachable from other hosts (for example over
    Tailscale), and prints both the loopback and LAN/Tailscale-facing URLs.

    Args:
        page: The HTML document to serve on every request.
        port: The port to bind; ``0`` lets the OS pick a free one.
        open_browser: Whether to open the loopback URL in a browser once serving.
    """
    runner = web.AppRunner(build_app(page))
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((BIND_HOST, port))
    bound = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()
    local = f"http://127.0.0.1:{bound}/"
    click.echo(f"serving on {local}  ·  http://{lan_ip()}:{bound}/  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(local)
    try:
        await anyio.sleep_forever()
    finally:
        with anyio.CancelScope(shield=True):
            await runner.cleanup()
        click.echo("\nstopped")
