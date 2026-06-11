"""Serve an ASGI app over a transient HTTP server bound to all interfaces."""

from __future__ import annotations

import socket
import webbrowser
from typing import TYPE_CHECKING

import click
import uvicorn

if TYPE_CHECKING:
    from fastapi import FastAPI

BIND_HOST = "0.0.0.0"


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


async def serve(app: FastAPI, *, port: int, open_browser: bool) -> None:
    """Serves ``app`` on all interfaces until interrupted, printing its URLs.

    Binds ``0.0.0.0`` so the dashboard is reachable from other hosts (for example
    over Tailscale), and prints both the loopback and LAN/Tailscale-facing URLs.

    Args:
        app: The ASGI application to serve.
        port: The port to bind; ``0`` lets the OS pick a free one.
        open_browser: Whether to open the loopback URL in a browser once serving.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((BIND_HOST, port))
    bound = sock.getsockname()[1]
    local = f"http://127.0.0.1:{bound}/"
    click.echo(f"serving on {local}  ·  http://{lan_ip()}:{bound}/  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(local)
    await uvicorn.Server(uvicorn.Config(app, log_level="warning")).serve(sockets=[sock])
    click.echo("\nstopped")
