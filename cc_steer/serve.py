"""Serve an ASGI app over a transient HTTP server, loopback by default.

Binds ``127.0.0.1`` unless a host is given. Any non-loopback bind (for example
``0.0.0.0`` for Tailscale) is gated: a per-run bearer token is minted at startup
and required on every request from a non-loopback client, while loopback clients
stay exempt and frictionless.
"""

from __future__ import annotations

import secrets
import socket
import webbrowser
from dataclasses import dataclass
from ipaddress import ip_address
from typing import TYPE_CHECKING

import click
import uvicorn
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response
    from starlette.middleware.base import RequestResponseEndpoint

COOKIE = "cc_steer_token"


@dataclass(frozen=True, slots=True)
class Allow:
    set_cookie: bool


@dataclass(frozen=True, slots=True)
class Deny: ...


def is_loopback(host: str) -> bool:
    return ip_address(host).is_loopback


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]


def authorize(request: Request, token: str) -> Allow | Deny:
    if (client := request.client) and is_loopback(client.host):
        return Allow(set_cookie=False)
    secret = token.encode()
    header = request.headers.get("authorization", "")
    bearer = header.removeprefix("Bearer ") if header.startswith("Bearer ") else None
    presented = [value for value in (bearer, request.cookies.get(COOKIE)) if value is not None]
    if any(secrets.compare_digest(value.encode(), secret) for value in presented):
        return Allow(set_cookie=False)
    if (query := request.query_params.get("token")) is not None and secrets.compare_digest(query.encode(), secret):
        return Allow(set_cookie=True)
    return Deny()


def guard_app(app: FastAPI, token: str) -> None:
    """Gates ``app`` behind a per-run bearer token, exempting loopback clients.

    A request from a loopback client passes untouched. From any other client the
    token must arrive as ``Authorization: Bearer <token>``, a ``?token=<token>``
    query parameter (which mints a cookie so later requests carry it), or that
    cookie; anything else gets a 403. Constant-time comparison guards the token.

    Args:
        app: The application to wrap; mutated in place.
        token: The secret required of non-loopback clients.
    """

    @app.middleware("http")
    async def gate(request: Request, call_next: RequestResponseEndpoint) -> Response:
        match authorize(request, token):
            case Deny():
                return JSONResponse({"detail": "token required"}, status_code=403)
            case Allow(set_cookie=set_cookie):
                response = await call_next(request)
                if set_cookie:
                    response.set_cookie(COOKIE, token, httponly=True, samesite="strict")
                return response


async def serve(app: FastAPI, *, host: str, port: int, open_browser: bool) -> None:
    """Serves ``app`` until interrupted, printing its URLs.

    Binds ``host``; a non-loopback host mints a per-run token via :func:`guard_app`
    and prints the tokened remote URL alongside the frictionless loopback one.

    Args:
        app: The ASGI application to serve.
        host: The interface to bind; ``0.0.0.0`` exposes it over the LAN/Tailscale.
        port: The port to bind; ``0`` lets the OS pick a free one.
        open_browser: Whether to open the loopback URL in a browser once serving.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    local = f"http://127.0.0.1:{sock.getsockname()[1]}/"
    match None if is_loopback(host) else secrets.token_urlsafe(32):
        case None:
            click.echo(f"serving on {local}  (Ctrl-C to stop)")
        case token:
            guard_app(app, token)
            remote = f"http://{lan_ip()}:{sock.getsockname()[1]}/?token={token}"
            click.echo(f"serving on {local}  ·  {remote}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(local)
    await uvicorn.Server(uvicorn.Config(app, log_level="warning")).serve(sockets=[sock])
    click.echo("\nstopped")
