"""Serve a rendered page from memory over a transient HTTP server."""

from __future__ import annotations

import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import click

BIND_HOST = "0.0.0.0"


def build_server(page: bytes, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def serve(page: bytes, *, port: int, open_browser: bool) -> None:
    """Serves ``page`` on all interfaces until interrupted, printing its URLs.

    Binds ``0.0.0.0`` so the page is reachable from other hosts (for example over
    Tailscale), and prints both the loopback and LAN/Tailscale-facing URLs.

    Args:
        page: The HTML document to serve on every request.
        port: The port to bind; ``0`` lets the OS pick a free one.
        open_browser: Whether to open the loopback URL in a browser once serving.
    """
    server = build_server(page, BIND_HOST, port)
    bound = server.server_address[1]
    local = f"http://127.0.0.1:{bound}/"
    click.echo(f"serving on {local}  ·  http://{lan_ip()}:{bound}/  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(local)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nstopped")
    finally:
        server.server_close()
