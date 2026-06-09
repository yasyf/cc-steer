from __future__ import annotations

import http.client
import threading
from contextlib import closing

from cc_pushback.serve import build_server


def test_build_server_serves_page_to_any_get() -> None:
    server = build_server(b"<html>hi</html>", "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with closing(http.client.HTTPConnection("127.0.0.1", server.server_address[1])) as conn:
            conn.request("GET", "/anything")
            response = conn.getresponse()
            body = response.read()
        assert response.status == 200
        assert body == b"<html>hi</html>"
        assert response.getheader("Content-Type") == "text/html; charset=utf-8"
        assert response.getheader("Content-Length") == str(len(body))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
