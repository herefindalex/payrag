from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from payrag.acquire import fetch_source


BODY = b"<html><body>fixture</body></html>"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("ETag", '"fixture-etag"')
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, format: str, *args: object) -> None:
        return


class AcquireTests(unittest.TestCase):
    def test_fetch_preserves_bytes_and_records_hash(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                raw_dir = Path(directory)
                result = fetch_source(
                    "D99",
                    f"http://127.0.0.1:{server.server_port}/fixture",
                    raw_dir,
                    timeout_seconds=2,
                    attempts=1,
                )

                self.assertEqual("acquired", result.status)
                self.assertEqual(hashlib.sha256(BODY).hexdigest(), result.sha256)
                self.assertEqual(BODY, (raw_dir / "D99.html").read_bytes())
                self.assertEqual('"fixture-etag"', result.etag)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
