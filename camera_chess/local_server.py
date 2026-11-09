"""What the local labelling tools' servers share: a quiet request handler with helpers for sending pages, JSON, images
and seekable videos, and saving what the page posts."""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from loguru import logger


class LocalHandler(BaseHTTPRequestHandler):
    """Serves only what a subclass's do_GET and do_POST send, rather than files from a directory"""

    @property
    def route(self) -> str:
        """The request's path, unquoted and without its query"""
        return unquote(self.path.split("?")[0])

    def send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status: int = 200) -> None:
        self.send(json.dumps(data).encode(), status=status)

    def send_page(self, path: Path) -> None:
        self.send(path.read_bytes(), "text/html; charset=utf-8")

    def send_file(self, path: Path, content_type: str) -> None:
        if path.is_file():
            self.send(path.read_bytes(), content_type)
        else:
            self.send_error(404)

    def send_video(self, path: Path) -> None:
        """The video, or the byte range asked for, as browsers only seek in a video served with byte ranges"""
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:
                start = size - int(match.group(2))
        end = min(end, size - 1)
        self.send_response(206 if match else 200)
        self.send_header("Content-Type", "video/mp4" if path.suffix == ".mp4" else "video/webm")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def read_json(self):
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def log_message(self, format: str, *args) -> None:
        pass


def save_json(path: Path, data) -> None:
    """Writes the JSON through a temporary file, so a crash mid-write can't lose what was saved before"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(path)


def serve(handler, port: int, message: str, host: str = "127.0.0.1") -> None:
    server = ThreadingHTTPServer((host, port), handler)
    logger.info(f"{message} at http://{host}:{port}")
    server.serve_forever()
