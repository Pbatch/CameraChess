import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from camera_chess.local_server import LocalHandler, save_json


@pytest.fixture
def server(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(bytes(range(256)) * 4)
    saved = tmp_path / "labels.json"

    class Handler(LocalHandler):
        def do_GET(self):
            if self.route == "/video":
                self.send_video(video)
            elif self.route.startswith("/img/"):
                self.send_file(tmp_path / Path(self.route).name, "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            save_json(saved, self.read_json())
            self.send_json({"ok": True})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", saved
    httpd.shutdown()
    httpd.server_close()


def get(url, **headers):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.status, dict(response.headers), response.read()


def test_video_byte_ranges(server):
    url, _ = server
    status, headers, body = get(f"{url}/video")
    assert status == 200 and len(body) == 1024 and headers["Accept-Ranges"] == "bytes"
    status, headers, body = get(f"{url}/video", Range="bytes=10-19")
    assert status == 206 and body == bytes(range(10, 20)) and headers["Content-Range"] == "bytes 10-19/1024"
    _, headers, body = get(f"{url}/video", Range="bytes=1000-")
    assert len(body) == 24 and headers["Content-Range"] == "bytes 1000-1023/1024"
    _, _, body = get(f"{url}/video", Range="bytes=-4")
    assert body == bytes(range(252, 256))


def test_missing_files_are_404_and_others_are_not_served(server, tmp_path):
    url, _ = server
    (tmp_path / "board.jpg").write_bytes(b"jpeg")
    assert get(f"{url}/img/board.jpg")[2] == b"jpeg"
    for path in ("/img/missing.jpg", "/labels.json", "/img/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd"):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(url + path)
        assert error.value.code == 404


def test_posted_json_is_saved(server):
    url, saved = server
    request = urllib.request.Request(f"{url}/labels", data=json.dumps({"b": 1, "a": [2]}).encode(), method="POST")
    with urllib.request.urlopen(request) as response:
        assert json.loads(response.read()) == {"ok": True}
    assert json.loads(saved.read_text()) == {"a": [2], "b": 1}
    assert not saved.with_suffix(".tmp").exists()
