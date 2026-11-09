import shutil
import subprocess

import pytest

from camera_chess.videos import overlay_moves
from camera_chess.videos.ffmpeg import read_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """2 seconds of red on the left half and blue on the right, 320x240"""
    path = tmp_path_factory.mktemp("video") / "video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=160x240:d=2:r=25",
            "-f", "lavfi", "-i", "color=blue:s=160x240:d=2:r=25",
            "-filter_complex", "[0][1]hstack", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def test_overlay_frames_crop_and_scale(video):
    frames = list(overlay_moves.read_frames(str(video), (80, 0, 160, 240)))
    assert abs(len(frames) - 2 * overlay_moves.FPS) <= 1
    t, frame = frames[1]
    assert t == pytest.approx(1 / overlay_moves.FPS)
    assert frame.shape == (overlay_moves.SIZE, overlay_moves.SIZE, 3)
    left, right = frame[:, :100].mean((0, 1)), frame[:, -100:].mean((0, 1))
    assert left[0] > 200 > left[2] and right[2] > 200 > right[0]


def test_rgba_frames_with_an_odd_exact_crop(video):
    # An odd offset and size, which only an exact crop keeps
    frames = list(read_frames(video, (151, 3, 17, 33), 16, 4, pix_fmt="rgba", scale="area", exact=True))
    assert abs(len(frames) - 8) <= 1
    _, frame = frames[0]
    assert frame.shape == (16, 16, 4) and (frame[..., 3] == 255).all()
    # 9 of the 17 columns are blue
    assert frame[:, -6:, 2].mean() > 200 and frame[:, :6, 0].mean() > 200


def test_stopping_early_ends_ffmpeg(video):
    frames = read_frames(video, (0, 0, 320, 240), 8, 25, skip_nonref=False)
    next(frames)
    frames.close()
