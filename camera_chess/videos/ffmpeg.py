"""Reading a video's frames with ffmpeg, cropped to a region and scaled to a square."""

import subprocess

import numpy as np

CHANNELS = {"rgb24": 3, "rgba": 4}


def read_frames(path, roi, size, fps, *, pix_fmt="rgb24", scale="bilinear", exact=False, skip_nonref=True):
    """Yields (time, size x size x channels uint8) of the [x, y, width, height] region at fps.

    skip_nonref skips decoding B-frames, which no other frame needs: decoding is most of ffmpeg's work and over half
    the frames are B-frames, and the fps filter takes the nearest decoded frame instead, at most a frame or two away.
    exact crops the exact region, as on YUV 4:2:0 ffmpeg otherwise rounds the crop down to even sizes and offsets."""
    x, y, w, h = roi
    crop = f"crop={w}:{h}:{x}:{y}" + (":exact=1" if exact else "")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        *(["-skip_frame", "noref"] if skip_nonref else []),
        "-i",
        str(path),
        "-vf",
        f"fps={fps},{crop},scale={size}:{size}:flags={scale}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-",
    ]
    channels = CHANNELS[pix_fmt]
    frame_bytes = size * size * channels
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert process.stdout is not None
    n = 0
    try:
        while len(raw := process.stdout.read(frame_bytes)) == frame_bytes:
            yield n / fps, np.frombuffer(raw, np.uint8).reshape(size, size, channels)
            n += 1
    finally:
        process.stdout.close()
        process.wait()
