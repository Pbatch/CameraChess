"""Finds the 2D board drawn over a Yunker video, its `overlay_roi` [x, y, width, height], so only `roi` (the 3D
board) needs drawing by hand in camera_chess.videos.roi_labeller.

The overlay is the only thing on screen in its two flat square colours, so it is measured rather than searched for:
1. Each pixel of about 24 keyframes is +1 (light square colour), -1 (dark) or 0, and averaged.
2. Where that flips between neighbouring pixels marks grid lines. Summed down columns (and along rows), they are spikes
   at the 9 lines of an upright grid, which the 3D board's slanted, uneven lines don't make.
3. 9 evenly spaced lines are fitted to each profile, giving the square size and the edges, also when the board runs
   off the frame (lines off it score nothing).
4. The lines fit as well a square or two out, so the light/dark parity (a8 light) picks the shift.
Each stretch of the video is then scored at that box, and a stretch that disagrees is fitted alone: a board found
elsewhere means the overlay moved, and the video is reported, not given a box.

The box is the board's 8 x 8 squares. Hand-drawn boxes were a few pixels larger (often taking in the eval bar's edge
on the left); on 145 of them, the median IoU is 0.98 and the lowest 0.94, at about 0.2 s a video on 16 threads.

Usage: python -m camera_chess.videos.find_overlay [-d yunker/<id> ...] [--overwrite] [--check]
With no -d, every dataset under data/yunker without an overlay_roi. --check compares with existing boxes instead of
writing.
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from loguru import logger

from camera_chess.config import get_config
from camera_chess.videos.config import load_all, save_all, video_path

LIGHT = np.array([234, 212, 175])
DARK = np.array([179, 134, 101])
# Colour distance counted as a square colour
TOL = 30
# Square sizes tried, in pixels (boards of 224 to 384)
SQUARE_MIN, SQUARE_MAX = 28.0, 48.0
# Keyframes read, and stretches of them checked for a moved overlay
FRAMES, STRETCHES = 24, 6
# Checker score below which there is no board (titles, intros, end screens)
MIN_SCORE = 0.3


def _lut():
    """Sign of each colour quantised to 5 bits a channel, so a frame is classified by a lookup."""
    q = np.arange(32) * 8 + 4
    rgb = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1)
    light = np.linalg.norm(rgb - LIGHT, axis=-1) < TOL
    dark = np.linalg.norm(rgb - DARK, axis=-1) < TOL
    return (light.astype(np.int8) - dark.astype(np.int8)).ravel()


LUT = _lut()


def keyframes(path, n=FRAMES):
    """About n keyframes spread over the video, (n, H, W, 3) uint8. Other packets are dropped before decoding, and
    only keyframes at least duration / n apart are converted to RGB."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    w, h, duration = int(probe[0]), int(probe[1]), float(probe[2])
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{duration / n:.3f})'"
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-discard",
            "nokey",
            "-skip_frame",
            "nokey",
            "-i",
            str(path),
            "-vf",
            select,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def signs(frames):
    """(n, H, W) int8: +1 light square colour, -1 dark, 0 neither."""
    q = (frames >> 3).astype(np.uint16)
    return LUT[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]]


def flips(s, axis):
    """How strongly the colour flips light <-> dark at each pixel along axis (2 px apart, for blurred edges)."""
    a = np.moveaxis(s, axis, -1)
    d = np.zeros_like(a)
    d[..., 1:-1] = np.abs(a[..., 2:] - a[..., :-2])
    return np.moveaxis(d, -1, axis)


def _comb(p, starts, sizes):
    """Scores (sizes x starts) of 9 lines from each start, interpolating the profile p (last entry 0 = off frame)."""
    n = len(p) - 1
    lines = starts[None, :, None] + sizes[:, None, None] * np.arange(9)
    i = np.floor(lines).astype(np.int32)
    f = lines - i
    ok = (i >= 0) & (i < n - 1)
    j = np.where(ok, i + 1, n)
    i = np.where(ok, i, n)
    return ((1 - f) * p[i] + f * p[j]).sum(-1)


def comb(profile):
    """(first line, square size) of the 9 evenly spaced lines best matching the profile's spikes: every size and
    whole-pixel start first, then to a twentieth of a pixel around the best."""
    p = np.concatenate([profile, np.zeros(1)]).astype(np.float32)

    def best(starts, sizes):
        i, j = np.unravel_index(_comb(p, starts, sizes).argmax(), (len(sizes), len(starts)))
        return starts[j], sizes[i]

    start, size = best(np.arange(-8 * SQUARE_MAX, len(profile), 1.0), np.arange(SQUARE_MIN, SQUARE_MAX, 0.5))
    return best(start + np.arange(-2, 2.01, 0.05), size + np.arange(-0.5, 0.51, 0.05))


def signed_squares(s, x0, y0, sx, sy, k):
    """Sums of s over the board's squares and k more all round (0 off the frame), negated on dark squares and over
    the board's area, from an integral image: any 8 x 8 window scores the board shifted by whole squares."""
    H, W = s.shape
    integral = np.zeros((H + 1, W + 1))
    integral[1:, 1:] = s.cumsum(0, dtype=np.float64).cumsum(1)
    m = np.arange(-k, 9 + k)
    r = np.clip(np.round(y0 + m * sy), 0, H).astype(int)[:, None]
    c = np.clip(np.round(x0 + m * sx), 0, W).astype(int)[None, :]
    sums = (
        integral[r[1:], c[:, 1:]]
        - integral[r[:-1], c[:, 1:]]
        - integral[r[1:], c[:, :-1]]
        + integral[r[:-1], c[:, :-1]]
    )
    sign = np.where(np.add.outer(m[:-1], m[:-1]) % 2 == 0, 1.0, -1.0)
    return sums * sign / (64 * sx * sy)


def checker_score(s, box):
    """Agreement of the board with a checkerboard, a8 light, over its whole area (0 off the frame): 1 at best, about
    -1 a square out, and lower than the true board's when shifted off the frame."""
    x0, y0, w, h = box
    return float(signed_squares(s, x0, y0, w / 8, h / 8, k=0).sum())


def fit(s):
    """(board [x, y, w, h] as floats, possibly off the frame; its checker score) from a mean sign map."""
    fx, fy = flips(s, 1), flips(s, 0)
    x0, sx = comb(fx.sum(0))
    y0, sy = comb(fy[:, max(int(x0), 0) : max(int(x0 + 8 * sx), 0)].sum(1))
    x0, sx = comb(fx[max(int(y0), 0) : max(int(y0 + 8 * sy), 0)].sum(0))
    squares = signed_squares(s, x0, y0, sx, sy, k=2)
    score, a, b = max(
        ((-1) ** (a + b) * squares[b + 2 : b + 10, a + 2 : a + 10].sum(), a, b)
        for a in range(-2, 3)
        for b in range(-2, 3)
    )
    return np.array([x0 + a * sx, y0 + b * sy, 8 * sx, 8 * sy]), float(score)


def clip(box, width, height):
    x, y, w, h = box
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    return [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3] + b[2] * b[3] - ix * iy)


def find(path):
    """(overlay_roi or None, why not, checker score)."""
    sg = signs(keyframes(path))
    H, W = sg.shape[1:]
    box, score = fit(sg.mean(0, dtype=np.float32))
    roi = clip(box, W, H) if score >= MIN_SCORE else None  # an overlay that moves can blur below MIN_SCORE
    seen = []
    for i, stretch in enumerate(np.array_split(np.arange(len(sg)), STRETCHES)):
        s = sg[stretch].mean(0, dtype=np.float32)
        if roi and checker_score(s, box) >= MIN_SCORE:
            continue
        other, other_score = fit(s)
        if other_score >= MIN_SCORE and (not roi or iou(clip(other, W, H), roi) < 0.9):
            seen.append(f"{clip(other, W, H)} in stretch {i + 1}/{STRETCHES}")
    if seen:
        return None, f"moves: {roi or 'none'} overall, " + ", ".join(seen), score
    return roi, "" if roi else "no overlay found", score


def _find(dataset):
    return find(video_path(dataset))


def main(datasets, overwrite, check):
    config = load_all()
    if not datasets:
        root = get_config().DATA_ROOT
        # a video of several games is cut into <id>_game_N datasets, which share its box
        found = sorted(p.parent.relative_to(root).as_posix() for p in root.glob("yunker/*/video.mp4"))
        found = [d for d in found if "_game_" not in d]
        datasets = [d for d in found if check == bool((config.get(d) or {}).get("overlay_roi")) or overwrite]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(_find, datasets))
    ious = []
    for dataset, (roi, why, score) in zip(datasets, results):
        d = config.setdefault(dataset, {})
        if check:
            label = d.get("overlay_roi")
            if label and roi:
                ious.append(iou(roi, label))
                logger.info(f"{dataset}: IoU {ious[-1]:.3f}, found {roi}, labelled {label}")
            else:
                logger.warning(f"{dataset}: {why}, labelled {label}")
        elif roi is None:
            logger.warning(f"{dataset}: {why} (score {score:.2f}), draw it by hand")
        elif d.get("overlay_roi") and not overwrite:
            logger.info(f"{dataset}: keeping {d['overlay_roi']}, found {roi}")
        else:
            d["overlay_roi"] = roi
            logger.info(f"{dataset}: {roi} (score {score:.2f})")
    if check and ious:
        ious = np.array(ious)
        logger.info(f"{len(ious)}/{len(datasets)} boxed: median IoU {np.median(ious):.3f}, lowest {ious.min():.3f}")
    elif not check:
        save_all(load_all() | {k: config[k] for k in datasets})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true", help="replace existing overlay_roi")
    parser.add_argument("--check", action="store_true", help="compare with existing overlay_roi, write nothing")
    args = parser.parse_args()
    main(args.dataset, args.overwrite, args.check)
