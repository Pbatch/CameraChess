"""Detect board x-corners with the LeYOLO xcorners model (480x288 input, Ultralytics v8 output)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort

from camera_chess.constants import MODEL_DIR

MODEL = Path(MODEL_DIR) / "480L_leyolo_xcorners.onnx"
FILL = 114


class XcornerDetector:
    def __init__(self, model: Path = MODEL) -> None:
        ort.preload_dlls()  # the CUDA/cuDNN libraries come from pip, not the system
        self.sess = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider"])
        if self.sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDAExecutionProvider unavailable; got {self.sess.get_providers()}")
        _, _, self.height, self.width = self.sess.get_inputs()[0].shape

    def __call__(self, image: np.ndarray, conf: float = 0.3, min_dist: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2) x-corner pixel positions in the RGB `image`, and their confidences, best first"""
        h, w = image.shape[:2]
        # Letterbox like CameraChess: fit inside the input, pad the rest with grey, centred
        s = min(self.width / w, self.height / h)
        rw, rh = round(w * s), round(h * s)
        dx, dy = self.width - rw, self.height - rh
        left, top = dx - dx // 2, dy - dy // 2
        canvas = np.full((self.height, self.width, 3), FILL, np.uint8)
        canvas[top : top + rh, left : left + rw] = cv2.resize(image, (rw, rh))
        x = (canvas.astype(np.float16) / 255).transpose(2, 0, 1)[None]
        y = cast(np.ndarray, self.sess.run(None, {"images": x})[0])[0].astype(np.float32)  # (5, n): cx, cy, w, h, conf

        keep = y[4] > conf
        points = np.stack([(y[0, keep] - left) / s, (y[1, keep] - top) / s], axis=1)
        scores = y[4, keep]
        order = np.argsort(-scores)
        points, scores = points[order], scores[order]
        # Greedy suppression: one detection per corner, keeping the most confident
        close = np.linalg.norm(points[:, None] - points[None], axis=2) < min_dist
        suppressed = np.zeros(len(points), dtype=bool)
        chosen = []
        for i in range(len(points)):
            if not suppressed[i]:
                chosen.append(i)
                suppressed |= close[i]
        return points[chosen], scores[chosen]
