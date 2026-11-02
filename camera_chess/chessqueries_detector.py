import os

import chess
import cv2
import numpy as np
import onnxruntime as ort

from camera_chess.constants import CHAR_TO_CATEGORY, CLASSES, CORNERS, MODEL_DIR

# ChessQ Lite (https://huggingface.co/joelseytre/chessq-lite), PolyForm Noncommercial 1.0.0.
# The int8 ONNX is the one Lichess serves: https://lichess1.org/assets/lifat/vision/chessq-lite-v4-int8.onnx
RESOLUTION = 644
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CQ_SYMBOLS = ".PNBRQKpnbrqk"

# CQ class index -> CLASSES index, skipping empty (index 0)
CQ_TO_CLASS = np.array([CLASSES.index(CHAR_TO_CATEGORY[s]) for s in CQ_SYMBOLS[1:]])
# CQ squares run a8..h1, ours run a1..h8 (python-chess order)
CQ_TO_SQUARE = np.array([chess.square(i % 8, 7 - i // 8) for i in range(64)])


def softmax(x, axis=-1):
    x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x / np.sum(x, axis=axis, keepdims=True)


def get_rotation(keypoints):
    """Number of 90 degree counter-clockwise turns that put White's side (a1-h1) at the bottom"""
    center = np.mean([keypoints[s] for s in CORNERS], axis=0)
    dx, dy = (keypoints["a1"] + keypoints["h1"]) / 2 - center
    if abs(dy) >= abs(dx):
        return 0 if dy > 0 else 2
    # np.rot90 moves the left edge to the bottom, and the right edge to the top
    return 1 if dx < 0 else 3


class ChessQueriesDetector:
    """Reads the whole board at once, returning a (64, len(CLASSES)) array of piece probabilities"""

    def __init__(self, model_basename="chessq-lite-v4-int8.onnx", padding_ratio=8):
        self.model_basename = model_basename
        self.padding_ratio = padding_ratio

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, self.model_basename), sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
        if self.sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDAExecutionProvider unavailable; got {self.sess.get_providers()}")

    def _crop(self, image, keypoints):
        height, width = image.shape[:2]
        keypoints_arr = np.array([keypoints[s] for s in CORNERS])
        x_min, y_min = np.min(keypoints_arr, axis=0)
        x_max, y_max = np.max(keypoints_arr, axis=0)
        pad_x = (x_max - x_min) / self.padding_ratio
        pad_y = (y_max - y_min) / self.padding_ratio
        # Pieces stick up above the far edge of the board, so pad the top more
        roi = np.array(
            [max(x_min - pad_x, 0), max(y_min - 2 * pad_y, 0), min(x_max + pad_x, width), min(y_max + pad_y, height)]
        ).astype(int)
        image = image[roi[1] : roi[3], roi[0] : roi[2]]
        return np.rot90(image, k=get_rotation(keypoints)), roi

    def _preprocess(self, image):
        # Like the training transform: a plain (stretching) resize to a square, then ImageNet normalization
        image = cv2.resize(np.ascontiguousarray(image), (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        image = (image.astype(np.float32) / 255 - MEAN) / STD
        return np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)

    def run(self, image, keypoints=None):
        """image is RGB, keypoints maps each of CORNERS to its (x, y) in the image"""
        if keypoints is not None:
            image, roi = self._crop(image, keypoints)
        else:
            roi = np.array([0, 0, image.shape[1], image.shape[0]])

        logits = self.sess.run(["logits"], {"image": self._preprocess(image)})[0][0]  # ty: ignore[not-subscriptable]
        probs = softmax(logits)

        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        state[np.ix_(CQ_TO_SQUARE, CQ_TO_CLASS)] = probs[:, 1:]

        return state, roi
