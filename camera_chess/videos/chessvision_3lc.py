"""The square classifier from ChessVision-3LC (https://github.com/gudbrandtandberg/ChessVision-3LC), with only what is
needed to run it: its weights, exported to ONNX together with how it expects a square.

It is a YOLOv8n classifier of 64x64 grayscale squares cut from a 2D board, from its weights/best_yolo_classifier.pt.
The weights are AGPL-3.0 (from Ultralytics), so keep it out of anything shipped.

The exported model takes whole boards and does the pre- and post-processing itself, so it all runs on the GPU:
  input `boards`: (N, 8 * SQUARE, 8 * SQUARE, 3) uint8 RGB images of 2D boards, white at the bottom
  output `labels`: (N, 64) int64, each square's index in SYMBOLS, or UNKNOWN when it is unsure
  output `log_probabilities`: (N, 64, len(SYMBOLS)) float32, floored at log(MIN_PROBABILITY)
Squares are in python-chess order (a1..h8). The constants below are baked in, so changing any needs a new export.

Export once with: python -m camera_chess.videos.chessvision_3lc path/to/best_yolo_classifier.pt (needs ultralytics)
"""

import argparse

from loguru import logger

from camera_chess.config import get_config

ONNX_PATH = get_config().MODELS_ROOT / "chessvision_3lc_board_classifier.onnx"
# Pixels per square side, as it was trained
SQUARE = 64
# Its classes, with 'f' for an empty square
LABELS = "BKNPQRbknpqrf"
# The classes it outputs, with '.' for an empty square, in python-chess's piece order
SYMBOLS = ".PNBRQKpnbrqk"
# The label of a square read with less confidence than MIN_CONFIDENCE, e.g. under an arrow
UNKNOWN = -1
MIN_CONFIDENCE = 0.5
# Lowest probability a square's reading can give, so one misread square costs at most -log(this) when scoring
MIN_PROBABILITY = 1e-4


class SquareClassifier:
    def __init__(self, onnx_path=ONNX_PATH):
        import onnxruntime as ort

        # onnxruntime-gpu[cuda,cudnn] installs CUDA as pip packages, which it only finds once loaded
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        self.session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def classify(self, board):
        """Each square's label and log-probabilities (see the module docstring) from one RGB 2D board image, white at
        the bottom, of 8 * SQUARE pixels a side"""
        labels, log_probs = self.session.run(["labels", "log_probabilities"], {"boards": board[None]})
        return labels[0], log_probs[0]  # ty: ignore[not-subscriptable]


def export(pt_path):
    import torch
    from ultralytics import YOLO

    class BoardClassifier(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.columns = [LABELS.index("f" if symbol == "." else symbol) for symbol in SYMBOLS]

        def forward(self, boards):
            r, g, b = boards.to(torch.int32).unbind(-1)
            # cv2.COLOR_RGB2GRAY's 15-bit fixed-point weights and rounding, exactly, as the squares it was trained on
            # were made with it
            gray = torch.div(r * 9798 + g * 19235 + b * 3735 + 16384, 32768, rounding_mode="floor")
            n = gray.shape[0]
            squares = gray.reshape(n, 8, SQUARE, 8, SQUARE).permute(0, 1, 3, 2, 4)
            # Rows of squares from rank 1 up, for python-chess order, and each square mirrored, as its training
            # squares were cut from a mirrored board
            squares = squares.flip(1).flip(-1).reshape(n * 64, 1, SQUARE, SQUARE)
            squares = squares.to(torch.float32).div(255).expand(-1, 3, -1, -1)
            probs = self.model(squares).reshape(n, 64, len(LABELS))[..., self.columns]
            confidence, best = probs.max(dim=-1)
            labels = torch.where(confidence >= MIN_CONFIDENCE, best, torch.full_like(best, UNKNOWN))
            return labels, probs.clamp(min=MIN_PROBABILITY).log()

    model = YOLO(pt_path).model.float().eval()
    # The head then returns only the softmax probabilities
    model.model[-1].export = True
    example = torch.zeros(1, 8 * SQUARE, 8 * SQUARE, 3, dtype=torch.uint8)
    torch.onnx.export(
        BoardClassifier(model),
        (example,),
        ONNX_PATH,
        input_names=["boards"],
        output_names=["labels", "log_probabilities"],
        dynamic_axes={name: {0: "n"} for name in ("boards", "labels", "log_probabilities")},
        opset_version=17,
        dynamo=False,
    )
    logger.info(f"Saved {ONNX_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path")
    args = parser.parse_args()
    export(args.pt_path)
