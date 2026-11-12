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


def default_onnx_path():
    return get_config().MODELS_ROOT / "chessvision_3lc_board_classifier.onnx"


# Boards per run of the TensorRT engine, which takes exactly this many: on an RTX 3060 laptop GPU it reads a board in
# about 1.8 ms at 32, against 2.3 ms one at a time and 3.5 ms on onnxruntime's CUDA provider
BATCH = 32


def build_engine(onnx_path):
    """A serialized FP32 TensorRT engine for BATCH boards. TF32 is off: with it (TensorRT's default on Ampere) 44 of
    185,536 squares on a video read differently from the ONNX model, against 14 without, all near MIN_CONFIDENCE.
    FP16 was no faster and differed on more."""
    import tensorrt as trt

    from camera_chess.videos.chesscam_model import TRT_LOGGER

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    shape = (BATCH, 8 * SQUARE, 8 * SQUARE, 3)
    profile = builder.create_optimization_profile()
    profile.set_shape("boards", shape, shape, shape)
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT could not build the engine")
    return engine


class SquareClassifier:
    def __init__(self, onnx_path=None, engine_path=None):
        import tensorrt as trt
        import torch

        from camera_chess.videos.chesscam_model import TRT_LOGGER

        onnx_path = onnx_path or default_onnx_path()
        engine_path = engine_path or onnx_path.with_suffix(".engine")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = None
        if engine_path.is_file() and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime:
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            logger.info("Building the square classifier's TensorRT engine, which takes about two minutes")
            serialized = build_engine(onnx_path)
            engine_path.write_bytes(serialized)
            engine = runtime.deserialize_cuda_engine(serialized)
        self.engine = engine
        self.context = engine.create_execution_context()
        self.context.set_input_shape("boards", (BATCH, 8 * SQUARE, 8 * SQUARE, 3))
        self.stream = torch.cuda.Stream()
        self.boards = torch.empty((BATCH, 8 * SQUARE, 8 * SQUARE, 3), dtype=torch.uint8, device="cuda")
        self.labels = torch.empty((BATCH, 64), dtype=torch.int64, device="cuda")
        self.log_probs = torch.empty((BATCH, 64, len(SYMBOLS)), dtype=torch.float32, device="cuda")
        for name, tensor in (("boards", self.boards), ("labels", self.labels), ("log_probabilities", self.log_probs)):
            self.context.set_tensor_address(name, tensor.data_ptr())

    def classify_batch(self, boards):
        """Each square's labels (N, 64) and log-probabilities (N, 64, len(SYMBOLS)) (see the module docstring) from up
        to BATCH (N, 8 * SQUARE, 8 * SQUARE, 3) RGB 2D board images, white at the bottom"""
        import torch

        n = len(boards)
        with torch.cuda.stream(self.stream):
            self.boards[:n].copy_(torch.from_numpy(boards), non_blocking=True)
            # The engine takes exactly BATCH boards, so a short batch is padded with blank ones
            self.boards[n:].zero_()
            self.context.execute_async_v3(self.stream.cuda_stream)
            labels, log_probs = self.labels[:n].cpu(), self.log_probs[:n].cpu()
        return labels.numpy(), log_probs.numpy()


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
        default_onnx_path(),
        input_names=["boards"],
        output_names=["labels", "log_probabilities"],
        dynamic_axes={name: {0: "n"} for name in ("boards", "labels", "log_probabilities")},
        opset_version=17,
        dynamo=False,
    )
    logger.info(f"Saved {default_onnx_path()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path")
    args = parser.parse_args()
    export(args.pt_path)
