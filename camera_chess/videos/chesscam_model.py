"""Runs CameraChessWeb's model on a video's 3D board, as the app would, on TensorRT, and saves each frame's
probabilities, so the app's move tracking can be replayed and scored offline.

The model is the app's (chessq-chesscam-dinov3-448, from https://huggingface.co/PBatch23888/chessq-chesscam at the
commit the app pins), exported to ONNX by scripts/export_tensorrt.py in ChessQueries:
  input `image`: (BATCH, 448, 448, 4) uint8 RGBA
  output `probs`: (N, 64, 13) float32, squares in FEN order (a8..h1), classes in CLASSES order
Like the app, each frame is cropped to the video's `roi` (video_config.yaml) and squashed to 448 x 448 with an area
resize, at the app's 4 detections per second of video.

TensorRT runs it in FP16, except for the layers that overflow FP16 (see build_engine). On 48 frames from 3 videos it
agrees with the PyTorch FP32 model on every square, with probabilities within 0.023, at about 5.6 ms a frame on an
RTX 3060 laptop GPU (CUDA FP32: 12 ms).

Usage: python -m camera_chess.videos.chesscam_model -d yunker/Br8Y3kCJRCs [-d ...] [--fps 8] [--model name]
Writes each dataset's probabilities as raw float32 (frames, 64, 13), gzipped, to chesscam_probs.f32.gz, which Node
reads with zlib.gunzipSync and a Float32Array, and what they are to chesscam_probs.json. Frame n is at n / fps
seconds. At a rate other than the app's 4 frames a second, the files are chesscam_probs_<fps>fps.*.

--model runs another model exported the same way, <MODELS_ROOT>/<name>.onnx, to compare it with the app's in the
replay. Its files are chesscam_probs_<name>.* (and chesscam_probs_<name>_<fps>fps.*), so the app's stay as they are.
"""

import argparse
import gzip
import json
import queue
import threading
import time

import numpy as np
import tensorrt as trt
import torch
from loguru import logger

from camera_chess.config import get_config
from camera_chess.videos import ffmpeg
from camera_chess.videos.config import load_config, video_path


def default_onnx_path(model=None):
    """A model's ONNX file, by default the app's (CHESSCAM_MODEL in the settings). Its TensorRT engine, built next to it
    as .engine, takes about a minute to build, so it is kept and reused. It only works on the GPU and TensorRT version it
    was built with, so it is rebuilt if it fails to load, or if the ONNX model is newer."""
    return get_config().MODELS_ROOT / f"{model or get_config().CHESSCAM_MODEL}.onnx"


SIZE = 448
# The app's classes, in the model's output order
CLASSES = ".PNBRQKpnbrqk"
# The app's detections per second of video, at normal playback speed
FPS = 4
# Frames per model run, fixed when the model is exported
BATCH = 16
# TensorRT's builder optimisation level, 0 to 5 (its default is 3): higher searches longer for faster kernels
OPTIMIZATION_LEVEL = 3
# TensorRT keeps the first logger it is given, so every builder and runtime shares this one
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path):
    """A serialized FP16 TensorRT engine, with the layers that overflow FP16 kept in FP32: each attention's scores, from
    the MatMul of queries and keys through its softmax (exported attention scales after the MatMul, and activations
    near 2000 overflow FP16's 65504 before then), and LayerNorm. Without them every output is NaN."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.builder_optimization_level = OPTIMIZATION_LEVEL

    layers = [network.get_layer(i) for i in range(network.num_layers)]
    producer = {layer.get_output(j).name: layer for layer in layers for j in range(layer.num_outputs)}
    # Each softmax's input, traced back to and including the MatMul that makes it
    scores = set()
    for layer in layers:
        if layer.type != trt.LayerType.SOFTMAX:
            continue
        todo = [layer.get_input(0)]
        while todo:
            source = producer.get(todo.pop().name)
            if source is None or source.name in scores:
                continue
            scores.add(source.name)
            if source.type != trt.LayerType.MATRIX_MULTIPLY:
                todo += [source.get_input(k) for k in range(source.num_inputs) if source.get_input(k) is not None]
    for layer in layers:
        if layer.type in (trt.LayerType.SOFTMAX, trt.LayerType.NORMALIZATION) or layer.name in scores:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT could not build the engine")
    return engine


class ChessCamModel:
    def __init__(self, onnx_path=None, engine_path=None):
        onnx_path = onnx_path or default_onnx_path()
        engine_path = engine_path or onnx_path.with_suffix(".engine")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = None
        if engine_path.is_file() and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime:
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            logger.info("Building the TensorRT engine, which takes about a minute")
            serialized = build_engine(onnx_path)
            engine_path.write_bytes(serialized)
            engine = runtime.deserialize_cuda_engine(serialized)
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.images = torch.empty((BATCH, SIZE, SIZE, 4), dtype=torch.uint8, device="cuda")
        self.probs = torch.empty((BATCH, 64, len(CLASSES)), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address("image", self.images.data_ptr())
        self.context.set_tensor_address("probs", self.probs.data_ptr())

    def probabilities(self, images):
        """(N, 64, 13) probabilities from up to BATCH (N, SIZE, SIZE, 4) uint8 RGBA images"""
        n = len(images)
        with torch.cuda.stream(self.stream):
            self.images[:n].copy_(torch.from_numpy(images), non_blocking=True)
            # The engine takes exactly BATCH images, so a short batch is padded with blank ones
            self.images[n:].zero_()
            self.context.execute_async_v3(self.stream.cuda_stream)
            probs = self.probs[:n].cpu()
        return probs.numpy()


def probs_name(fps, model=None):
    """The base name of a dataset's probability files at a frame rate, from the app's model unless another is named"""
    name = "chesscam_probs" if model in (None, get_config().CHESSCAM_MODEL) else f"chesscam_probs_{model}"
    return name if fps == FPS else f"{name}_{fps:g}fps"


def read_frames(path, roi, fps=FPS):
    """Yields (time, SIZE x SIZE x 4 RGBA) of the 3D board region at fps, squashed to a square like the app does"""
    # Up to 4 frames a second B-frames are skipped, which puts each frame at most 80 ms from its time. Any faster, that
    # would sometimes be the same frame twice. The crop is exact, as the app crops the exact region.
    return ffmpeg.read_frames(path, roi, SIZE, fps, pix_fmt="rgba", scale="area", exact=True, skip_nonref=fps <= FPS)


def save(dataset, probs, roi, fps=FPS, model=None):
    """Writes a dataset's (frames, 64, 13) probabilities, and what they are, for the replay"""
    d = get_config().DATA_ROOT / dataset
    with gzip.open(d / f"{probs_name(fps, model)}.f32.gz", "wb") as f:
        f.write(np.ascontiguousarray(probs, dtype="<f4").tobytes())
    metadata = {
        "model": model or get_config().CHESSCAM_MODEL,
        "shape": list(probs.shape),
        "dtype": "float32, little-endian",
        "fps": fps,
        "squares": "FEN order, a8, b8, ..., h1",
        "classes": CLASSES,
        "roi": roi,
    }
    (d / f"{probs_name(fps, model)}.json").write_text(json.dumps(metadata, indent=2))


def read_batches(path, roi, fps=FPS):
    """Yields (up to BATCH, SIZE, SIZE, 4) batches of frames, read on a thread, so ffmpeg keeps decoding while the
    model runs instead of waiting on a full pipe"""
    batches = queue.Queue(maxsize=4)
    errors = []

    def read():
        try:
            batch = []
            for _, frame in read_frames(path, roi, fps):
                batch.append(frame)
                if len(batch) == BATCH:
                    batches.put(np.stack(batch))
                    batch = []
            if batch:
                batches.put(np.stack(batch))
        except Exception as error:  # noqa: BLE001 - re-raised below, on the caller's thread
            errors.append(error)
        finally:
            batches.put(None)

    threading.Thread(target=read, daemon=True).start()
    while (batch := batches.get()) is not None:
        yield batch
    if errors:
        raise errors[0]


def run(dataset, model, fps=FPS, name=None):
    roi = load_config(dataset)["roi"]
    start = time.perf_counter()
    probs = np.concatenate([model.probabilities(batch) for batch in read_batches(video_path(dataset), roi, fps)])
    save(dataset, probs, roi, fps, name)
    seconds = time.perf_counter() - start
    logger.info(f"{dataset}: {len(probs)} frames in {seconds:.1f}s ({len(probs) / seconds:.0f} frames/s)")


def main(datasets, fps, name=None):
    model = ChessCamModel(default_onnx_path(name))
    for dataset in datasets:
        run(dataset, model, fps, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", action="append", required=True)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--model", help="Another model's name, <MODELS_ROOT>/<name>.onnx, instead of the app's")
    args = parser.parse_args()
    main(args.dataset, args.fps, args.model)
