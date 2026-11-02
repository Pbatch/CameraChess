"""Export a ChessQueries safetensors checkpoint to ONNX for TensorRT, taking the same input as the browser model.

Input ``image`` is uint8 ``(N, R, R, 4)`` RGBA, like the LiteRT model the web app runs; output ``probs`` is float32
``(N, 64, 13)`` softmax probabilities, squares in FEN order (a8..h1), classes in ``Piece`` order. N is fixed at
export (``--batch``): with every shape known, the shape arithmetic of DINOv3's RoPE folds away, which TensorRT needs to
take the whole graph.

Unlike scripts/export_litert.py, the graph is the original model, not the WebGPU rewrite: TensorRT runs it as is.
The weights stay FP32; TensorRT builds an FP16 engine from them. The attention centroids are left out, as they only
give the board's orientation.

Usage:
    uv run python -m scripts.export_tensorrt \
        models/chessq-chesscam-dinov3-448-fp32.safetensors \
        models/chessq-chesscam-dinov3-448.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from camera_chess.chessqueries.models.browser import RGBANormalize, bake_pos_embed
from camera_chess.chessqueries.models.checkpoint import load_safetensors_model, read_safetensors_metadata


class TensorRTModel(nn.Module):
    """RGBA NHWC uint8 in, per-square softmax out; preprocessing lives in the graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.normalize = RGBANormalize()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(self.normalize(image)), dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    resolution = int(read_safetensors_metadata(args.checkpoint)["resolution"])
    recognizer = load_safetensors_model(args.checkpoint, device="cuda").float()
    bake_pos_embed(recognizer.encoder, resolution)
    model = TensorRTModel(recognizer).cuda().eval()
    dummy = torch.zeros(args.batch, resolution, resolution, 4, dtype=torch.uint8, device="cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The decoder's fused MultiheadAttention kernel has no ONNX op
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(args.output),
            dynamo=False,
            opset_version=args.opset,
            input_names=["image"],
            output_names=["probs"],
            do_constant_folding=True,
        )
    logger.info(f"-> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, {resolution}px, batch {args.batch})")


if __name__ == "__main__":
    main()
