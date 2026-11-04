"""Single-image inference: checkpoint + image path(s) -> `Board` per image.

The eval path (`predict_all`) runs a recognizer over a `BoardImageDataset`; this
is the arbitrary-image path for any caller that has photos rather than a dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torchcodec.decoders import decode_image

from camera_chess.chessqueries.core import Board
from camera_chess.chessqueries.data.transforms import build_transform
from camera_chess.chessqueries.models.base import BoardRecognizer

# The released/paper checkpoints are ViT-L trained at 644px. Resolution is NOT a
# LitChessQueriesModel hyperparameter (it belongs to the data module), so it cannot be
# recovered from a checkpoint file — every entrypoint has to state it explicitly.
PAPER_RESOLUTION = 644


@dataclass(frozen=True)
class Prediction:
    """One image's predicted board."""

    image_path: Path
    board: Board

    @property
    def fen(self) -> str:
        return self.board.to_fen()

    @property
    def lichess_url(self) -> str:
        return self.board.lichess_url()


class Predictor:
    """A recognizer plus its eval transform: load the checkpoint once, predict many.

    `resolution` must match the checkpoint's training resolution (see
    `PAPER_RESOLUTION`); it is not stored in the checkpoint.
    """

    def __init__(self, model: BoardRecognizer, *, resolution: int, device: str = "cpu") -> None:
        self.model = model
        self.resolution = resolution
        self.device = device
        self._transform = build_transform(resolution, train=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        resolution: int = PAPER_RESOLUTION,
        device: str | None = None,
    ) -> Predictor:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = Path(checkpoint)
        if checkpoint.suffix == ".safetensors":
            from camera_chess.chessqueries.models.checkpoint import load_safetensors_model

            model = load_safetensors_model(checkpoint, device=device)
            if torch.device(device).type == "cpu":
                model = model.float()
        else:
            from camera_chess.chessqueries.train.lit import LitChessQueriesModel

            lit = LitChessQueriesModel.load_from_checkpoint(str(checkpoint), map_location=device)
            model = lit.model.eval().to(device)
        return cls(model, resolution=resolution, device=device)

    @torch.no_grad()
    def predict(self, paths: Sequence[Path | str], *, batch_size: int = 1) -> list[Prediction]:
        """Predict a board per image, in the order the paths were given."""
        paths = [Path(p) for p in paths]
        if not paths:
            return []
        out: list[Prediction] = []
        parameter = next(self.model.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            # Transform per image before stacking: arbitrary photos differ in size,
            # and the resize to a square `resolution` is what makes them stackable.
            batch = torch.stack([self._load(p) for p in chunk]).to(self.device, dtype=dtype)
            labels = self.model.predict_labels(batch).cpu()
            out.extend(Prediction(image_path=p, board=Board.from_tensor(labels[i])) for i, p in enumerate(chunk))
        return out

    def _load(self, path: Path) -> torch.Tensor:
        # RGB mode drops alpha and expands grayscale: uploads are not dataset images.
        return self._transform(decode_image(path, mode="RGB").float())
