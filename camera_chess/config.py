"""Project settings and filesystem layout.

Values load from `.env` (via pydantic-settings) and the process environment.
Everything is anchored to the repo root so paths work regardless of cwd.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (camera_chess/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=True)

    # Root for all downloaded datasets and the checkpoints cache.
    DATA_ROOT: Path = REPO_ROOT / "data"
    CHECKPOINTS_ROOT: Path = REPO_ROOT / "checkpoints"
    OUTPUTS_ROOT: Path = REPO_ROOT / "outputs"
    # Exported models the video tools run (ONNX and TensorRT engines).
    MODELS_ROOT: Path = REPO_ROOT / "models"
    # The folder under DATA_ROOT of the synthetic renders trained on (camera_chess/synthetic/render.py --out)
    SYNTHETIC_SET: str = "synthetic_5k"
    # The app's model, as <MODELS_ROOT>/<name>.onnx, that camera_chess/videos/chesscam_model.py runs
    CHESSCAM_MODEL: str = "chessq-chesscam-dinov3-448"

    @property
    def labels_root(self) -> Path:
        # The ChessCam and SLCC labelling work (camera_chess/labels)
        return self.DATA_ROOT / "labels"

    @property
    def chessred_root(self) -> Path:
        return self.DATA_ROOT / "chessred"

    @property
    def chesscog_root(self) -> Path:
        return self.DATA_ROOT / "chesscog"

    @property
    def cvchess_root(self) -> Path:
        return self.DATA_ROOT / "cvchess"

    @property
    def chesscam_root(self) -> Path:
        return self.DATA_ROOT / "chesscam"

    @property
    def yunker_root(self) -> Path:
        # Crops of YunkerChess videos labelled from their 2D overlay (camera_chess/videos/make_training_data.py)
        return self.DATA_ROOT / "yunker_crops"

    @property
    def digitizer_root(self) -> Path:
        # Crops built from the Realtime-Chess-Digitizer games (camera_chess/chessqueries/data/digitizer.py)
        return self.DATA_ROOT / "digitizer"

    @property
    def fenit_root(self) -> Path:
        # Crops built from the Fenit photos on Mendeley Data (camera_chess/chessqueries/data/fenit.py)
        return self.DATA_ROOT / "fenit"

    @property
    def samryan_root(self) -> Path:
        # Crops built from samryan18/chess-dataset (camera_chess/chessqueries/data/samryan.py)
        return self.DATA_ROOT / "samryan"

    @property
    def synthetic_root(self) -> Path:
        # Renders from camera_chess/synthetic/render.py; data/synthetic holds only the FEN lists they came from
        return self.DATA_ROOT / self.SYNTHETIC_SET


@lru_cache(maxsize=1)
def get_config() -> Settings:
    return Settings()
