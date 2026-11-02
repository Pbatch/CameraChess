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
    def synthetic_root(self) -> Path:
        # Renders from camera_chess/synthetic/render.py; data/synthetic holds only the FEN lists they came from
        return self.DATA_ROOT / "synthetic_5k"


@lru_cache(maxsize=1)
def get_config() -> Settings:
    return Settings()
