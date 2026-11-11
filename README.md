# ChessCam Research

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/), on Python 3.13 (Linux only).

```bash
uv sync
uv run python -m camera_chess.videos.overlay_moves -d yunker/Br8Y3kCJRCs
uv run ruff check .
uv run ty check
uv run pytest
uv run pre-commit install  # once: ruff, ty and uv lock --check on each commit
```

`onnxruntime-gpu` brings its own CUDA and cuDNN as pip packages, so only an NVIDIA driver is needed for the GPU.

The heaviest dependencies are in groups, so `uv sync` leaves them out; add `--group <name>` (or `--all-groups`) for:
* `synthetic` - Blender as a module (`bpy`), for `camera_chess/synthetic/`
* `tensorrt` - TensorRT, for `camera_chess/videos/chesscam_model.py`
* `chessvision-export` - Ultralytics, to export the ChessVision-3LC weights once

A plain `uv sync` removes groups that aren't asked for, so keep passing them.

Copy `.env.example` to `.env` to put data, checkpoints or outputs on another disk. Datasets live under the
gitignored `data/`; tests that need one skip when it isn't there.

## Layout
* `camera_chess/chessqueries/` - ChessQueries: datasets, model, metrics and Lightning training
* `camera_chess/labels/` - building and reviewing the ChessCam and SLCC labels
* `camera_chess/synthetic/` - synthetic boards rendered with Blender (`bpy`), see its README
* `camera_chess/videos/` - recorded games: downloading, drawing board regions (`roi_labeller`), reading the true
  moves from the 2D overlay (`overlay_moves`), running the app's model (`chesscam_model`) and cutting training crops
  (`make_training_data`); `config.py` reads `data/video_config.yaml`
* `scripts/` - downloading and packing datasets, cross-dataset evaluation, ONNX/LiteRT/TensorRT export, and
  training on Modal (see `scripts/train_chessqueries_modal.py`)

## Licenses
This repo is under `LICENSE` (PolyForm Noncommercial 1.0.0), the license the ChessQueries code came with
(`LICENSE.chessqueries`). The ChessQueries data is under `DATA_LICENSE.chessqueries` (CC BY-NC 4.0). ChessQ Lite
is PolyForm Noncommercial, and the ChessVision-3LC weights are AGPL-3.0, so keep them out of anything shipped.

## Links
* Chesscog - https://github.com/georg-wolflein/chesscog
* Play online - https://github.com/karayaman/Play-online-chess-with-real-chess-board
* ChessboardDetect - https://github.com/Elucidation/ChessboardDetect
* LiveChess2FEN - https://github.com/davidmallasen/LiveChess2FEN

## Papers
* Chesscog - https://github.com/georg-wolflein/chesscog-report/raw/master/report.pdf
* Chamfer matching - https://static1.squarespace.com/static/5fb9c8b175e654100b273fd3/t/5fd4b60d0f11661d4c015070/1607775761754/2018_ChessPieceRecognition_WACV.pdf

## Useful

Tar scripts for Google Drive
```bash
tar -zcvf models.tar.gz models
tar --exclude='*.webm' --exclude='*.tar.gz' --exclude='*.MOV' --exclude='*.mov' --exclude='*.mp4' --exclude='*.avi' --exclude='*.npy' --exclude='data/chesscog' --exclude='data/chessred2k' --exclude='data/chessvision' -zcvf data.tar.gz data
```
