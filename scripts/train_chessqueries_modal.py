"""Train ChessQueries with Lightning on Modal and export the best model to safetensors.

Needs the local Modal tooling (the `modal` group, installed by default), which also lets the launcher check
the settings before submitting. Upload the packed datasets (scripts/pack_datasets.py) once:

    .venv/bin/modal volume put chessqueries-training dist/chessqueries-data-v8.tar /datasets/chessqueries-data-v8.tar

then submit a run (DINOv3 ViT-S/16 at 448px by default), which returns as soon as it is queued:

    .venv/bin/modal run --detach -m scripts.train_chessqueries_modal \
        --dataset /datasets/chessqueries-data-v8.tar --run-name dinov3-s-448-yunker

Every setting not passed takes TrainingConfig's default (camera_chess/chessqueries/train/training_job.py), the
only place they are set; the run's full settings are saved under data/modal-runs/.

Fine-tune released weights uploaded to the volume (the run's encoder must match them; the
resolution may differ), or just score them on the test splits as a baseline with --evaluate-only:

    .venv/bin/modal run --detach -m scripts.train_chessqueries_modal \
        --dataset /datasets/chessqueries-data-v8.tar --run-name lite-ft --encoder vit_small_patch14_dinov2.lvd142m --resolution 644 --unfreeze-epoch 0 \
        --init-weights /weights/cq-vit-s-644-v4-seed2-safetensors-fp16-r2.safetensors

Resume a stopped run from its last checkpoint with the same run name and settings:

    .venv/bin/modal run --detach -m scripts.train_chessqueries_modal \
        --dataset /datasets/chessqueries-data-v8.tar --run-name dinov3-s-448-yunker \
        --resume-checkpoint /runs/dinov3-s-448-yunker/checkpoints/last.ckpt
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
from loguru import logger

app = modal.App("chessqueries-training")
volume = modal.Volume.from_name("chessqueries-training", create_if_missing=True)
# Versions match the local uv environment (.python-version and uv.lock); tests/test_modal_training.py checks them
PYTHON_VERSION = "3.13"
TRAINING_PACKAGES = (
    "torch==2.15.0",
    "torchcodec==0.17.0",
    "torchvision==0.29.2",
    "timm==1.0.30",
    "pytorch-lightning==2.6.6",
    "torchmetrics==1.9.0",
    "safetensors==0.8.0",
    "wandb==0.22.3",
    "numpy==2.5.4",
    "pillow==13.0.0",
    "tqdm==4.70.1",
    "requests==2.34.2",
    "pydantic==2.14.2",
    "pydantic-settings==2.16.2",
    "python-dotenv==1.2.4",
    "loguru==0.7.3",
    # Only for the Digitizer, Fenit and Samryan loaders
    "chess==1.11.2",
    "opencv-python==4.11.0.86",
)
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("libgl1", "libglib2.0-0")  # for opencv-python
    .uv_pip_install(*TRAINING_PACKAGES)
    .add_local_python_source("camera_chess")
    # add_local_python_source ships only .py files; CVChess's labels are vendored JSON
    .add_local_dir("camera_chess/chessqueries/data/resources", "/root/camera_chess/chessqueries/data/resources")
)


with image.imports():
    from camera_chess.chessqueries.train.training_job import ChessQueriesTrainingJob, TrainingConfig


@app.cls(
    image=image,
    gpu="H100",
    cpu=8,
    memory=16384,
    timeout=86400,
    volumes={"/mnt/chessqueries": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
class ModalChessQueriesTrainer:
    @modal.method()
    def train(self, config: dict) -> None:
        ChessQueriesTrainingJob(TrainingConfig(**config), Path("/mnt/chessqueries"), volume.commit).run()

    @modal.method()
    def evaluate(self, config: dict) -> dict:
        return ChessQueriesTrainingJob(
            TrainingConfig(**config), Path("/mnt/chessqueries"), volume.commit
        ).evaluate_only()


@app.local_entrypoint()
def main(
    # The packed datasets on the volume, e.g. /datasets/chessqueries-data-v8.tar
    dataset: str,
    run_name: str,
    # Settings left out take TrainingConfig's defaults, so they are only set there
    epochs: int | None = None,
    batch_size: int | None = None,
    lr_scaling: str | None = None,
    resume_checkpoint: str | None = None,
    precision: str | None = None,
    resolution: int | None = None,
    encoder: str | None = None,
    datasets: str | None = None,
    learning_rate: float | None = None,
    encoder_learning_rate: float | None = None,
    warmup_epochs: int | None = None,
    unfreeze_epoch: int | None = None,
    seed: int | None = None,
    init_weights: str | None = None,
    eval_datasets: str | None = None,
    gpu_augment: bool | None = None,
    # Train on every split (no validation or test on the trained datasets)
    all_splits: bool | None = None,
    # Distil from these weights on the volume, e.g. /runs/dinov3-s-448-yunker-ft-all-v10/last.safetensors
    distill_from: str | None = None,
    distill_weight: float | None = None,
    distill_temperature: float | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    evaluate_only: bool = False,
    # Any Modal GPU type, e.g. A100-80GB
    gpu: str = "H100",
) -> None:
    settings: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_scaling": lr_scaling,
        "resume_checkpoint": resume_checkpoint,
        "precision": precision,
        "resolution": resolution,
        "encoder": encoder,
        "datasets": datasets,
        "learning_rate": learning_rate,
        "encoder_learning_rate": encoder_learning_rate,
        "warmup_epochs": warmup_epochs,
        "unfreeze_epoch": unfreeze_epoch,
        "seed": seed,
        "init_weights": init_weights,
        "eval_datasets": eval_datasets,
        "gpu_augment": gpu_augment,
        "all_splits": all_splits,
        "distill_from": distill_from,
        "distill_weight": distill_weight,
        "distill_temperature": distill_temperature,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
    }
    # Checked here rather than after a GPU has spun up, and sent and saved in full, defaults included
    config: dict[str, Any] = asdict(
        TrainingConfig(dataset=dataset, run_name=run_name, **{k: v for k, v in settings.items() if v is not None})
    )
    if evaluate_only and not config["init_weights"]:
        raise ValueError("--evaluate-only scores --init-weights, so pass them")
    submission_dir = Path("data/modal-runs")
    submission_dir.mkdir(parents=True, exist_ok=True)
    # app.cls is typed as returning the plain class; at runtime it is a modal.Cls
    trainer = ModalChessQueriesTrainer.with_options(gpu=gpu)()  # ty: ignore[unresolved-attribute]
    call = (trainer.evaluate if evaluate_only else trainer.train).spawn(config)
    dashboard_url = app.get_dashboard_url()
    logger.info(f"Submitted training run: {run_name}")
    logger.info(f"Modal call ID: {call.object_id}")
    logger.info(f"View run: {dashboard_url}")
    submission = {
        "submitted_at": datetime.now(UTC).isoformat(),
        "app_id": app.app_id,
        "call_id": call.object_id,
        "dashboard_url": dashboard_url,
        "mode": "evaluate" if evaluate_only else "train",
        "gpu": gpu,
        **config,
    }
    submission_path = submission_dir / f"{call.object_id}.json"
    submission_path.write_text(json.dumps(submission, indent=2) + "\n")
    logger.info(f"Submission saved: {submission_path}")
