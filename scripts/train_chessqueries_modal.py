"""Train ChessQueries with Lightning on Modal and export the best model to safetensors.

Needs the local Modal tooling (the `modal` group, installed by default), which also lets the launcher check
the settings before submitting. Upload the packed datasets (scripts/pack_datasets.py) once:

    .venv/bin/modal volume put chessqueries-training dist/chessqueries-data.tar /datasets/chessqueries-data.tar

then submit a run (DINOv3 ViT-S/16 at 448px by default), which returns as soon as it is queued:

    .venv/bin/modal run --detach -m scripts.train_chessqueries_modal --run-name dinov3-s-448-chesscam

Fine-tune released weights uploaded to the volume (the run's encoder must match them; the
resolution may differ), or just score them on the test splits as a baseline with --evaluate-only:

    .venv/bin/modal run --detach -m scripts.train_chessqueries_modal --run-name lite-ft \
        --encoder vit_small_patch14_dinov2.lvd142m --resolution 644 --unfreeze-epoch 0 \
        --init-weights /weights/cq-vit-s-644-v4-seed2-safetensors-fp16-r2.safetensors

Resume a stopped run from its last checkpoint with the same run name and settings:

    .venv/bin/modal run --detach -m scripts.train_chessqueries_modal --run-name dinov3-s-448-chesscam \
        --resume-checkpoint /runs/dinov3-s-448-chesscam/checkpoints/last.ckpt
"""

import json
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
)
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(*TRAINING_PACKAGES)
    .add_local_python_source("camera_chess")
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
    dataset: str = "/datasets/chessqueries-data.tar",
    run_name: str = "dinov3-s-448-chesscam",
    epochs: int = 45,
    batch_size: int = 16,
    lr_scaling: str = "sqrt",
    resume_checkpoint: str = "",
    precision: str = "bf16-mixed",
    resolution: int = 448,
    encoder: str = "vit_small_patch16_dinov3.lvd1689m",
    datasets: str = "chessred,chesscog,chesscam,slcc,synthetic",
    learning_rate: float = 1.4e-4,
    encoder_learning_rate: float = 1.4e-5,
    warmup_epochs: int = 3,
    unfreeze_epoch: int = 5,
    seed: int = 1,
    init_weights: str = "",
    eval_datasets: str = "chesscam,chessred,chesscog,slcc",
    gpu_augment: bool = True,
    evaluate_only: bool = False,
    # Any Modal GPU type, e.g. A100-80GB
    gpu: str = "H100",
    wandb_project: str = "chessqueries",
    wandb_entity: str = "",
) -> None:
    config: dict[str, Any] = {
        "dataset": dataset,
        "run_name": run_name,
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
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
    }
    # Fail here rather than after a GPU has spun up
    TrainingConfig(**config)
    if evaluate_only and not init_weights:
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
