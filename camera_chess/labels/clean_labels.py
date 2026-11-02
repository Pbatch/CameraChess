from __future__ import annotations

import argparse
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import onnxruntime as ort
import orjson
from cleanlab.object_detection.rank import get_label_quality_scores, issues_from_scores
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import LABELS_DIR, MODEL_DIR

NUM_CLASSES = 12
INPUT_HEIGHT = 288
INPUT_WIDTH = 480


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO labels with Cleanlab and ONNX predictions.")
    parser.add_argument("--dataset", type=Path, default=Path(LABELS_DIR) / "pieces")
    parser.add_argument("--model", type=Path, default=Path(MODEL_DIR) / "480M_leyolo_pieces.onnx")
    parser.add_argument("--prefix", default="nyh")
    parser.add_argument("--cache", type=Path, default=Path(LABELS_DIR) / ".cleanlab/nyh_predictions.pkl")
    parser.add_argument("--report", type=Path, default=Path(LABELS_DIR) / "cleanlab_nyh_report.json")
    parser.add_argument("--backup", type=Path, default=Path(LABELS_DIR) / ".cleanlab/backup")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prediction-confidence", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument("--class-confidence", type=float, default=0.90)
    parser.add_argument("--missing-confidence", type=float, default=0.92)
    return parser.parse_args()


def discover_samples(dataset: Path, prefix: str) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(dataset.glob(f"*/images/{prefix}*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = image_path.parents[1] / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        samples.append(Sample(image_path, label_path, width, height))
    if not samples:
        raise RuntimeError(f"No images beginning with {prefix!r} found under {dataset}")
    return samples


def read_yolo_label(sample: Sample) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows: list[list[float]] = []
    for line_number, line in enumerate(sample.label.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5:
            raise ValueError(f"{sample.label}:{line_number}: expected 5 fields")
        class_id = int(values[0])
        if class_id != values[0] or not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"{sample.label}:{line_number}: invalid class {values[0]}")
        if not all(0 <= value <= 1 for value in values[1:]):
            raise ValueError(f"{sample.label}:{line_number}: box outside normalized range")
        rows.append(values)

    yolo = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    if len(yolo) == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
        classes = np.empty(0, dtype=np.int64)
    else:
        classes = yolo[:, 0].astype(np.int64)
        x, y, width, height = yolo[:, 1:].T
        boxes = np.column_stack(
            (
                (x - width / 2) * sample.width,
                (y - height / 2) * sample.height,
                (x + width / 2) * sample.width,
                (y + height / 2) * sample.height,
            )
        ).astype(np.float32)
    return {"bboxes": boxes, "labels": classes}, yolo


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min(INPUT_WIDTH / width, INPUT_HEIGHT / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (INPUT_WIDTH - resized_width) / 2
    pad_y = (INPUT_HEIGHT - resized_height) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,) * 3)
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32) / 255, scale, pad_x, pad_y


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    if len(many) == 0:
        return np.empty(0, dtype=np.float32)
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    one_area = np.prod(np.maximum(one[2:] - one[:2], 0))
    many_area = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    keep: list[int] = []
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(classes == class_id)
        indices = indices[np.argsort(scores[indices])[::-1]]
        while len(indices):
            current = int(indices[0])
            keep.append(current)
            indices = indices[1:][box_iou(boxes[current], boxes[indices[1:]]) <= threshold]
    keep.sort(key=lambda index: scores[index], reverse=True)
    return np.asarray(keep, dtype=np.int64)


def predict(
    session: ort.InferenceSession,
    sample: Sample,
    confidence: float,
    nms_iou: float,
) -> np.ndarray:
    image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {sample.image}")
    tensor, scale, pad_x, pad_y = letterbox(image)
    model_input = session.get_inputs()[0]
    if model_input.type == "tensor(float16)":
        tensor = tensor.astype(np.float16)
    output = cast(np.ndarray, session.run(None, {model_input.name: tensor})[0])[0].T
    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_scores)), classes]
    selected = scores >= confidence
    boxes_xywh, scores, classes = boxes_xywh[selected], scores[selected], classes[selected]

    boxes = np.empty((len(boxes_xywh), 4), dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, sample.width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, sample.height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    keep = nms(boxes, scores, classes, nms_iou)

    by_class = np.empty(NUM_CLASSES, dtype=object)
    for class_id in range(NUM_CLASSES):
        class_indices = keep[classes[keep] == class_id]
        by_class[class_id] = np.column_stack((boxes[class_indices], scores[class_indices])).astype(np.float32)
    return by_class


def cache_key(samples: list[Sample], model: Path, confidence: float, nms_iou: float) -> dict[str, object]:
    return {
        "model": str(model.resolve()),
        "model_size": model.stat().st_size,
        "model_mtime_ns": model.stat().st_mtime_ns,
        "images": [str(sample.image) for sample in samples],
        "confidence": confidence,
        "nms_iou": nms_iou,
    }


def get_predictions(args: argparse.Namespace, samples: list[Sample]) -> list[np.ndarray]:
    key = cache_key(samples, args.model, args.prediction_confidence, args.nms_iou)
    if args.cache.exists():
        with args.cache.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("key") == key:
            print(f"Loaded {len(samples)} predictions from {args.cache}")
            return cached["predictions"]

    ort.preload_dlls()
    session = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA execution provider is not active: {session.get_providers()}")
    predictions = [
        predict(session, sample, args.prediction_confidence, args.nms_iou)
        for sample in tqdm(samples, desc="CUDA inference")
    ]
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as file:
        pickle.dump({"key": key, "predictions": predictions}, file, protocol=pickle.HIGHEST_PROTOCOL)
    return predictions


def flatten_predictions(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    for class_id, class_predictions in enumerate(prediction):
        if len(class_predictions):
            boxes.append(class_predictions[:, :4])
            scores.append(class_predictions[:, 4])
            classes.append(np.full(len(class_predictions), class_id, dtype=np.int64))
    if not boxes:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)


def proposed_changes(
    label: dict[str, np.ndarray],
    prediction: np.ndarray,
    class_confidence: float,
    missing_confidence: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    boxes = label["bboxes"].copy()
    classes = label["labels"].copy()
    pred_boxes, pred_scores, pred_classes = flatten_predictions(prediction)
    changes: list[dict[str, object]] = []

    for label_index, (box, old_class) in enumerate(zip(boxes, classes, strict=True)):
        overlaps = box_iou(box, pred_boxes)
        if not len(overlaps):
            continue
        candidate_indices = np.flatnonzero((overlaps >= 0.8) & (pred_scores >= class_confidence))
        if not len(candidate_indices):
            continue
        candidate = candidate_indices[np.argmax(pred_scores[candidate_indices])]
        new_class = int(pred_classes[candidate])
        old_class_score = pred_scores[(overlaps >= 0.5) & (pred_classes == old_class)].max(initial=0)
        if new_class != old_class and old_class_score < 0.25:
            classes[label_index] = new_class
            changes.append(
                {
                    "type": "class",
                    "label_index": label_index,
                    "old_class": int(old_class),
                    "new_class": new_class,
                    "confidence": float(pred_scores[candidate]),
                    "iou": float(overlaps[candidate]),
                }
            )

    if 0 < len(boxes) < 32:
        candidates = np.flatnonzero(pred_scores >= missing_confidence)
        candidates = candidates[np.argsort(pred_scores[candidates])[::-1]]
        for candidate in candidates:
            if len(boxes) >= 32:
                break
            overlaps = box_iou(pred_boxes[candidate], boxes)
            if len(overlaps) and overlaps.max() >= 0.15:
                continue
            boxes = np.vstack((boxes, pred_boxes[candidate]))
            classes = np.append(classes, pred_classes[candidate])
            changes.append(
                {
                    "type": "missing",
                    "new_class": int(pred_classes[candidate]),
                    "confidence": float(pred_scores[candidate]),
                    "box_xyxy": pred_boxes[candidate].round(3).tolist(),
                }
            )

    yolo = np.empty((len(boxes), 5), dtype=np.float32)
    yolo[:, 0] = classes
    yolo[:, 1] = (boxes[:, 0] + boxes[:, 2]) / (2 * label["image_width"])
    yolo[:, 2] = (boxes[:, 1] + boxes[:, 3]) / (2 * label["image_height"])
    yolo[:, 3] = (boxes[:, 2] - boxes[:, 0]) / label["image_width"]
    yolo[:, 4] = (boxes[:, 3] - boxes[:, 1]) / label["image_height"]
    return yolo, changes


def write_label(path: Path, yolo: np.ndarray) -> None:
    text = "".join(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n" for row in yolo)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    samples = discover_samples(args.dataset, args.prefix)
    labels: list[dict[str, np.ndarray]] = []
    for sample in tqdm(samples, desc="Reading labels"):
        label, _ = read_yolo_label(sample)
        label["image_width"] = sample.width
        label["image_height"] = sample.height
        labels.append(label)
    cleanlab_labels = [{"bboxes": label["bboxes"], "labels": label["labels"]} for label in labels]
    predictions = get_predictions(args, samples)

    print("Computing Cleanlab label-quality scores")
    scores = get_label_quality_scores(cleanlab_labels, predictions, verbose=True)
    eligible = issues_from_scores(scores, threshold=args.quality_threshold)
    issue_mask = np.zeros(len(samples), dtype=bool)
    issue_mask[eligible] = True

    change_records: list[dict[str, Any]] = []
    if args.apply:
        for index in tqdm(eligible, desc="Applying conservative fixes"):
            new_yolo, changes = proposed_changes(
                labels[index], predictions[index], args.class_confidence, args.missing_confidence
            )
            if not changes:
                continue
            sample = samples[index]
            relative_label = sample.label.relative_to(args.dataset)
            backup_path = args.backup / relative_label
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(sample.label, backup_path)
            write_label(sample.label, new_yolo)
            change_records.append(
                {
                    "image": str(sample.image),
                    "label": str(sample.label),
                    "quality_score": float(scores[index]),
                    "changes": changes,
                }
            )

    ranked = np.argsort(scores)
    report = {
        "scope": {"dataset": str(args.dataset), "prefix": args.prefix, "images": len(samples)},
        "model": str(args.model),
        "execution_provider": "CUDAExecutionProvider",
        "cleanlab": {
            "flagged_images": int(issue_mask.sum()),
            "quality_threshold": args.quality_threshold,
            "score_summary": {
                "minimum": float(scores.min()),
                "median": float(np.median(scores)),
                "mean": float(scores.mean()),
                "maximum": float(scores.max()),
            },
            "ranked_issues": [
                {"image": str(samples[index].image), "score": float(scores[index])}
                for index in ranked
                if issue_mask[index]
            ],
        },
        "applied": args.apply,
        "changed_images": len(change_records),
        "class_corrections": sum(
            change["type"] == "class" for record in change_records for change in record["changes"]
        ),
        "missing_boxes_added": sum(
            change["type"] == "missing" for record in change_records for change in record["changes"]
        ),
        "changes": change_records,
    }
    args.report.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
    print(
        f"Audited {len(samples)} images; Cleanlab flagged {issue_mask.sum()}; "
        f"changed {len(change_records)} label files; report: {args.report}"
    )


if __name__ == "__main__":
    main()
