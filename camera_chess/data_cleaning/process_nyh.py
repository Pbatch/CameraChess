import json
import os
import shutil
from glob import glob

import chess
import chess.pgn
import cv2
import numpy as np
import yaml
from PIL import Image
from decord import VideoReader
from tqdm import tqdm

from camera_chess.constants import CORNERS, PIECE_TO_CLASS, CLASSES, DATA_DIR
from camera_chess.sequence_generator import SequenceGenerator
from camera_chess.tracker import Tracker
from camera_chess.utils import clear_dir


def process_dataset(dataset):
    dataset_dir = os.path.join(DATA_DIR, dataset)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    clear_dir(images_dir)
    clear_dir(labels_dir)

    model_basename = "640X_v10_pieces_640x448.onnx"
    video_config_path = os.path.join("nyh", "video_config.yaml")
    sequence_generator = SequenceGenerator(dataset, model_basename, video_config_path)
    _, boxes = sequence_generator.create_sequence()

    video = sequence_generator.video
    tracker = Tracker(dataset, model_basename,
                      video_config_path=video_config_path)
    logs = tracker.process_sequence(sequence_generator.sequence_path)

    board = chess.Board(fen=sequence_generator.video_config.fen)
    n_images = 0
    for i, (image, frame) in tqdm(enumerate(video), desc='Frame', total=len(video)):
        try:
            log = logs[str(i)]
        except KeyError:
            continue

        for move in log["moves"].split():
            board.push(board.parse_san(move))

        label = {
            "keypoints": {
                square: [x / video.width, y / video.height]
                for (x, y), square in zip(video.video_config.keypoints, CORNERS)
            },
            "bboxes": []
        }

        dets = boxes[boxes[:, 0] == i]
        for square, l, t, r, b, cls, conf in dets[:, 1:]:
            square = int(square)
            piece = None
            if square != -1:
                piece = board.piece_at(square)
            if piece is not None:
                cls = PIECE_TO_CLASS[piece]
            else:
                cls = CLASSES[int(cls)]
            label["bboxes"].append([
                cls,
                l / video.width,
                t / video.height,
                (r - l) / video.width,
                (b - t) / video.height,
            ])

        new_image_path = os.path.join(images_dir, f'{n_images}.jpg')
        Image.fromarray(image).save(new_image_path)
        with open(os.path.join(labels_dir, f"{n_images}.json"), "w", encoding="utf-8") as f:
            json.dump(label, f, indent=4)
        n_images += 1

        board.pop()


def mouse_callback(event, x, y, _, keypoints):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    keypoints.append([x, y])


def make_config(dataset, path):
    vr = VideoReader(path, num_threads=1)
    first = Image.fromarray(vr[10].asnumpy()).convert("RGBA")
    last = Image.fromarray(vr[-10].asnumpy()).convert("RGBA")
    blend = Image.blend(first, last, alpha=0.5).convert("RGB")
    blend.thumbnail((640, 640), Image.Resampling.LANCZOS)

    keypoints = []
    cv2.imshow(dataset, cv2.cvtColor(np.array(blend), cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback(dataset, mouse_callback, keypoints)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if (key & 0xFF) == ord("s"):
        print("Stopping")
        exit(1)
    if (key & 0xFF) == ord("q"):
        return False
    if len(keypoints) != 4:
        print(f"n_keypoints={len(keypoints)}")
        return False

    dataset_key = dataset.replace("\\", "/")
    keypoints = [[int(x * first.width / blend.width), int(y * first.height / blend.height)]
                 for x, y in keypoints]
    end = int(len(vr) / vr.get_avg_fps())

    with open(os.path.join(DATA_DIR, "video_config.yaml"), "a") as f:
        f.write(f"\n\n{dataset_key}:\n  start: 0\n  end: {end}\n  keypoints: {keypoints}")

    return True


def make_configs():
    config_path = os.path.join("nyh", "video_config.yaml")
    with open(os.path.join(config_path)) as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}

    blacklist_path = os.path.join("nyh", "blacklist.txt")
    with open(blacklist_path, "r", encoding="utf-8") as f:
        blacklist = [line.strip() for line in f.readlines()]

    for path in tqdm(sorted(glob(os.path.join("nyh/videos/*")))):
        dataset, ext = os.path.splitext(path)
        dataset = dataset.replace(".", "")
        dataset_key = dataset.replace("\\", "/").replace("/videos", "")

        if dataset_key in config or dataset in blacklist:
            continue

        vr = VideoReader(path, num_threads=1)
        first = Image.fromarray(vr[10].asnumpy()).convert("RGBA")
        last = Image.fromarray(vr[-10].asnumpy()).convert("RGBA")
        blend = Image.blend(first, last, alpha=0.5).convert("RGB")
        blend.thumbnail((640, 640), Image.Resampling.LANCZOS)

        keypoints = []
        cv2.imshow(dataset, cv2.cvtColor(np.array(blend), cv2.COLOR_RGB2BGR))
        cv2.setMouseCallback(dataset, mouse_callback, keypoints)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if (key & 0xFF) == ord("s"):
            print("Stopping")
            exit(1)
        if (key & 0xFF) == ord("q"):
            with open(blacklist_path, "a") as f:
                f.write(f"\n{dataset}")
            continue
        if len(keypoints) != 4:
            print(f"n_keypoints={len(keypoints)}")
            continue

        dataset_key = dataset.replace("\\", "/")
        keypoints = [[int(x * first.width / blend.width), int(y * first.height / blend.height)]
                     for x, y in keypoints]
        end = int(len(vr) / vr.get_avg_fps())

        with open(config_path, "a") as f:
            f.write(f"\n\n{dataset_key}:\n  start: 0\n  end: {end}\n  keypoints: {keypoints}")


def write_labels():
    config_path = os.path.join("nyh", "video_config.yaml")
    with open(os.path.join(config_path)) as f:
        config = yaml.safe_load(f)

    for path in sorted(glob(os.path.join("nyh/videos/*"))):
        dataset, ext = os.path.splitext(path)
        dataset = dataset.replace(".", "").replace("/videos", "")
        dataset_key = dataset.replace("\\", "/")
        if dataset_key not in config:
            continue

        dataset_dir = os.path.join(DATA_DIR, dataset)
        if os.path.isdir(dataset_dir):
            continue
        os.makedirs(dataset_dir)

        shutil.copyfile(path, os.path.join(dataset_dir, f"video{ext}"))
        process_dataset(dataset)


def main():
    write_labels()


if __name__ == '__main__':
    main()
