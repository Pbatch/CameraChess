import argparse
import json
import os

import chess
import chess.pgn
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import CORNERS, STUDIO_IMAGE_DIR, STUDIO_LABEL_DIR, PIECE_TO_CLASS, CLASSES
from camera_chess.sequence_generator import SequenceGenerator
from camera_chess.tracker import Tracker
from camera_chess.utils import clear_dir


def main(dataset):
    clear_dir(STUDIO_IMAGE_DIR)
    clear_dir(STUDIO_LABEL_DIR)

    sequence_generator = SequenceGenerator(dataset)
    _, boxes = sequence_generator.create_sequence()

    video = sequence_generator.video
    keypoints_template = {'original_width': video.width,
                          'original_height': video.height,
                          'from_name': 'kp-1',
                          'to_name': 'img-1',
                          'type': 'keypointlabels'}
    keypoints_labels = []
    for (x, y), square in zip(video.new_keypoints, CORNERS):
        keypoints_label = keypoints_template.copy()
        keypoints_label['value'] = {'x': 100 * x / video.width,
                                    'y': 100 * y / video.height,
                                    'width': 1.0,
                                    'keypointlabels': [square]}
        keypoints_labels.append(keypoints_label)

    tracker = Tracker(dataset)
    logs = tracker.process_sequence(sequence_generator.sequence_path)

    board = chess.Board(fen=sequence_generator.video_config.fen)
    for i, (image, frame) in tqdm(enumerate(video), desc='Frame', total=len(video)):
        try:
            log = logs[str(i)]
        except KeyError:
            continue

        for move in log["moves"].split():
            board.push(board.parse_san(move))

        new_image_path = os.path.join(STUDIO_IMAGE_DIR, f'{frame}.jpg')
        Image.fromarray(image).save(new_image_path)

        d = {'data': {'img': f'/data/local-files/?d=images/{frame}.jpg'},
             'annotations': [{
                 'result': keypoints_labels.copy()
             }]}

        labels_template = {'original_width': video.width,
                           'original_height': video.height,
                           'from_name': 'bbox-1',
                           'to_name': 'img-1',
                           'type': 'rectanglelabels'}

        dets = boxes[boxes[:, 0] == i]
        for square, l, t, r, b, cls, conf in dets[:, 1:]:
            square = int(square)
            if square == -1:
                cls = CLASSES[int(cls)]
            else:
                piece = board.piece_at(square)
                if piece is None:
                    continue
                cls = PIECE_TO_CLASS[piece]
            label = labels_template.copy()
            label['value'] = {'x': 100 * l / video.width,
                              'y': 100 * t / video.height,
                              'width': 100 * (r - l) / video.width,
                              'height': 100 * (b - t) / video.height,
                              'rectanglelabels': [cls]}
            d['annotations'][0]['result'].append(label)

        with open(os.path.join(STUDIO_LABEL_DIR, f'{frame}.json'), 'w') as f:
            json.dump(d, f, indent=4)

        board.pop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
