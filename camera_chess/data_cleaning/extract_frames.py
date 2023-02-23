import json
import os

import chess
import chess.pgn
from PIL import Image
from tqdm import tqdm

from camera_chess.classifier import Classifier
from camera_chess.constants import CORNERS, PIECE_TO_CLASS, STUDIO_IMAGE_DIR, STUDIO_LABEL_DIR
from camera_chess.utils import load_video_config, clear_dir
from camera_chess.video import Video


def main():
    clear_dir(STUDIO_IMAGE_DIR)
    clear_dir(STUDIO_LABEL_DIR)

    dataset = 'youtube/harika_nana'
    video_config = load_video_config(dataset)
    video = Video(video_config, target_fps=4)
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

    classifier = Classifier(model_path='models/480S.xml',
                            conf_thres=0.1,
                            keypoints=video.new_keypoints)
    board = chess.Board(fen=video_config.fen)

    move_idx = 0
    for image, frame in tqdm(video):
        pred = classifier.run(image)

        pred_occupied = {p.square for p in pred}
        move = video_config.moves[move_idx]
        from_square = move[:2]
        to_square = move[2:]
        if from_square in pred_occupied or to_square not in pred_occupied:
            continue

        move_idx += 1
        board.push(chess.Move.from_uci(move))

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
        for p in pred:
            x = 100 * p.bbox[0] / video.width
            y = 100 * p.bbox[1] / video.height
            w = 100 * (p.bbox[2] - p.bbox[0]) / video.width
            h = 100 * (p.bbox[3] - p.bbox[1]) / video.height
            label = labels_template.copy()

            # Use the board to overwrite the piece classification
            square = chess.parse_square(p.square)
            piece = board.piece_at(square)
            if piece is None:
                continue
            piece = PIECE_TO_CLASS[piece]

            label['value'] = {'x': x, 'y': y, 'width': w, 'height': h, 'rectanglelabels': [piece]}
            d['annotations'][0]['result'].append(label)

        with open(os.path.join(STUDIO_LABEL_DIR, f'{frame}.json'), 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == '__main__':
    main()
