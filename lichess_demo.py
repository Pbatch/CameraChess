import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import LICHESS_TOKEN_1, LICHESS_TOKEN_2, LICHESS_TOKEN_PETER, LICHESS_TOKEN_TOM
from camera_chess.detector import Detector
from camera_chess.lichess_game import LichessGame
from camera_chess.state import State
from camera_chess.tracker.tracker import Tracker
from camera_chess.utils import clear_dir
from camera_chess.visualizer import Visualizer
from camera_chess.webcam import Webcam


def main():
    keypoints = np.array([[477, 134], [609, 451], [13, 406], [173, 133]], dtype=np.float32)
    webcam = Webcam(keypoints, camera_id=1)

    visualizer = Visualizer()
    detector = Detector(model_path='models/480S-quant.xml',
                        weights_path='models/480S-quant.bin',
                        keypoints=webcam.new_keypoints)
    tracker = Tracker(fps=20,
                      keypoints=webcam.new_keypoints,
                      new_track_thresh=0.3,
                      track_high_thresh=0.3,
                      track_low_thresh=0.1)
    state = State()
    games = []
    tokens = [LICHESS_TOKEN_PETER, LICHESS_TOKEN_TOM]
    for token in tokens:
        game = LichessGame(token)
        game.daemon = True
        game.start()
        games.append(game)

    clear_dir('debug')

    i = 0
    for image in tqdm(webcam):
        detections = detector.run(image)
        tracks = tracker.update(detections)
        state.update(tracks)
        if not state.change:
            continue

        image = Image.fromarray(image)
        image = visualizer.add_board(image, state)
        image = visualizer.add_bboxes(image, tracks, webcam.new_keypoints)
        image.save(os.path.join('debug', f'{i}.jpg'))
        i += 1

        for game in games:
            if state.last_colour == game.colour:
                game.make_move(state.last_move)


if __name__ == '__main__':
    main()
