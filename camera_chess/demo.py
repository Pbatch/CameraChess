import os

import numpy as np

from camera_chess.classifier import Classifier
from camera_chess.state import State
from camera_chess.video import Video
from camera_chess.visualizer import Visualizer


def main():
    # start = TODO
    # end = TODO
    # video_path = 'data/youtube/It_s_Blitz_says_Hikaru_Nakamura_after_his_game_against_Nihal_Sarin_World_Blitz_2022.webm'
    # keypoints = np.array([[778, 1028], [616, 723], [1181, 624], [1481, 869]], dtype=np.float32)

    start = 81
    end = 428
    video_path = 'data/youtube/Dubov_s_Phenomenal_opening_preparation_leaves_Nepomniachtchi_clueless_World_Blitz_2022.webm'
    keypoints = np.array([[493, 882], [759, 619], [1263, 685], [1150, 1005]], dtype=np.float32)

    video = Video(video_path, keypoints, start, end, target_fps=4)
    classifier = Classifier(model_path='data/480M.onnx',
                            conf_thres=0.3,
                            keypoints=video.new_keypoints)
    visualizer = Visualizer()
    state = State(video.new_keypoints)
    i = 0
    for image in video:
        pred = classifier.run(image)
        state.update(pred)
        if state.change:
            print(state.game)
            image = visualizer.add_bboxes(image, pred)
            image = visualizer.add_board(image, state)
            image.save(os.path.join('data', 'hikaru', 'positions', f'{i}.jpg'))
            image.show()
            i += 1
    visualizer.create_gif(os.path.join('data', 'hikaru', 'positions'), 'replay.gif')


if __name__ == '__main__':
    main()
