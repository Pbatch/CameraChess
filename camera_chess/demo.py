import os
from glob import glob

import numpy as np
from PIL import Image

from camera_chess.classifier import Classifier
from camera_chess.state import State
from camera_chess.visualizer import Visualizer


def main():
    keypoints = np.array([[177, 476], [16, 171], [574, 73], [884, 318]], dtype=np.float32)
    image_paths = sorted(glob('data/hikaru/crops/*.jpg'),
                         key=lambda x: int(os.path.basename(x).replace('.jpg', '')))

    classifier = Classifier(model_path='data/best.onnx',
                            conf_thres=0.3,
                            keypoints=keypoints)
    visualizer = Visualizer()
    state = State(keypoints)
    i = 0
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        pred = classifier.run(image, keypoints)
        state.update(pred)
        if state.change:
            print(state.game)
            image = visualizer.add_bboxes(image, pred)
            image = visualizer.add_board(image, state)
            image.show()
            input()
    visualizer.create_gif(os.path.join('data', 'hikaru', 'positions'), 'replay.gif')


if __name__ == '__main__':
    main()
