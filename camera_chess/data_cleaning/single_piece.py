import os

import numpy as np
from PIL import Image, ImageOps
from decord import VideoReader
from openvino.runtime import Tensor
from tqdm import tqdm

from camera_chess.constants import DATA_DIR, CLASSES
from camera_chess.detector import Detector
from camera_chess.utils import clear_dir


class SinglePieceVideo:
    def __init__(self, path, image_size=480, target_fps=4):
        self.path = path
        self.image_size = image_size
        self.target_fps = target_fps

        self.vr = VideoReader(self.path, num_threads=1)
        self.fps = self.vr.get_avg_fps()
        self.mod = int(round(self.fps / self.target_fps))
        self.frames = [i for i in range(len(self.vr))
                       if i % self.mod == 0]

    def __iter__(self):
        for frame in self.frames:
            image = self.vr[frame].asnumpy()
            image = np.array(ImageOps.contain(Image.fromarray(image), (self.image_size, self.image_size)))
            yield image, frame

    def __len__(self):
        return len(self.frames)


def main():
    piece = 'white-rook'
    piece_dir = os.path.join(DATA_DIR, 'single_piece', piece)
    image_dir = os.path.join(piece_dir, 'images')
    label_dir = os.path.join(piece_dir, 'labels')

    path = os.path.join(piece_dir, 'video.mp4')
    video = SinglePieceVideo(path)
    detector = Detector(model_path='models/480S-quant.xml',
                        weights_path='models/480S-quant.bin',
                        keypoints=np.array([[0, 0], [0, 1000], [1000, 1000], [1000, 0]], dtype=np.float32))
    clear_dir(image_dir)
    clear_dir(label_dir)

    class_id = CLASSES.index(piece)
    for image, frame in tqdm(video):
        detector.infer_request.set_tensor(detector.input_layer_ir, Tensor(np.expand_dims(image, axis=0)))
        detector.infer_request.infer()
        pred = detector.infer_request.get_output_tensor(0).data
        if len(pred) == 0:
            continue
        best_pred = max(pred, key=lambda x: x[4])
        conf = best_pred[4]
        if conf < 0.2:
            continue
        bbox = best_pred[:4]
        height, width = image.shape[:2]

        Image.fromarray(image).save(os.path.join(image_dir, f'{frame}.jpg'))
        with open(os.path.join(label_dir, f'{frame}.txt'), 'w') as f:
            xc = (bbox[0] + bbox[2]) / (2 * width)
            yc = (bbox[1] + bbox[3]) / (2 * height)
            w = (bbox[2] - bbox[0]) / width
            h = (bbox[3] - bbox[1]) / height

            f.write(f'{class_id} {xc} {yc} {w} {h}')


if __name__ == '__main__':
    main()
