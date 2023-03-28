import json
import os

import numpy as np
from PIL import Image, ImageOps
from decord import VideoReader
from openvino.runtime import Tensor
from tqdm import tqdm

from camera_chess.constants import DATA_DIR, CLASSES, STUDIO_LABEL_DIR, STUDIO_IMAGE_DIR
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
    clear_dir(STUDIO_IMAGE_DIR)
    clear_dir(STUDIO_LABEL_DIR)

    piece = 'white-rook'
    piece_dir = os.path.join(DATA_DIR, 'single_piece', piece)
    image_dir = os.path.join(piece_dir, 'images')
    label_dir = os.path.join(piece_dir, 'labels')

    path = os.path.join(piece_dir, 'video.mp4')
    video = SinglePieceVideo(path)
    detector = Detector(model_path='models/480S-quant.xml',
                        weights_path='models/480S-quant.bin',
                        keypoints=np.array([[0, 0], [0, 2000], [2000, 2000], [2000, 0]], dtype=np.float32))
    clear_dir(image_dir)
    clear_dir(label_dir)

    labels_template = {'from_name': 'bbox-1',
                       'to_name': 'img-1',
                       'type': 'rectanglelabels'}
    for image, frame in tqdm(video):
        d = {'data': {'img': f'/data/local-files/?d=images/{frame}.jpg'},
             'annotations': [{
                 'result': []
             }]}

        new_image_path = os.path.join(STUDIO_IMAGE_DIR, f'{frame}.jpg')
        Image.fromarray(image).save(new_image_path)

        detector.infer_request.set_tensor(detector.input_layer_ir, Tensor(np.expand_dims(image, axis=0)))
        detector.infer_request.infer()
        pred = detector.infer_request.get_output_tensor(0).data
        if len(pred) > 0:
            best_pred = max(pred, key=lambda x: x[4])
            bbox = best_pred[:4]
            height, width = image.shape[:2]

            x = 100 * bbox[0] / width
            y = 100 * bbox[1] / height
            w = 100 * (bbox[2] - bbox[0]) / width
            h = 100 * (bbox[3] - bbox[1]) / height
            label = labels_template.copy()
            label['original_width'] = width
            label['original_height'] = height
            label['value'] = {'x': x,
                              'y': y,
                              'width': w,
                              'height': h,
                              'rectanglelabels': [piece]}
            d['annotations'][0]['result'].append(label)

        with open(os.path.join(STUDIO_LABEL_DIR, f'{frame}.json'), 'w') as f:
            json.dump(d, f, indent=4)

        Image.fromarray(image).save(os.path.join(image_dir, f'{frame}.jpg'))


if __name__ == '__main__':
    main()
