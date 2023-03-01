import os
from glob import glob

import cv2
from PIL import ImageDraw, ImageFont, Image
import imagesize

from camera_chess.constants import COLOUR_MAP


class Visualizer:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.font = ImageFont.load_default()

    def _draw_text(self, d, bbox, text):
        text_width, text_height = self.font.getsize(text)
        y_offset = 5
        x = (bbox[0] + bbox[2] - text_width) / 2
        y = bbox[1] - y_offset

        mid_x = (bbox[0] + bbox[2]) / 2
        text_bbox = (mid_x - text_width / 2 - 5,
                     bbox[1] - y_offset - text_height,
                     mid_x + text_width / 2 + 5,
                     bbox[1] - y_offset)
        d.rectangle(text_bbox,
                    fill='black')
        d.rectangle(tuple(bbox))
        d.text((x, y - text_height), text=text)

    def add_bboxes(self, image, pred):
        image = image.copy()

        d = ImageDraw.Draw(image)
        for p in pred:
            d.rectangle(tuple(p.bbox), width=5, outline=COLOUR_MAP[p.piece])
            self._draw_text(d, p.bbox, f'{p.piece} ({p.conf:.2f})')

            bbox = [p.center[0] - 5, p.center[1] - 5,
                    p.center[0] + 5, p.center[1] + 5]
            d.ellipse(bbox, fill='green')
        for x, y in self.keypoints:
            bbox = [x - 5, y - 5,
                    x + 5, y + 5]
            d.ellipse(bbox, fill='black')

        return image

    def add_board(self, image, state):
        board_image = state.get_image()
        board_image = board_image.resize((image.height, image.height))

        new_image = Image.new('RGB', (image.width + image.height, image.height))
        new_image.paste(image, (0, 0))
        new_image.paste(board_image, (image.width, 0))

        return new_image

    def create_gif(self, image_dir, save_path):
        image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')),
                             key=lambda x: int(os.path.basename(x).replace('.jpg', '')))
        images = [Image.open(p) for p in image_paths]
        images[0].save(save_path,
                       save_all=True,
                       append_images=images[1:],
                       optimize=False,
                       duration=1000,
                       loop=0)

    def create_video(self, image_dir, fps, save_name='_replay'):
        image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')),
                             key=lambda x: int(os.path.basename(x).replace('.jpg', '')))
        width, height = imagesize.get(image_paths[0])

        video = cv2.VideoWriter(os.path.join(image_dir, f'{save_name}.avi'), 0, fps, (width, height))
        for p in image_paths:
            video.write(cv2.imread(p))

        cv2.destroyAllWindows()
        video.release()
