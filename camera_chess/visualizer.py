import os
from collections import defaultdict
from glob import glob

import cv2
import imagesize
from PIL import ImageDraw, ImageFont, Image

from camera_chess.constants import COLOUR_MAP


class Visualizer:
    def __init__(self):
        self.font = ImageFont.load_default()

    def _draw_text(self, d, bbox, text):
        text_width, text_height = self.font.getsize(text)
        y_offset = -10
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

    @staticmethod
    def _draw_points(d, xy, colour, radius=5):
        for x, y in xy:
            bbox = [x - radius, y - radius,
                    x + radius, y + radius]
            d.ellipse(bbox, fill=colour)

    def add_bboxes_from_tracks(self, image, tracks, keypoints=None):
        image = image.copy()

        d = ImageDraw.Draw(image)
        square_to_tracks = defaultdict(list)
        for track in tracks:
            d.rectangle(tuple(track.bbox), width=5, outline=COLOUR_MAP[track.piece])
            square_to_tracks[track.square].append(track)

            self._draw_points(d, [track.center], 'green')

        for square, tracks in square_to_tracks.items():
            bbox = tracks[0].bbox
            speed = tracks[0].speed
            text_items = [f'{track.piece}={track.score:.2f}, {track.square}' for track in tracks]
            if speed > 1.0:
                text_items.append(f'v={speed:.6f}')
            text = ', '.join(text_items)
            self._draw_text(d, bbox, text)

        if keypoints is not None:
            self._draw_points(d, keypoints, 'black')

        return image

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
