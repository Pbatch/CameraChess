from PIL import ImageDraw, ImageFont
from icecream import ic


class Visualizer:
    COLOUR_MAP = {'pawn': 'white',
                  'knight': 'blue',
                  'bishop': 'red',
                  'rook': 'yellow',
                  'king': 'black',
                  'queen': 'grey'}

    def __init__(self):
        self.font = ImageFont.load_default()

    def _draw_text(self, d, bbox, text):
        text_width, text_height = self.font.getsize(text)
        y_offset = 5
        x = (bbox[0] + bbox[2] - text_width) / 2
        y = bbox[1] - y_offset

        mid_x = (bbox[0] + bbox[2]) / 2
        text_bbox = (mid_x - text_width/2 - 5,
                     bbox[1] - y_offset - text_height,
                     mid_x + text_width / 2 + 5,
                     bbox[1] - y_offset)
        d.rectangle(text_bbox,
                    fill='black')
        d.rectangle(tuple(bbox))
        d.text((x, y - text_height), text=text)

    def plot_bboxes(self, image, pred):
        d = ImageDraw.Draw(image)
        for p in pred:
            print(p.bbox, image.width, image.height)
            d.rectangle(tuple(p.bbox), width=5, outline=self.COLOUR_MAP[p.piece.split('-')[1]])
            self._draw_text(d, p.bbox, p.piece)
        image.show()
        input()
