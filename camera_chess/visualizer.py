from PIL import ImageDraw
from icecream import ic


class Visualizer:
    COLOUR_MAP = {'pawn': 'white',
                  'knight': 'blue',
                  'bishop': 'red',
                  'rook': 'yellow',
                  'king': 'black',
                  'queen': 'grey'}

    def __init__(self):
        pass

    def plot_bboxes(self, image, pred):
        d = ImageDraw.Draw(image)
        for p in pred:
            print(p.bbox, image.width, image.height)
            d.rectangle(tuple(p.bbox), width=5, outline=self.COLOUR_MAP[p.piece.split('-')[1]])
        image.show()
        input()
