import json
import os

from PIL import Image
from tqdm import tqdm

from camera_chess.constants import DATA_DIR, CHAR_TO_CATEGORY
from camera_chess.utils import clear_dir


def main():
    cv_dir = os.path.join(DATA_DIR, 'chessvision')

    for split in ['test', 'train']:
        clear_dir(os.path.join(cv_dir, split, 'images'))
        clear_dir(os.path.join(cv_dir, split, 'labels'))
        with open(os.path.join(cv_dir, 'raw', 'labels', f'{split}_bounding_boxes.json'), 'r') as f:
            annotations = json.load(f)

        for image_id, bboxes in tqdm(annotations.items()):
            old_image_path = os.path.join(cv_dir, 'raw', split, f'CV_{image_id.zfill(7)}.jpg')
            new_image_path = os.path.join(cv_dir, split, 'images', f'{image_id}.jpg')
            new_label_path = os.path.join(cv_dir, split, 'labels', f'{image_id}.json')

            image = Image.open(old_image_path).convert('RGB')
            image.save(new_image_path)

            width, height = image.width, image.height
            label = {"bboxes": []}
            for category, (l, t, r, b) in bboxes.items():
                category = CHAR_TO_CATEGORY[category[0]]
                label["bboxes"].append([
                    category,
                    l / width,
                    t / height,
                    (r - l) / width,
                    (b - t) / height
                ])
            with open(new_label_path, 'w') as f:
                json.dump(label, f, indent=4)


if __name__ == '__main__':
    main()
