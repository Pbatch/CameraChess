import argparse
import json
import os
from glob import glob

from PIL import Image
from tqdm import tqdm

from camera_chess.constants import STUDIO_IMAGE_DIR, STUDIO_LABEL_DIR
from camera_chess.utils import clear_dir


def main(start, end):
    clear_dir(STUDIO_IMAGE_DIR)
    clear_dir(STUDIO_LABEL_DIR)

    for old_image_path in tqdm(sorted(glob('data/logan_spiers/raw/*.jpeg'))[start:end]):
        basename = os.path.basename(old_image_path)
        new_image_path = os.path.join(STUDIO_IMAGE_DIR, basename)
        image = Image.open(old_image_path).convert('RGB')
        image.save(new_image_path)

        label_path = os.path.join(STUDIO_LABEL_DIR, basename.replace('.jpeg', '.json'))
        d = {'data': {'img': f'/data/local-files/?d=images/{basename}'}}
        with open(label_path, 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int, required=True)
    parser.add_argument('--end', '-e', type=int, required=True)
    args = parser.parse_args()
    main(args.start, args.end)

