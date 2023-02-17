import os
from glob import glob

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop():
    roi = [600, 550, 1500, 1050]
    for image_path in sorted(glob('data/hikaru/raw/*.jpg'),
                             key=lambda x: int(os.path.basename(x).replace('.jpg', ''))):
        frame = int(os.path.basename(image_path).replace('.jpg', ''))
        if not 1710 <= frame <= 19350:
            continue

        image = Image.open(image_path).crop(roi)
        image.save(image_path.replace('raw', 'crops'))


def main():

    crop()


if __name__ == '__main__':
    main()
