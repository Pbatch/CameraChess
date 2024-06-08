import os
import subprocess
from glob import glob

from tqdm import tqdm
from camera_chess.constants import DATA_DIR


def main():
    for path in tqdm(sorted(glob(os.path.join(DATA_DIR, '*', '*', '*.mov')))):
        basename = os.path.basename(path)
        new_basename = f'{os.path.splitext(basename)[0]}.webm'
        new_path = os.path.join(os.path.dirname(path), new_basename)
        if os.path.isfile(new_path):
            continue
        subprocess.run([
            "ffmpeg",
            "-i", path,
            "-an",
            "-y",
            "-c:v", "libvpx-vp9",
            "-b:v", "1M",
            "-crf", "10",
            "-threads", "0",
            "-row-mt", "1",
            new_path
        ])


if __name__ == '__main__':
    main()
