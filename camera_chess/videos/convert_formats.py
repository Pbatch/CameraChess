import subprocess

from tqdm import tqdm

from camera_chess.config import get_config


def main():
    for path in tqdm(sorted(get_config().DATA_ROOT.glob("*/*/*.mov"))):
        new_path = path.with_suffix(".webm")
        if new_path.is_file():
            continue
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                path,
                "-an",
                "-y",
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                "1M",
                "-crf",
                "10",
                "-threads",
                "0",
                "-row-mt",
                "1",
                new_path,
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
