"""Download datasets.

Usage:
    uv run python -m scripts.download_data chessred
    uv run python -m scripts.download_data chesscog cvchess
    uv run python -m scripts.download_data all
"""

import argparse

from loguru import logger

from camera_chess.chessqueries.data import download as dl
from camera_chess.chessqueries.data.base import DatasetName


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "datasets",
        nargs="+",
        choices=[*(d.value for d in dl.DATASETS), "all"],
        help="Which dataset(s) to download.",
    )
    args = p.parse_args(argv)

    names = list(dl.DATASETS) if "all" in args.datasets else [DatasetName(d) for d in args.datasets]
    for name in names:
        dl.DATASETS[name]()

    logger.info("Done.")


if __name__ == "__main__":
    main()
