"""hf/yunker-chess/crop_yunker.py, the Hugging Face dataset's script: reading the labelled frames from a video."""

import av
import numpy as np
import pytest

from camera_chess.videos.make_training_data import crop_script

FPS = 25
FRAMES = 100


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """An H.264 video with keyframes every 10 frames and B-frames, each frame a different noise image"""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = 10
        stream.codec_context.options = {"bf": "2", "sc_threshold": "0"}
        for _ in range(FRAMES):
            image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")))
        container.mux(stream.encode())
    return path


def every_frame(path):
    """Each frame of the video, reading them all in order"""
    with av.open(str(path)) as container:
        return [f.reformat(format="bgr24", src_colorspace="ITU601").to_ndarray() for f in container.decode(video=0)]


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1, 2],  # within the first keyframe's frames
        [5, 9, 10, 11],  # either side of a keyframe
        [3, 47, 48, 88, 99],  # far apart, needing seeks
        [10, 20, 30],  # each on a keyframe
        list(range(FRAMES)),
    ],
)
def test_the_frames_read_are_those_reading_them_all_in_order_gives(video, indices):
    everything = every_frame(video)
    with av.open(str(video)) as container:
        frames = list(crop_script().read_frames(container, indices, FPS))
    assert [i for i, _ in frames] == indices
    for i, frame in frames:
        assert np.array_equal(frame, everything[i]), i


def test_frames_past_the_end_are_not_read(video):
    with av.open(str(video)) as container:
        frames = list(crop_script().read_frames(container, [50, FRAMES - 1, FRAMES, FRAMES + 30], FPS))
    assert [i for i, _ in frames] == [50, FRAMES - 1]


def test_each_row_is_cropped_from_its_frame_with_its_metadata(video, tmp_path):
    script = crop_script()
    everything = every_frame(video)
    box, overlay = [8, 4, 32, 24], [30, 0, 34, 10]
    rows = []
    for i, frame in enumerate([60, 12, 12, FRAMES + 5]):
        expected = script.crop(everything[frame], box, overlay) if frame < FRAMES else None
        rows.append(
            {
                "id": f"row{i}",
                "fen": f"fen{i}",
                "video_id": "clip",
                "game": 1,
                "frame": frame,
                "time": frame / FPS,
                "ply": i,
                "since_move": 0.5,
                "until_move": None,
                "video_width": 64,
                "video_height": 48,
                "fps": FPS,
                "board_box": box,
                "overlay_box": overlay,
                "crop_hash": script.dhash(expected) if expected is not None else "0" * 16,
            }
        )
    metadata = script.crop_video(video, rows, tmp_path)
    # In frame order, without the frame past the end
    assert [m["file_name"] for m in metadata] == ["row1.jpg", "row2.jpg", "row0.jpg"]
    assert metadata[2] == {"file_name": "row0.jpg", **{k: rows[0][k] for k in script.METADATA_FIELDS}}
    image = script.cv2.imread(str(tmp_path / "row0.jpg"))
    assert image.shape == (24, 32, 3)
    assert (image[:6, 22:] < 20).all()  # the overlay's corner is blacked out
