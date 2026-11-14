"""The Hugging Face rows of the Yunker dataset: frames in the YouTube video, and clips matched to their video."""

import numpy as np
import pytest

from camera_chess.videos import make_hf_dataset

CONFIG = {
    "yunker/abc": {"roi": [10, 20, 300, 200], "overlay_roi": [400, 0, 200, 200]},
    "yunker/xyz_game_2": {"roi": [-5, 30, 320, 210], "overlay_roi": [500, 10, 250, 250]},
}
INFO = {
    "yunker/abc": (1000, 25.0, 1280, 720),
    "yunker/xyz": (5000, 30.0, 1280, 720),
    "yunker/xyz_game_2": (900, 30.0, 1280, 720),
}


@pytest.fixture
def videos(monkeypatch):
    monkeypatch.setattr(make_hf_dataset, "video_info", INFO.__getitem__)
    monkeypatch.setattr(make_hf_dataset, "cut_starts", lambda groups: {"yunker/abc": 0, "yunker/xyz_game_2": 4000})
    monkeypatch.setattr(make_hf_dataset, "move_context", lambda group, t, fen: (5, 1.5, None))


def test_a_clip_is_found_where_its_frames_are_in_the_video():
    rng = np.random.default_rng(0)
    video = rng.uniform(0, 255, (400, 16)).astype(np.float32)
    start, error = make_hf_dataset.match_start(video[123:300] + 0.1, video)
    assert start == 123
    assert error == pytest.approx(0.1, abs=1e-4)


def test_rows_are_frames_of_the_youtube_video(videos):
    annotations = [
        {"group": "yunker/xyz_game_2", "time": 2.0, "fen": "8/8/8/8/8/8/8/8", "model_agreement": None},
        {"group": "yunker/abc", "time": 1.0, "fen": "8/8/8/8/8/8/8/K7", "model_agreement": 63},
    ]
    abc, xyz = make_hf_dataset.rows(annotations, CONFIG, ["0" * 16, "f" * 16])
    assert abc["frame"] == 25 and abc["time"] == 1.0 and abc["game"] == 1
    assert abc["youtube_url"] == "https://www.youtube.com/watch?v=abc"
    assert abc["board_box"] == [10, 20, 300, 200] and abc["model_agreement"] == 63
    # The clip's frame 60 is the video's 4060
    assert xyz["video_id"] == "xyz" and xyz["game"] == 2
    assert xyz["frame"] == 4060 and xyz["time"] == round(4060 / 30, 3)
    assert xyz["id"] == "xyz_004060"
    # Its board region, [-5, 30, 320, 210], cut off at the frame's left edge
    assert xyz["board_box"] == [0, 30, 315, 210]
    assert xyz["crop_hash"] == "0" * 16 and xyz["ply"] == 5 and xyz["until_move"] is None


def test_a_frame_labelled_twice_is_refused(videos):
    a = {"group": "yunker/abc", "time": 1.0, "fen": "8/8/8/8/8/8/8/8"}
    with pytest.raises(ValueError, match="twice"):
        make_hf_dataset.rows([a, a], CONFIG, [None, None])


def test_validation_holds_out_whole_videos_until_it_has_its_share():
    records = [{"video_id": v} for v in "aaaabbbbccddeeeeffffgggghhhhiiiijjjj"]
    train, validation = make_hf_dataset.hold_out(records, 0.1)
    assert 0.1 * len(records) <= len(validation) < 0.25 * len(records)
    assert not {r["video_id"] for r in train} & {r["video_id"] for r in validation}
    assert make_hf_dataset.hold_out(records, 0.1) == (train, validation)


def test_a_frame_knows_its_ply_and_how_far_it_is_from_the_moves(monkeypatch):
    game = {"moves": ["e2e4", "e7e5", "g1f3"], "times": [10.0, 20.0, 30.0], "pawns_left": {}}
    monkeypatch.setattr(make_hf_dataset, "load_game", lambda group: game)
    make_hf_dataset.game_positions.cache_clear()
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    ply, since, until = make_hf_dataset.move_context("yunker/m", 25.0, fen)
    # The board changes OFFSET from when the overlay shows the move
    assert ply == 2
    assert since == pytest.approx(5.0 - make_hf_dataset.OFFSET)
    assert until == pytest.approx(5.0 + make_hf_dataset.OFFSET)
    with pytest.raises(ValueError, match="no position"):
        make_hf_dataset.move_context("yunker/m", 25.0, "8/8/8/8/8/8/8/8")
    make_hf_dataset.game_positions.cache_clear()


def test_the_crop_hash_is_the_same_through_jpeg(tmp_path):
    import cv2

    script = make_hf_dataset.crop_script()
    rng = np.random.default_rng(1)
    image = cv2.resize(rng.uniform(0, 255, (9, 16, 3)).astype(np.uint8), (400, 225), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(tmp_path / "a.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    again = cv2.imread(str(tmp_path / "a.jpg"))
    assert script.hash_distance(script.dhash(image), script.dhash(again)) <= 2
