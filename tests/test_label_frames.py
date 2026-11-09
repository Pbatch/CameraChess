import chess
import cv2
import numpy as np
import pytest
from PIL import Image

from camera_chess.labels import frames
from camera_chess.labels.frames import (
    CLASSES,
    SYMBOLS,
    Frame,
    apply_label_fixes,
    assign_squares,
    choose_rotation_by_model,
    fit_lattice,
    is_excluded,
    load_frame,
    read_yolo,
    rotate,
    source,
    to_fen,
    to_labels,
)

LATTICE = np.array([(i, j) for i in range(7) for j in range(7)], dtype=float)


def handedness(H):
    """Cross product, in y-down pixels, of the lattice's file and rank directions at the board centre"""
    inv = np.linalg.inv(H)
    o, u, v = cv2.perspectiveTransform(np.array([[[3, 3], [4, 3], [3, 4]]], dtype=np.float32), inv)[0]
    du, dv = u - o, v - o
    return du[0] * dv[1] - du[1] * dv[0]


def frame(squares, classes, name="v_1"):
    return Frame(
        name=name,
        split="train",
        image=None,  # ty: ignore[invalid-argument-type]
        group="train/v",
        squares=np.array(squares).reshape(-1, 2),
        classes=np.array(classes),
        off_board=0,
        H=np.eye(3),
    )


def test_symbols_follow_classes():
    for name, symbol in zip(CLASSES, SYMBOLS):
        colour, piece = name.split("-")
        assert symbol == chess.Piece(chess.PIECE_NAMES.index(piece), colour == "white").symbol()


def test_source_and_exclusions():
    assert source("roboflow_11_68") == "roboflow_11"
    assert is_excluded("chessred2k_5") and is_excluded("roboflow_11_68")
    assert not is_excluded("roboflow_12_68") and not is_excluded("game_3")


def test_read_yolo_converts_to_pixel_corners(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("3 0.5 0.25 0.1 0.2\n\n7 0 0 0.2 0.4\n")
    classes, boxes = read_yolo(path, 200, 100)
    assert classes.tolist() == [3, 7]
    np.testing.assert_allclose(boxes, [[90, 15, 110, 35], [-20, -20, 20, 20]])


def test_read_yolo_of_empty_file(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("\n")
    classes, boxes = read_yolo(path, 200, 100)
    assert classes.shape == (0,) and boxes.shape == (0, 4)


@pytest.mark.parametrize(
    "pixels",
    [
        np.stack([100 + 50 * LATTICE[:, 0], 500 - 50 * LATTICE[:, 1]], axis=1),
        # Mirrored, so the fit must swap axes to stay right-handed
        np.stack([100 + 50 * LATTICE[:, 0], 200 + 50 * LATTICE[:, 1]], axis=1),
        # Seen at an angle
        cv2.perspectiveTransform(
            LATTICE[None].astype(np.float32),
            np.array([[40, 8, 120], [-5, 30, 400], [0.01, 0.02, 1]], dtype=np.float32),
        )[0],
    ],
)
def test_fit_lattice_puts_corners_on_a_right_handed_integer_grid(pixels):
    H = fit_lattice(pixels)
    assert H is not None
    lattice = cv2.perspectiveTransform(pixels[None].astype(np.float32), H)[0]
    np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-3)
    assert {tuple(p) for p in np.round(lattice).astype(int)} == {tuple(p) for p in LATTICE.astype(int)}
    assert handedness(H) < 0


def test_fit_lattice_rejects_points_off_a_grid():
    assert fit_lattice(np.random.default_rng(0).uniform(0, 500, (49, 2))) is None
    jittered = LATTICE * 50 + 100
    jittered[24] += 20  # the centre corner, 0.4 squares out
    assert fit_lattice(jittered) is None


def test_rotate_quarter_turns():
    a1 = np.array([[0, 0]])
    assert rotate(a1, 1).tolist() == [[0, 7]]
    assert rotate(a1, 2).tolist() == [[7, 7]]
    squares = np.array([[0, 0], [3, 5], [7, 2]])
    assert rotate(squares, 4).tolist() == squares.tolist()
    assert rotate(rotate(squares, 1), 3).tolist() == squares.tolist()


def test_labels_and_fen_agree_on_square_order():
    # White king on e1, black rook on a8
    squares, classes = np.array([[4, 0], [0, 7]]), np.array([CLASSES.index("white-king"), CLASSES.index("black-rook")])
    labels = to_labels(squares, classes)
    assert labels[0] == CLASSES.index("black-rook") and labels[60] == CLASSES.index("white-king")
    assert (labels == -1).sum() == 62
    assert to_fen(squares, classes) == "r7/8/8/8/8/8/8/4K3"


def test_choose_rotation_by_model_finds_the_rotation_and_its_lead():
    squares, classes = np.array([[0, 1], [4, 0], [4, 7]]), np.array([9, 7, 1])
    truth = to_labels(rotate(squares, 2), classes).tolist()
    frames_ = [frame(squares, classes, "v_1"), frame(squares, classes, "v_2")]
    k, lead = choose_rotation_by_model(frames_, {"v_1": truth, "v_2": truth})
    assert k == 2
    # 64 agree at k=2; every other rotation moves all three pieces, so at best 64 - 6
    assert lead >= 4


def test_label_fixes_apply_only_while_the_reviewed_label_stands():
    fen = "4k3/8/8/8/8/8/4P3/4K3"
    fixes = {"e2": ["P", "p"], "e8": ["k", ""], "d1": ["Q", "K"]}
    fixed, used = apply_label_fixes(fen, fixes)
    assert fixed == "8/8/8/8/8/8/4p3/4K3"
    # d1 holds no queen any more, so its fix is not used
    assert used == {"e2": ["P", "p"], "e8": ["k", ""]}


def test_assign_squares_matches_anchors_one_to_one():
    lattice = np.array(
        [
            [-0.5, -0.5],  # a1's centre
            [6.5, 6.5],  # h8's centre
            [3.4, 3.4],  # inside e5 (centre 3.5, 3.5)
            [3.6, 3.6],  # also inside e5, so it takes the next nearest free square
            [7.3, 3.5],  # past the h-file edge (7) but within the margin
            [9.0, 3.5],  # off the board
        ]
    )
    squares = assign_squares(lattice, margin=0.5)
    assert squares[0].tolist() == [0, 0] and squares[1].tolist() == [7, 7]
    assert squares[5].tolist() == [-1, -1]
    assert squares[4].tolist() == [7, 4]
    assert {tuple(squares[2]), tuple(squares[3])} >= {(4, 4)}
    assert len({tuple(s) for s in squares[:5]}) == 5


@pytest.fixture
def labels_root(tmp_path, monkeypatch):
    """A labels folder with one image in both the pieces and xcorners sets, and frames' caches cleared"""
    monkeypatch.setattr(frames, "_alignments", None)
    monkeypatch.setattr(frames, "_auto_xcorners", None)
    monkeypatch.setattr(frames, "_xcorner_review", {})
    width = height = 640
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8))
    # Corners 60px apart, white at the bottom of the image, so square (f, r)'s centre is at (110 + 60f, 530 - 60r)
    corners = np.stack([140 + 60 * LATTICE[:, 0], 500 - 60 * LATTICE[:, 1]], axis=1)
    pieces = [  # class, square, box width in pixels
        (CLASSES.index("white-king"), (4, 0), 30),
        (CLASSES.index("white-pawn"), (4, 1), 24),
        (CLASSES.index("black-king"), (4, 7), 30),
    ]
    piece_lines = []
    for cls, (f, r), w in pieces:
        cx, anchor_y = 110 + 60 * f, 530 - 60 * r
        bottom = anchor_y + w / 3
        h = 2 * w
        piece_lines.append(f"{cls} {cx / width} {(bottom - h / 2) / height} {w / width} {h / height}")
    # One piece off the board
    piece_lines.append(f"{CLASSES.index('black-pawn')} {600 / width} {600 / height} {20 / width} {20 / height}")
    corner_lines = [f"0 {x / width} {y / height} {6 / width} {6 / height}" for x, y in corners]
    for kind, lines in (("pieces", piece_lines), ("xcorners", corner_lines)):
        (tmp_path / kind / "train" / "images").mkdir(parents=True)
        (tmp_path / kind / "train" / "labels").mkdir(parents=True)
        image.save(tmp_path / kind / "train" / "images" / "game_3.jpg", quality=95)
        (tmp_path / kind / "train" / "labels" / "game_3.txt").write_text("\n".join(lines))
    return tmp_path


def test_load_frame_places_pieces_on_the_grid(labels_root):
    loaded = load_frame("game_3", "train", labels_root, anchor_ratio=1 / 3, margin=0.5)
    assert loaded is not None
    assert loaded.group == "train/game" and loaded.off_board == 1
    # The lattice's rotation is arbitrary, so compare up to one
    fens = {to_fen(rotate(loaded.squares, k), loaded.classes) for k in range(4)}
    assert "4k3/8/8/8/8/8/4P3/4K3" in fens


def test_load_frame_caches_alignments(labels_root):
    load_frame("game_3", "train", labels_root, anchor_ratio=1 / 3, margin=0.5)
    frames.save_alignments(labels_root)
    cache = (labels_root / frames.ALIGNMENT_CACHE).read_text()
    assert '"train/game_3":[[1.0,0.0,0.0],[0.0,1.0,0.0]]' in cache


def test_load_frame_without_a_full_grid(labels_root):
    labels = labels_root / "xcorners" / "train" / "labels" / "game_3.txt"
    labels.write_text("\n".join(labels.read_text().splitlines()[:48]))
    assert load_frame("game_3", "train", labels_root, anchor_ratio=1 / 3, margin=0.5) is None


def test_auto_xcorners_need_a_passed_review(labels_root):
    (labels_root / frames.AUTO_XCORNERS).write_text('{"train/nyh_1": [[0, 0]]}')
    (labels_root / frames.XCORNER_REVIEW).write_text('{"train/nyh": "bad"}')
    assert frames.auto_xcorners("nyh_1", "train", labels_root) is None
    assert not frames.has_xcorners("nyh_1", "train", labels_root)
    assert frames.has_xcorners("game_3", "train", labels_root)
