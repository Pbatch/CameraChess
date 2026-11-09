import math

import numpy as np
import pytest

from camera_chess.synthetic.set_layout import (
    BACK_RANK,
    Part,
    assign_colours,
    assign_types,
    board_angle,
    board_frame,
    cluster,
    find_pieces,
)

HEIGHTS = {"p": 0.6, "r": 0.8, "n": 0.9, "b": 1.0, "q": 1.2, "k": 1.3}


def box(x, y, width, depth, z0, z1, angle=0.0):
    corners = np.array(
        [(dx, dy, z) for dx in (-width / 2, width / 2) for dy in (-depth / 2, depth / 2) for z in (z0, z1)]
    )
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    xy = corners[:, :2] @ np.array([[c, -s], [s, c]]).T
    return np.column_stack([xy + (x, y), corners[:, 2]])


def turn(xy, angle):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return np.array(xy) @ np.array([[c, -s], [s, c]]).T


def piece_parts(kind, x, y, name, angle=0.0):
    """A piece's parts at (x, y): a body, and for a knight a head pushed off the base axis and longer than wide"""
    h = HEIGHTS[kind]
    parts = [Part(None, box(x, y, 0.5, 0.5, 0, 0.55 * h, angle), name, [])]
    if kind == "n":
        dx, dy = turn([0, 0.12], angle)
        parts.append(Part(None, box(x + dx, y + dy, 0.1, 0.45, 0.55 * h, h, angle), name, []))
    else:
        parts.append(Part(None, box(x, y, 0.35, 0.35, 0.55 * h, h, angle), name, []))
    return parts


def chess_set(angle=0.0, names=False):
    """The 32 pieces of the start position on a board, square size 1, all turned by angle about the board's centre,
    with a felt pad under the white king. Returns the parts, and each piece's (symbol, x, y)."""
    parts = [Part(None, box(0, 0, 8.4, 8.4, -0.2, 0, angle), "Board", ["Wood"])]
    layout = []
    for f in range(8):
        for rank, kind in ((0, BACK_RANK[f]), (1, "p"), (6, "p"), (7, BACK_RANK[f])):
            symbol = kind.upper() if rank < 2 else kind
            x, y = turn([f - 3.5, rank - 3.5], angle)
            name = f"{'white' if rank < 2 else 'black'} piece" if names else f"Object.{len(layout):03d}"
            parts += piece_parts(kind, x, y, name, angle)
            layout.append((symbol, x, y))
    _, x, y = next(p for p in layout if p[0] == "K")
    parts.append(Part(None, box(x, y, 0.45, 0.45, 0, 0.01, angle), "felt", []))
    return parts, layout


def colour_lum(pieces, layout):
    """Each piece's luminance: light for White, from where it stands"""
    symbols = {(round(x, 3), round(y, 3)): s for s, x, y in layout}
    return np.array([0.8 if symbols[tuple(np.round(p.base, 3))].isupper() else 0.2 for p in pieces])


def test_cluster_chains_nearby_parts():
    parts = [Part(None, box(x, 0, 0.1, 0.1, 0, 1), "", []) for x in (0, 0.2, 0.4, 2)]
    groups = sorted(len(g) for g in cluster(parts, radius=0.25))
    assert groups == [1, 3]


def test_find_pieces_groups_parts_and_finds_the_board_top():
    parts, _ = chess_set()
    pieces, rest, rough, board_top = find_pieces(parts)
    assert len(pieces) == 32 and board_top == pytest.approx(0)
    assert [p.name for p in rest] == ["Board"]
    king = next(p for p in pieces if np.allclose(p.base, (0.5, -3.5)))
    # Body, top and the felt pad underneath
    assert len(king.parts) == 3
    assert rough == pytest.approx(8.4 / 9)


def test_knights_have_off_axis_heads():
    parts, _ = chess_set()
    pieces, *_ = find_pieces(parts)
    tall = sorted(pieces, key=lambda p: -(p.head_offset + p.elongation))
    assert all(p.height == pytest.approx(HEIGHTS["n"]) for p in tall[:4])


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0, 20.0])
def test_start_position_sets_the_types_and_turns_white_to_minus_y(angle):
    parts, layout = chess_set(angle)
    pieces, rest, _, _ = find_pieces(parts)
    assert assign_colours(pieces, colour_lum(pieces, layout), swap=False) == "brightness"
    lo = np.min([p.lo[:2] for p in rest], axis=0)
    hi = np.max([p.hi[:2] for p in rest], axis=0)
    frame = board_frame(pieces, (lo + hi) / 2, board_angle(rest))
    assert frame["square"] == pytest.approx(1, abs=1e-6)
    assert frame["max_off_grid"] < 1e-6
    for p in pieces:
        x, y = frame["rotation"] @ p.base - frame["centre_rotated"]
        p.square = (round(x / frame["square"] + 3.5), round(y / frame["square"] + 3.5))
    assert assign_types(pieces) == "start"
    by_square = {p.square: p for p in pieces}
    assert [by_square[(f, 0)].type for f in range(8)] == list(BACK_RANK)
    assert all(by_square[(f, 0)].colour == "white" and by_square[(f, 7)].colour == "black" for f in range(8))
    # The king is the taller of the d and e pieces
    assert by_square[(4, 0)].type == "k" and by_square[(4, 7)].type == "k"


def test_board_angle_undoes_the_boards_turn():
    for angle in (0.0, 20.0, -30.0):
        board = [Part(None, box(0, 0, 8, 8, -0.2, 0, angle), "Board", [])]
        assert abs(board_angle(board) - angle) <= 0.25 or abs(abs(board_angle(board) - angle) - 90) <= 0.25


def test_names_decide_colours_only_when_brightness_cannot():
    parts, layout = chess_set(names=True)
    pieces, *_ = find_pieces(parts)
    assert assign_colours(pieces, np.full(len(pieces), 0.5), swap=False) == "names"
    assert sum(p.colour == "white" for p in pieces) == 16
    white = {tuple(np.round(p.base, 3)) for p in pieces if p.colour == "white"}
    assert white == {(round(x, 3), round(y, 3)) for s, x, y in layout if s.isupper()}
    # Clearly lighter pieces win over their names
    lum = colour_lum(pieces, layout)
    assert assign_colours(pieces, 1 - lum, swap=False) == "brightness"
    assert sum(p.colour == "white" and tuple(np.round(p.base, 3)) in white for p in pieces) == 0


def test_swap_colours():
    parts, layout = chess_set()
    pieces, *_ = find_pieces(parts)
    assign_colours(pieces, colour_lum(pieces, layout), swap=True)
    assert all(p.colour == "black" for p in pieces if p.base[1] < 0)


def test_shape_types_pieces_off_the_start_squares():
    parts, layout = chess_set()
    pieces, *_ = find_pieces(parts)
    assign_colours(pieces, colour_lum(pieces, layout), swap=False)
    # Not on any square, so neither the start layout nor (meaningless) names can say
    assert assign_types(pieces) == "shape"
    symbols = {(round(x, 3), round(y, 3)): s for s, x, y in layout}
    for p in pieces:
        assert p.type == symbols[tuple(np.round(p.base, 3))].lower()


def test_shape_typing_needs_sixteen_a_side():
    parts, layout = chess_set()
    pieces, *_ = find_pieces(parts)
    assign_colours(pieces, colour_lum(pieces, layout), swap=False)
    with pytest.raises(SystemExit, match="15 white pieces"):
        assign_types([p for p in pieces if not np.allclose(p.base, (-3.5, -2.5))])
