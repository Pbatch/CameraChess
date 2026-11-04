"""How ``prepare_set.py`` reads a chess-set model: grouping its loose parts into pieces, telling their colours and
types, and finding the board's grid. Only numpy, so it is tested without Blender; ``prepare_set.py`` makes the parts
from the Blender objects.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

# Checked in this order, so "koenigin" is a queen and "knight" a knight before
# either can match king. ".{1,3}" also covers mangled umlauts ("k��nig")
TYPE_PATTERNS = [
    ("q", r"queen|k.{1,3}nigin|dame|regina"),
    ("n", r"knight|springer|horse|pferd|cavallo"),
    ("k", r"king|k.{1,3}nig"),
    ("r", r"rook|turm|torre|castle"),
    ("b", r"bishop|l(ae|ä)ufer|alfiere"),
    ("p", r"pawn|bauer|pedone"),
]
COLOUR_PATTERNS = {
    "white": r"white|wei(ss|ß)|bianc|blanc|blank|light",
    "black": r"black|schwarz|nero|scuro|dark|dunkel",
}


# -- piece detection -----------------------------------------------------------------


class Part:
    """One loose part of the model: its Blender object, kept for the caller, its vertices, and the object's and its
    materials' names"""

    def __init__(self, obj: Any, v: np.ndarray, name: str, materials: list[str]) -> None:
        self.obj = obj
        self.v = v
        self.name = name
        self.materials = materials
        self.lo, self.hi = v.min(axis=0), v.max(axis=0)
        self.centre = (self.lo + self.hi) / 2
        self.footprint = max(self.hi[0] - self.lo[0], self.hi[1] - self.lo[1])
        self.height = self.hi[2] - self.lo[2]


class Piece:
    def __init__(self, parts: list[Part]) -> None:
        self.parts = parts
        v = np.concatenate([p.v for p in parts])
        self.lo, self.hi = v.min(axis=0), v.max(axis=0)
        self.height = self.hi[2] - self.lo[2]
        # Centre of the base, not of the bounding box, which a knight's head skews
        base = v[v[:, 2] < self.lo[2] + 0.1 * self.height]
        self.base = (base.min(axis=0)[:2] + base.max(axis=0)[:2]) / 2
        top = v[v[:, 2] > self.lo[2] + 0.6 * self.height][:, :2]
        eig = np.linalg.eigvalsh(np.cov((top - top.mean(axis=0)).T)) if len(top) > 3 else (1, 1)
        self.elongation = float(math.sqrt(max(eig) / max(min(eig), 1e-12)))
        # How far the top sits off the base axis, relative to the base width:
        # large for a knight's head, near zero for turned pieces
        width = max(base.max(axis=0)[:2] - base.min(axis=0)[:2])
        self.head_offset = float(np.linalg.norm(top.mean(axis=0) - self.base) / width) if len(top) else 0.0
        self.square: tuple[int, int] | None = None
        self.text = " ".join([p.name for p in parts] + [m for p in parts for m in p.materials]).lower()
        self.colour: str | None = None
        self.type: str | None = None


def cluster(parts: list[Part], radius: float) -> list[list[Part]]:
    """Group parts whose footprint centres are chained within ``radius``."""
    centres = np.array([p.centre[:2] for p in parts])
    parent = list(range(len(parts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    for i, j in zip(*np.nonzero(np.triu(dist < radius, 1))):
        parent[root(i)] = root(j)
    groups: dict[int, list[Part]] = {}
    for i, p in enumerate(parts):
        groups.setdefault(root(i), []).append(p)
    return list(groups.values())


def find_pieces(parts: list[Part]) -> tuple[list[Piece], list[Part], float, float]:
    """Group parts into pieces. Returns the pieces, the remaining parts, a
    rough square size and the board-top height."""
    rough = max(p.footprint for p in parts) / 9  # the largest part is the board or its base
    small = [p for p in parts if p.footprint < 1.3 * rough and p.height > 0.02 * rough]
    if not small:
        raise SystemExit("found no piece-sized parts")
    # Each tall column of parts is a piece resting on the board, so the board
    # top is the median column bottom. Counting columns rather than parts
    # keeps pieces made of many parts from outvoting the rest
    columns = cluster(small, 0.35 * rough)
    bottoms = [
        min(p.lo[2] for p in c) for c in columns if max(p.hi[2] for p in c) - min(p.lo[2] for p in c) > 0.4 * rough
    ]
    if not bottoms:
        raise SystemExit("found no piece-sized parts")
    board_top = float(np.median(bottoms))

    candidates = [p for p in small if p.lo[2] > board_top - 0.05 * rough and p.hi[2] > board_top + 0.08 * rough]
    pieces = [Piece(g) for g in cluster(candidates, 0.35 * rough)]
    pieces = [p for p in pieces if p.height > 0.4 * rough and p.lo[2] < board_top + 0.15 * rough]

    # Flat bits such as felt pads were too thin to count above; give each
    # leftover small part standing on the board to the piece over it
    used = {id(part) for piece in pieces for part in piece.parts}
    bases = np.array([p.base for p in pieces])
    gaps = np.linalg.norm(bases[:, None] - bases[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    spacing = float(np.median(gaps.min(axis=1)))
    extra: dict[int, list[Part]] = {}
    for part in parts:
        if id(part) in used or part.footprint >= 1.3 * rough or part.lo[2] < board_top - 0.05 * rough:
            continue
        # Clearly narrower than a square, so board squares are never taken
        dist = np.linalg.norm(bases - part.centre[:2], axis=1)
        if dist.min() < 0.35 * rough and part.footprint < 0.9 * spacing:
            extra.setdefault(int(dist.argmin()), []).append(part)
            used.add(id(part))
    pieces = [Piece(p.parts + extra.get(i, [])) for i, p in enumerate(pieces)]
    rest = [p for p in parts if id(p) not in used]
    return pieces, rest, rough, board_top


# -- colour and type -----------------------------------------------------------------


def assign_colours(pieces: list[Piece], lum: np.ndarray, swap: bool) -> str:
    """The lighter half, by each piece's luminance ``lum``, is white. Names only
    decide when the two halves are too close in brightness to tell apart, since
    some models label their light-wood pieces "Black"."""
    order = np.argsort(lum)
    half = len(pieces) // 2
    lighter = np.zeros(len(pieces), bool)
    lighter[order[half:]] = True
    dark, light = lum[~lighter], lum[lighter]
    gap = (light.min() - dark.max()) / max(light.mean(), 1e-6)

    named = []
    for p in pieces:
        hits = [c for c, pat in COLOUR_PATTERNS.items() if re.search(pat, p.text)]
        named.append(hits[0] if len(hits) == 1 else None)
    names_usable = all(named) and named.count("white") == named.count("black")

    if gap < 0.1 and names_usable:
        method = "names"
        for p, c in zip(pieces, named):
            p.colour = c
    else:
        method = "brightness"
        for p, is_light in zip(pieces, lighter):
            p.colour = "white" if is_light else "black"
    if swap:
        for p in pieces:
            p.colour = "white" if p.colour == "black" else "black"
    return method


BACK_RANK = "rnbqkbnr"


def knight_score(piece: Piece) -> float:
    """High for a knight's head: off the base axis and longer than wide."""
    return piece.head_offset + piece.elongation - 1


def assign_types(pieces: list[Piece]) -> str:
    """Starting layout first: when all 32 pieces stand on the squares of the
    initial position, the squares say what each piece is (king and queen by
    height, in case the model swapped them). Then names, then shape."""
    squares = {p.square: p for p in pieces if p.square is not None}
    start = {(f, r): "p" for f in range(8) for r in (1, 6)}
    start.update({(f, r): BACK_RANK[f] for f in range(8) for r in (0, 7)})
    if len(squares) == 32 == len(pieces) and set(squares) == set(start):
        colour_ok = all((p.colour == "white") == (square[1] < 2) for square, p in squares.items())
        if colour_ok:
            for square, p in squares.items():
                p.type = start[square]
            for rank in (0, 7):
                d, e = squares[(3, rank)], squares[(4, rank)]
                d.type, e.type = ("k", "q") if d.height > e.height else ("q", "k")
            # Some models stand the bishops on the knights' squares, sometimes for
            # one colour only; the pair with the off-axis, elongated heads is the knights
            swapped = []
            for colour in ("white", "black"):
                knights = [p for p in pieces if p.colour == colour and p.type == "n"]
                bishops = [p for p in pieces if p.colour == colour and p.type == "b"]
                if np.mean([knight_score(p) for p in bishops]) > np.mean([knight_score(p) for p in knights]):
                    for p in knights:
                        p.type = "b"
                    for p in bishops:
                        p.type = "n"
                    swapped.append(colour)
            if swapped:
                return f"start, {' and '.join(swapped)} knights and bishops swapped"
            return "start"

    named = []
    for p in pieces:
        hit = next((t for t, pat in TYPE_PATTERNS if re.search(pat, p.text)), None)
        named.append(hit)
    expected = sorted("ppppppppnnbbrrqk" * 2)
    found = [t for t in named if t is not None]
    if len(found) == len(named) and sorted(found) == expected:
        for p, t in zip(pieces, found):
            p.type = t
        return "names"
    for colour in ("white", "black"):
        side = sorted((p for p in pieces if p.colour == colour), key=lambda p: p.height)
        if len(side) != 16:
            raise SystemExit(f"{len(side)} {colour} pieces; shape-based typing needs all 16")
        for p in side[:8]:
            p.type = "p"
        rest = side[8:]
        rest[-1].type, rest[-2].type = "k", "q"
        minor = sorted(rest[:6], key=lambda p: -knight_score(p))
        for p in minor[:2]:
            p.type = "n"
        rook_bishop = sorted(minor[2:], key=lambda p: p.height)
        for p, t in zip(rook_bishop, "rrbb"):
            p.type = t
    return "shape"


# -- board frame ---------------------------------------------------------------------


def board_angle(board: list[Part]) -> float:
    """Rotation in [-45, 45) degrees that makes the board's footprint an
    axis-aligned square, i.e. minimises its bounding-box area."""
    xy = np.concatenate([p.v[:, :2] for p in board])
    xy = xy[:: max(1, len(xy) // 20000)] - xy.mean(axis=0)
    best = min(
        np.arange(-45, 45, 0.25),
        key=lambda a: np.prod(
            np.ptp(
                xy
                @ np.array(
                    [
                        [math.cos(math.radians(a)), -math.sin(math.radians(a))],
                        [math.sin(math.radians(a)), math.cos(math.radians(a))],
                    ]
                ).T,
                axis=0,
            )
        ),
    )
    return -float(best)


def board_frame(pieces: list[Piece], board_centre: np.ndarray, grid_angle: float) -> dict:
    """Square size and centre from the piece grid; rotation from the board
    outline plus which side the white pieces stand on."""
    bases = np.array([p.base for p in pieces])
    diff = bases[:, None] - bases[None]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.argmin(axis=1)
    square = float(np.median(dist[np.arange(len(bases)), nearest]))

    # Turn so white sits on -y
    white = bases[[p.colour == "white" for p in pieces]].mean(axis=0)
    black = bases[[p.colour == "black" for p in pieces]].mean(axis=0)
    toward_black = math.degrees(math.atan2(*(black - white)[::-1]))
    turn = 90 * round((90 - (toward_black - grid_angle)) / 90) - grid_angle

    rot = np.array(
        [
            [math.cos(math.radians(turn)), -math.sin(math.radians(turn))],
            [math.sin(math.radians(turn)), math.cos(math.radians(turn))],
        ]
    )
    turned = bases @ rot.T
    # Square centres sit at centre + (i - 3.5) * square, so the grid phase fixes the centre
    # up to whole squares, and the board's own bounding box picks which one
    phase = np.angle(np.exp(2j * np.pi * (turned / square - 0.5)).mean(axis=0)) / (2 * np.pi)
    guess = rot @ board_centre / square
    centre = square * (phase + np.round(guess - phase))
    board_centre_rotated = rot @ board_centre
    residual = turned - centre
    off_grid = np.abs(residual / square - np.round(residual / square - 0.5) - 0.5).max()
    return {
        "square": square,
        "turn": turn,
        "centre_rotated": centre,
        "grid_vs_board": [float(v) for v in (centre - board_centre_rotated) / square],
        "max_off_grid": float(off_grid),
        "rotation": rot,
    }
