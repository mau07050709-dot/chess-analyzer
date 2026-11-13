"""Remove circular legal-move hints from empty squares before classifying pieces."""

import math

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter

from ..board_coordinates import square_to_grid
from .board_region import rgb

_TILE_SIDE = 32
_AXIS = (np.arange(_TILE_SIDE) + .5) / _TILE_SIDE - .5
_DISTANCE = np.hypot(_AXIS[:, None], _AXIS[None, :])
_CENTER = _DISTANCE <= .075
_BACKGROUND = (_DISTANCE >= .28) & (_DISTANCE <= .42)
_INTERIOR = _DISTANCE <= .43


def _dot_background(tile):
    """Require a centered, nearly uniform disk on a uniform background."""
    background = np.median(tile[_BACKGROUND], axis=0)
    center = np.median(tile[_CENTER], axis=0)
    contrast = np.max(np.abs(center - background))
    if contrast < 10:
        return None
    difference = np.max(np.abs(tile - background), axis=2)
    marked = difference > max(7, contrast * .32)
    radius = math.sqrt(np.count_nonzero(marked & _INTERIOR) / math.pi) / _TILE_SIDE
    if not .085 <= radius <= .205:
        return None
    core = _DISTANCE <= radius - .025
    outside = (_DISTANCE >= radius + .035) & _INTERIOR
    center_variation = np.max(np.abs(tile[core] - center), axis=1)
    if (np.mean(center_variation <= max(8, contrast * .22)) < .94
            or np.mean(marked[outside]) > .035):
        return None
    return background


def without_move_dots(image, board, orientation):
    """Clean only empty destinations reachable from the confirmed board.

    The chess state and original image stay untouched. In particular, a piece
    arriving on a previously empty destination must still be recognized.
    """
    if min(image.width(), image.height()) < 128:
        return image
    destinations_by_piece = {}
    for move in board.legal_moves:
        if board.piece_at(move.to_square) is None:
            destinations_by_piece.setdefault(move.from_square, set()).add(move.to_square)
    if not destinations_by_piece:
        return image
    size = _TILE_SIDE * 8
    pixels = rgb(image.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)).astype(np.float32)
    # Chess.com tints the selected source and paints a disk on each empty
    # destination. Require that whole pattern before touching any pixels.
    corners = pixels.reshape(8, _TILE_SIDE, 8, _TILE_SIDE, 3)
    corners = corners[:, (2, 29), :, :, :][:, :, :, (2, 29), :].transpose(0, 2, 1, 3, 4)
    corners = corners.reshape(8, 8, 4, 3)
    palette = np.median(corners, axis=2)
    parity = np.indices((8, 8)).sum(axis=0) % 2
    typical = np.array([np.median(palette[parity == color], axis=0) for color in (0, 1)])
    backgrounds = {}
    for square in set().union(*destinations_by_piece.values()):
        column, row = square_to_grid(square, orientation)
        y, x = row * _TILE_SIDE, column * _TILE_SIDE
        background = _dot_background(pixels[y:y + _TILE_SIDE, x:x + _TILE_SIDE])
        if background is not None:
            backgrounds[square] = background

    hints = set()
    for origin, destinations in destinations_by_piece.items():
        if not destinations <= backgrounds.keys():
            continue
        column, row = square_to_grid(origin, orientation)
        color = palette[row, column]
        if (np.max(np.abs(color - typical[parity[row, column]])) >= 18
                and np.max(np.abs(corners[row, column] - color)) <= 24):
            hints.update(destinations)
    if not hints:
        return image

    cleaned = image.copy()
    painter = QPainter(cleaned)
    for square in hints:
        column, row = square_to_grid(square, orientation)
        x0 = math.floor((column + .255) * image.width() / 8)
        x1 = math.ceil((column + .745) * image.width() / 8)
        y0 = math.floor((row + .255) * image.height() / 8)
        y1 = math.ceil((row + .745) * image.height() / 8)
        painter.fillRect(QRect(x0, y0, x1 - x0, y1 - y0), QColor(*map(round, backgrounds[square])))
    painter.end()
    return cleaned
