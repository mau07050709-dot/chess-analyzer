"""Conservative board-wide style matching when a CNN reading is uncertain."""
from functools import lru_cache
import numpy as np
from PySide6.QtCore import Qt
from .capture.board_region import rgb
from .paths import resource_root


def describe(tile):
    """Pool foreground occupancy and colour over 6x6 regions of each square."""
    tile = tile.scaled(36, 36, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    pixels = rgb(tile).astype(np.float32)
    corners = pixels[np.ix_([2, 3, 4, 31, 32, 33], [2, 3, 4, 31, 32, 33])]
    background = np.median(corners.reshape(-1, 3), axis=0)
    contrast = np.linalg.norm(pixels - background, axis=2)
    mask = np.clip((contrast - 26) / 34, 0, 1)
    mask[:2] = mask[-2:] = 0
    mask[:, :2] = mask[:, -2:] = 0
    # Suppress rank/file glyphs near outer corners.
    mask[:7, :7] = mask[-7:, -7:] = 0
    occupied = mask[5:-5, 5:-5].mean() >= .034
    channels = np.concatenate((mask[:, :, None], mask[:, :, None] * (pixels / 127.5 - 1)), axis=2)
    return channels.reshape(6, 6, 6, 6, 4).mean(axis=(1, 3)).ravel(), occupied


@lru_cache(maxsize=1)
def _bank():
    path = resource_root() / "models" / "piece_style_signatures.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return data["features"], data["labels"]
    return None


def match_style(image):
    bank = _bank()
    if bank is None:
        return None
    features, labels = bank  # style, class, rendering, descriptor
    occupied = []
    values = []
    for row, col in np.ndindex(8, 8):
        x, right = (round(i * image.width() / 8) for i in (col, col + 1))
        y, bottom = (round(i * image.height() / 8) for i in (row, row + 1))
        value, present = describe(image.copy(x, y, right - x, bottom - y))
        if present:
            occupied.append(row * 8 + col)
            values.append(value)
    if len(occupied) < 4:
        return None
    # Shortlist coherent sets using spatially separated pieces before matching all tiles.
    probes = np.linspace(0, len(values) - 1, min(5, len(values)), dtype=int)
    errors = np.zeros(len(features))
    for index in probes:
        distances = np.abs(features - values[index]).mean(axis=-1)
        errors += distances.min(axis=(1, 2))
    accepted = None
    for style in np.argsort(errors)[:3]:
        results = [(None, 1.)] * 64
        for square, value in zip(occupied, values):
            distances = np.abs(features[style] - value).mean(axis=-1).min(axis=-1)
            first, second = np.argsort(distances)[:2]
            distance = float(distances[first])
            if distance > .080 or distance > .84 * distances[second]:
                break
            results[square] = (str(labels[first]), 1. - distance)
        else:
            symbols = [s for s, _ in results]
            if symbols.count("k") == symbols.count("K") == 1:
                if accepted is not None and symbols != [s for s, _ in accepted]:
                    return None
                if accepted is None:
                    accepted = results
    return accepted
