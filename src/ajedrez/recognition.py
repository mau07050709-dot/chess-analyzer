"""Local ONNX recognition for cropped digital chessboards with stage timings.

The crop is expected to contain one square board with no surrounding UI. The
local model classifies 64 fixed cells and reports per-square confidence; this
module never guesses the non-visual fields of a full FEN.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Mapping

import chess
from .board_coordinates import grid_to_square
import numpy as np
import onnxruntime as ort
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QTransform


FEATURE_SIDE = 24
FEATURE_SIZE = FEATURE_SIDE * FEATURE_SIDE
MIN_BOARD_SIZE = 128
MAX_ASPECT_DEVIATION = 0.12
MIN_CONFIDENCE = 0.70
MODEL_CLASSES = "1KQRBNPkqrbnp"
EMPTY_LIGHT = "__empty_light__"
EMPTY_DARK = "__empty_dark__"
PIECE_SYMBOLS = "KQRBNPkqrbnp"
TEMPLATE_CLASSES = (*PIECE_SYMBOLS, EMPTY_LIGHT, EMPTY_DARK)


@dataclass(frozen=True)
class SquarePrediction:
    square: chess.Square
    symbol: str | None
    template_class: str
    confidence: float


@dataclass(frozen=True)
class TemplateProfile:
    version: int
    # Version 2 stores all calibration exemplars. Version 1 profiles are loaded
    # as a single exemplar per class for compatibility, but should be recalibrated.
    templates: Mapping[str, tuple[tuple[float, ...], ...]]

    def to_dict(self) -> dict:
        if self.version == 1:
            serialized = {key: list(samples[0]) for key, samples in self.templates.items()}
        else:
            serialized = {
                key: [list(sample) for sample in samples]
                for key, samples in self.templates.items()
            }
        return {
            "version": self.version,
            "templates": serialized,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TemplateProfile":
        version = payload.get("version")
        if version not in (1, 2):
            raise ValueError("La versión del perfil de piezas no es compatible.")
        raw_templates = payload.get("templates")
        if not isinstance(raw_templates, dict):
            raise ValueError("El perfil no contiene plantillas de piezas.")
        templates: dict[str, tuple[float, ...]] = {}
        for key in TEMPLATE_CLASSES:
            values = raw_templates.get(key)
            if version == 1:
                raw_samples = [values]
            elif isinstance(values, list) and values and all(isinstance(sample, list) for sample in values):
                raw_samples = values
            else:
                raw_samples = []
            if not raw_samples:
                raise ValueError(f"La plantilla '{key}' falta o está dañada.")
            samples: list[tuple[float, ...]] = []
            for raw_sample in raw_samples:
                if not isinstance(raw_sample, list) or len(raw_sample) != FEATURE_SIZE:
                    raise ValueError(f"La plantilla '{key}' tiene un ejemplo dañado.")
                vector = tuple(float(value) for value in raw_sample)
                if not all(math.isfinite(value) and -1.01 <= value <= 1.01 for value in vector):
                    raise ValueError(f"La plantilla '{key}' contiene valores no válidos.")
                samples.append(vector)
            templates[key] = tuple(samples)
        return cls(version=version, templates=templates)


@dataclass(frozen=True)
class RecognitionResult:
    piece_placement: str
    orientation: str
    predictions: tuple[SquarePrediction, ...]
    confidence_threshold: float = MIN_CONFIDENCE
    timings: "RecognitionTimings | None" = None

    @property
    def uncertain_squares(self) -> tuple[str, ...]:
        return tuple(
            chess.square_name(prediction.square)
            for prediction in self.predictions
            if prediction.confidence < self.confidence_threshold
        )

    @property
    def mean_confidence(self) -> float:
        return sum(p.confidence for p in self.predictions) / max(1, len(self.predictions))

    @property
    def minimum_confidence(self) -> float:
        return min((p.confidence for p in self.predictions), default=0.0)


@dataclass(frozen=True)
class RecognitionTimings:
    """Wall-clock timings for one warm recognition pass, in milliseconds."""

    model_session_ms: float
    board_detection_ms: float
    image_prepare_ms: float
    board_split_ms: float
    inference_ms: float
    reconstruction_ms: float
    total_ms: float


def _validate_image(image: QImage) -> None:
    if image.isNull():
        raise ValueError("La imagen capturada está vacía. Vuelve a capturar el tablero.")
    if image.width() < MIN_BOARD_SIZE or image.height() < MIN_BOARD_SIZE:
        raise ValueError(f"La imagen debe medir al menos {MIN_BOARD_SIZE} × {MIN_BOARD_SIZE} píxeles.")
    aspect = image.width() / image.height()
    if abs(aspect - 1.0) > MAX_ASPECT_DEVIATION:
        raise ValueError("El recorte debe contener solo el tablero y ser aproximadamente cuadrado. Vuelve a capturarlo.")


def _image_square(row: int, column: int, orientation: str) -> chess.Square:
    return grid_to_square(row, column, orientation)


def _feature_for_cell(image: QImage, row: int, column: int) -> tuple[float, ...]:
    x0 = round(column * image.width() / 8)
    x1 = round((column + 1) * image.width() / 8)
    y0 = round(row * image.height() / 8)
    y1 = round((row + 1) * image.height() / 8)
    width, height = x1 - x0, y1 - y0
    inset = max(1, round(min(width, height) * 0.08))
    inner = QRect(x0 + inset, y0 + inset, width - 2 * inset, height - 2 * inset)
    cell = image.copy(inner).scaled(
        FEATURE_SIDE,
        FEATURE_SIDE,
        aspectMode=Qt.AspectRatioMode.IgnoreAspectRatio,
    ).convertToFormat(QImage.Format.Format_Grayscale8)
    values = [cell.constBits()[y * cell.bytesPerLine() + x] for y in range(FEATURE_SIDE) for x in range(FEATURE_SIDE)]
    border = [
        values[y * FEATURE_SIDE + x]
        for y in range(FEATURE_SIDE)
        for x in range(FEATURE_SIDE)
        if x < 2 or x >= FEATURE_SIDE - 2 or y < 2 or y >= FEATURE_SIDE - 2
    ]
    background = median(border)
    differences = [value - background for value in values]
    peak = max((abs(value) for value in differences), default=0)
    if peak < 12:
        return (0.0,) * FEATURE_SIZE
    return tuple(max(-1.0, min(1.0, value / peak)) for value in differences)


def calibrate_profile(
    image: QImage,
    orientation: str = "white",
    piece_placement: str | None = None,
) -> TemplateProfile:
    """Build a theme profile from a known position, preserving each sample.

    When ``piece_placement`` is omitted, the confirmed standard starting
    position is used. Supplying a known placement also permits labeled real
    screenshots to act as calibration samples.
    """
    _validate_image(image)
    if orientation not in ("white", "black"):
        raise ValueError("La orientación debe ser 'white' o 'black'.")
    board = chess.Board() if piece_placement is None else board_from_placement(piece_placement)
    samples: dict[str, list[tuple[float, ...]]] = {key: [] for key in TEMPLATE_CLASSES}
    for row in range(8):
        for column in range(8):
            square = _image_square(row, column, orientation)
            piece = board.piece_at(square)
            if piece:
                key = piece.symbol()
            else:
                key = EMPTY_DARK if (row + column) % 2 == 0 else EMPTY_LIGHT
            samples[key].append(_feature_for_cell(image, row, column))
    missing = [key for key in TEMPLATE_CLASSES if not samples[key]]
    if missing:
        raise ValueError("No se pudieron crear las 12 plantillas de piezas y las dos de casillas vacías.")
    return TemplateProfile(
        version=2,
        templates={key: tuple(vectors) for key, vectors in samples.items()},
    )


def save_profile(profile: TemplateProfile, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile.to_dict(), separators=(",", ":")), encoding="utf-8")


def load_profile(path: str | Path) -> TemplateProfile:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer el perfil de piezas: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("El archivo de perfil no tiene un formato válido.")
    return TemplateProfile.from_dict(payload)


@lru_cache(maxsize=2)
def _load_classifier(model_path: str) -> ort.InferenceSession:
    path = Path(model_path)
    if not path.is_file():
        raise ValueError(f"No se encontró el modelo local de reconocimiento: {path}")
    try:
        options = ort.SessionOptions()
        # Sleep between sparse captures instead of spinning idle worker threads.
        # The model, input tensor, batching and confidence rules stay unchanged.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError(f"No se pudo iniciar el reconocedor local: {exc}") from exc
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("El modelo de reconocimiento tiene una interfaz incompatible.")
    return session


def _model_input(image: QImage, stage_times: list[float] | None = None) -> np.ndarray:
    """Reproduce Fenshot's documented 256px grayscale, rank-major tile input."""
    prepare_started = time.perf_counter()
    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    source = np.frombuffer(rgba.constBits(), dtype=np.uint8).reshape(rgba.height(), rgba.bytesPerLine())
    source = source[:, : rgba.width() * 4].reshape(rgba.height(), rgba.width(), 4)
    rgb = source[:, :, :3].astype(np.float32)
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    # Pixel-center bilinear resampling to match the model's fixed board input.
    coords = (np.arange(256, dtype=np.float32) + 0.5) * (image.width() / 256.0) - 0.5
    xs = np.clip(coords, 0, image.width() - 1)
    ys = np.clip((np.arange(256, dtype=np.float32) + 0.5) * (image.height() / 256.0) - 0.5, 0, image.height() - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, image.width() - 1), np.minimum(y0 + 1, image.height() - 1)
    dx, dy = (xs - x0)[None, :], (ys - y0)[:, None]
    top = gray[y0[:, None], x0[None, :]] * (1 - dx) + gray[y0[:, None], x1[None, :]] * dx
    bottom = gray[y1[:, None], x0[None, :]] * (1 - dx) + gray[y1[:, None], x1[None, :]] * dx
    board = ((top * (1 - dy) + bottom * dy) / 255.0).astype(np.float32)
    prepare_ms = (time.perf_counter() - prepare_started) * 1000
    split_started = time.perf_counter()
    # The network expects chess-rank order (rank 1 first), with files left to
    # right. A single view/transpose replaces 64 Python slicing/assignment loops.
    tiles = np.ascontiguousarray(
        board.reshape(8, 32, 8, 32)[::-1].transpose(0, 2, 1, 3).reshape(64, 1024),
        dtype=np.float32,
    )
    if stage_times is not None:
        stage_times.extend((prepare_ms, (time.perf_counter() - split_started) * 1000))
    return tiles


def recognize_position(
    image: QImage,
    orientation: str = "white",
    confidence_threshold: float = MIN_CONFIDENCE,
    model_path: str | Path | None = None,
    *, rescue_degraded: bool = True,
) -> RecognitionResult:
    """Classify the board with a local pretrained CNN; low-probability squares are flagged."""
    total_started = time.perf_counter()
    _validate_image(image)
    if orientation not in ("white", "black"):
        raise ValueError("La orientación debe ser 'white' o 'black'.")
    from .paths import resource_root
    path = Path(model_path) if model_path else resource_root() / "models" / "chess-tiles-v2.onnx"
    stage_started = time.perf_counter()
    session = _load_classifier(str(path.resolve()))
    session_ms = (time.perf_counter() - stage_started) * 1000
    split_times: list[float] = []
    # A platform flips the square order, not the glyphs within each square.
    # Rotating the entire raster makes the CNN see upside-down pieces.
    tensor = _model_input(image, split_times)
    stage_started = time.perf_counter()
    try:
        probabilities = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0], dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Falló la clasificación del tablero: {exc}") from exc
    if probabilities.shape != (64, len(MODEL_CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError("El reconocedor devolvió resultados incompletos o no válidos.")
    inference_ms = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    predictions: list[SquarePrediction] = []
    pieces: dict[chess.Square, str | None] = {}
    for index, row_probs in enumerate(probabilities):
        rank, file_index = divmod(index, 8)
        best_index = int(np.argmax(row_probs))
        label = MODEL_CLASSES[best_index]
        symbol = None if label == "1" else label
        confidence = float(row_probs[best_index])
        square = grid_to_square(7 - rank, file_index, orientation)
        key = "empty" if symbol is None else symbol
        predictions.append(SquarePrediction(square, symbol, key, confidence))
        pieces[square] = symbol
    placement = placement_from_pieces(pieces)
    # Parse the candidate here as a structural validation, but do not invent
    # turn/castling/clock data: the live tracker owns the confirmed full state.
    board_from_placement(placement)
    reconstruction_ms = (time.perf_counter() - stage_started) * 1000
    timings = RecognitionTimings(
        session_ms, 0.0, split_times[0], split_times[1], inference_ms, reconstruction_ms,
        (time.perf_counter() - total_started) * 1000,
    )
    result = RecognitionResult(placement, orientation, tuple(predictions), confidence_threshold, timings)
    if rescue_degraded and result.uncertain_squares:
        from .theme_exemplars import classify
        exemplars = classify(image)
        method = 'rgb_exemplar_similarity'
        if exemplars is None:
            from .piece_styles import match_style
            exemplars = match_style(image)
            method = 'board_style_similarity'
        if exemplars is not None:
            predictions=tuple(SquarePrediction(grid_to_square(row,col,orientation),symbol,
                method,similarity)
                for (row,col),(symbol,similarity) in zip(np.ndindex(8,8),exemplars))
            placement=placement_from_pieces({p.square:p.symbol for p in predictions})
            return RecognitionResult(placement,orientation,predictions,confidence_threshold,
                replace(timings,total_ms=(time.perf_counter()-total_started)*1000))
    if rescue_degraded and 0 < len(result.uncertain_squares) <= 4:
        return _rescue_degraded(image, result, path, total_started)
    return result


def _rescue_degraded(image, original, model_path, started):
    """Photometric retry for a nearly resolved board, never a piece correction.

    Both different contrast views must keep ALL 64 original labels. At least
    one must meet the unchanged probability threshold on every square. This
    does not recover information erased by compression or infer from chess.
    Confident original frames perform no extra work.
    """
    from .capture.board_region import rgb
    pixels = rgb(image).astype(np.float32) / 255.0
    # Gamma correction and linear exposure gain are different responses to
    # washed-out edges. Keep backgrounds/highlights in the ORIGINAL image for
    # geometry, premoves and last-move detection.
    variants = (pixels * pixels, pixels * 1.275)
    candidates = []
    for values in variants:
        data = np.ascontiguousarray(np.clip(values * 255, 0, 255).astype(np.uint8))
        prepared = QImage(data.data, image.width(), image.height(), data.strides[0],
                          QImage.Format.Format_RGB888).copy()
        candidate = recognize_position(prepared, original.orientation,
                                       original.confidence_threshold, model_path,
                                       rescue_degraded=False)
        candidates.append(candidate)
    elapsed = (time.perf_counter()-started)*1000
    t = original.timings
    timings = replace(t, total_ms=elapsed,
                      inference_ms=t.inference_ms+sum(c.timings.inference_ms for c in candidates),
                      image_prepare_ms=t.image_prepare_ms+sum(c.timings.image_prepare_ms for c in candidates),
                      board_split_ms=t.board_split_ms+sum(c.timings.board_split_ms for c in candidates),
                      reconstruction_ms=t.reconstruction_ms+sum(c.timings.reconstruction_ms for c in candidates))
    if all(c.piece_placement == original.piece_placement for c in candidates):
        eligible = [c for c in candidates if not c.uncertain_squares]
        if eligible:
            selected = max(eligible, key=lambda c: c.minimum_confidence)
            return replace(selected, timings=timings)
    textured, extra = _rescue_empty_texture(image, original, model_path)
    if extra:
        timings = replace(timings, total_ms=(time.perf_counter()-started)*1000,
                          inference_ms=timings.inference_ms+sum(c.timings.inference_ms for c in extra),
                          image_prepare_ms=timings.image_prepare_ms+sum(c.timings.image_prepare_ms for c in extra))
    return replace(textured or original, timings=timings)


def _rescue_empty_texture(image, original, model_path):
    """Confirm only uncertain EMPTY cells on colored, textured backgrounds.

    Neutral foreground pixels veto this route, so it cannot erase the gray or
    white pieces in the wood reference. Three color-distance masks must retain
    every original label; each empty cell needs two actual CNN probabilities
    above the unchanged threshold. Never synthesize confidence from a rule.
    """
    doubts=[p for p in original.predictions if p.confidence < original.confidence_threshold]
    if not doubts or any(p.symbol is not None for p in doubts):return None,[]
    from .capture.board_region import rgb, background_colors
    from .board_coordinates import square_to_grid
    pixels=rgb(image).astype(np.float32);colors=background_colors(image)
    if colors is None or np.min((colors.max(axis=1)-colors.min(axis=1))/np.maximum(colors.max(axis=1),1))<.2:
        return None,[]
    neutral=(pixels.max(axis=2)-pixels.min(axis=2))/np.maximum(pixels.max(axis=2),1)<.18
    for prediction in doubts:
        col,row=square_to_grid(prediction.square,original.orientation)
        tile=neutral[int((row+.2)*image.height()/8):int((row+.8)*image.height()/8),
                     int((col+.2)*image.width()/8):int((col+.8)*image.width()/8)]
        if tile.size==0 or tile.mean()>.08:return None,[]
    rows,cols=np.indices(pixels.shape[:2])
    background=colors[((rows*8//image.height())+(cols*8//image.width()))%2]
    chroma=pixels/np.maximum(pixels.sum(axis=2,keepdims=True),1)
    reference=background/np.maximum(background.sum(axis=2,keepdims=True),1)
    distance=np.linalg.norm(chroma-reference,axis=2)
    candidates=[]
    for tolerance in (.03,.07,.10):
        data=np.ascontiguousarray(np.where((distance<tolerance)[:,:,None],background,pixels).astype(np.uint8))
        view=QImage(data.data,image.width(),image.height(),data.strides[0],QImage.Format.Format_RGB888).copy()
        result=recognize_position(view,original.orientation,original.confidence_threshold,
                                  model_path,rescue_degraded=False)
        candidates.append(result)
    if any(c.piece_placement!=original.piece_placement for c in candidates):return None,candidates
    predictions=[]
    for index,prediction in enumerate(original.predictions):
        if prediction.confidence<original.confidence_threshold:
            votes=sorted((c.predictions[index].confidence for c in candidates
                          if c.predictions[index].symbol is None
                          and c.predictions[index].confidence>=original.confidence_threshold),reverse=True)
            if len(votes)<2:return None,candidates
            prediction=replace(prediction,confidence=votes[1])
        predictions.append(prediction)
    return replace(original,predictions=tuple(predictions)),candidates


def recognize_position_auto_orientation(
    image: QImage,
    confidence_threshold: float = MIN_CONFIDENCE,
    model_path: str | Path | None = None,
) -> RecognitionResult:
    """Classify upright pieces and choose the view using the initial/layout prior.

    This is intended for live-session bootstrap only; ordinary polling reuses the
    orientation already determined here and performs one inference per frame.
    """
    white = recognize_position(image, "white", confidence_threshold, model_path)
    black = recognize_position(image, "black", confidence_threshold, model_path)
    return choose_recognition_orientation(white,black)


def remap_recognition(result: RecognitionResult, orientation: str) -> RecognitionResult:
    """Interpret the same 64 upright classifications in the other board view.

    No pixels are rotated, no colors/turn are changed and ONNX is not run again.
    """
    if orientation not in ("white","black"):
        raise ValueError("Orientación no válida.")
    if orientation==result.orientation:
        return result
    board=board_from_placement(result.piece_placement)
    placement=placement_from_pieces({square^63:piece.symbol() for square,piece in board.piece_map().items()})
    predictions=tuple(SquarePrediction(p.square^63,p.symbol,p.template_class,p.confidence) for p in result.predictions)
    return RecognitionResult(placement,orientation,predictions,result.confidence_threshold,result.timings)


def choose_recognition_orientation(white: RecognitionResult, black: RecognitionResult) -> RecognitionResult:
    """Opening/layout prior for views without a known legal history."""
    start = chess.Board().board_fen()
    if white.piece_placement == start and black.piece_placement != start:
        return white
    if black.piece_placement == start and white.piece_placement != start:
        return black
    # Both mappings now see upright glyphs and have the same confidence.
    # In openings/middlegames, pawn advancement and king home ranks provide a
    # visual orientation prior. It never determines the side to move. Sparse
    # endgames remain ambiguous without external coordinates/manual selection.
    def orientation_cost(result):
        board = board_from_placement(result.piece_placement)
        return sum((chess.square_rank(sq) if piece.color else 7-chess.square_rank(sq))
                   * (2 if piece.piece_type == chess.PAWN else 1)
                   for sq, piece in board.piece_map().items()
                   if piece.piece_type in (chess.PAWN, chess.KING))
    return black if orientation_cost(black) < orientation_cost(white) else white


def placement_from_pieces(pieces: Mapping[chess.Square, str | None]) -> str:
    """Create only the first FEN field; missing game metadata remains unspecified."""
    rows: list[str] = []
    for rank_index in range(7, -1, -1):
        row = ""
        empty_count = 0
        for file_index in range(8):
            symbol = pieces.get(chess.square(file_index, rank_index))
            if symbol is None:
                empty_count += 1
                continue
            if symbol not in PIECE_SYMBOLS:
                raise ValueError(f"Símbolo de pieza no válido: {symbol}")
            if empty_count:
                row += str(empty_count)
                empty_count = 0
            row += symbol
        if empty_count:
            row += str(empty_count)
        rows.append(row)
    return "/".join(rows)


def board_from_placement(placement: str) -> chess.Board:
    """Load a piece placement for display only, without asserting game metadata."""
    board = chess.Board(None)
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError("La colocación de piezas debe contener ocho filas.")
    for row_index, row in enumerate(ranks):
        file_index = 0
        for token in row:
            if token.isdigit() and token in "12345678":
                file_index += int(token)
            elif token in PIECE_SYMBOLS:
                if file_index >= 8:
                    raise ValueError("Una fila de la colocación ocupa más de ocho casillas.")
                board.set_piece_at(chess.square(file_index, 7 - row_index), chess.Piece.from_symbol(token))
                file_index += 1
            else:
                raise ValueError(f"Símbolo FEN no válido en la colocación: {token}")
        if file_index != 8:
            raise ValueError("Cada fila de la colocación debe representar ocho casillas.")
    return board


def compose_full_fen(
    placement: str,
    turn: str,
    castling: str,
    en_passant: str,
    halfmove_clock: str,
    fullmove_number: str,
) -> str:
    """Build a full FEN only after the user explicitly supplies every metadata field."""
    if turn not in ("w", "b"):
        raise ValueError("Indica explícitamente quién tiene el turno.")
    if castling != "-" and (not castling or any(char not in "KQkq" for char in castling) or len(set(castling)) != len(castling)):
        raise ValueError("Selecciona derechos de enroque válidos o confirma que no hay ninguno.")
    if en_passant != "-" and en_passant not in {f"{file}{rank}" for file in "abcdefgh" for rank in "36"}:
        raise ValueError("La casilla al paso debe ser '-' o una casilla de la fila 3 o 6.")
    try:
        halfmove = int(halfmove_clock)
        fullmove = int(fullmove_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("Indica los contadores de media jugada y número de jugada.") from exc
    if halfmove < 0 or fullmove < 1:
        raise ValueError("El contador de media jugada debe ser ≥ 0 y el número de jugada ≥ 1.")
    fen = f"{placement} {turn} {castling} {en_passant} {halfmove} {fullmove}"
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"La FEN completa no es válida: {exc}") from exc
    if not board.is_valid():
        raise ValueError(
            "La posición no es legal: revisa la colocación de las piezas y los derechos de enroque. "
            "Debe haber un rey de cada color y ninguna pieza en una casilla imposible."
        )
    return fen
