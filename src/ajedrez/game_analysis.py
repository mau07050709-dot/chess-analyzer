"""Full-game review built on the existing single-position Stockfish analysis."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import chess

from .engine import AnalysisResult, UCIEngineSession
from .pgn import PGNGame


@dataclass(frozen=True)
class ClassificationThresholds:
    """Centipawn loss boundaries; values are approximate, not universal truths."""

    acceptable_max: int = 50
    inaccuracy_max: int = 150
    error_max: int = 300

    def __post_init__(self) -> None:
        if not (0 <= self.acceptable_max < self.inaccuracy_max < self.error_max):
            raise ValueError("Los umbrales deben cumplir 0 ≤ aceptable < imprecisión < error.")


DEFAULT_THRESHOLDS = ClassificationThresholds()
MATE_SCORE_CP = 100_000


@dataclass(frozen=True)
class MoveReview:
    ply: int
    move_number: int
    player: str
    move_san: str
    before_fen: str
    after_fen: str
    best_move_san: str | None
    best_move_uci: str | None
    loss_cp: int | None
    classification: str
    lost_forced_mate: bool = False
    category_id: str = "unrated"
    symbol: str = ""
    explanation: str = ""
    move_uci: str | None = None


CLASSIFICATION_META = {
    "brilliant": ("Brillante", "!!"),
    "great": ("Gran jugada", "!"),
    "best": ("Mejor jugada", "★"),
    "excellent": ("Excelente", "✓"),
    "good": ("Buena", "✓"),
    "book": ("Teoría de apertura", "📖"),
    "inaccuracy": ("Imprecisión", "?!"),
    "mistake": ("Error", "?"),
    "blunder": ("Error grave", "??"),
    "miss": ("Oportunidad perdida", "×"),
    "unrated": ("No evaluada", "—"),
}
CRITICAL_CATEGORIES = frozenset({"inaccuracy", "mistake", "blunder", "miss"})
MATERIAL_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                   chess.ROOK: 500, chess.QUEEN: 900}


@dataclass(frozen=True)
class GameAnalysisReport:
    game: PGNGame
    moves: tuple[MoveReview, ...]
    position_results: dict[str, AnalysisResult]
    cache_entries: dict[tuple, AnalysisResult]
    processed_positions: int
    total_positions: int
    cancelled: bool = False
    thresholds: ClassificationThresholds = DEFAULT_THRESHOLDS
    mode: str = "personalizado"
    analyzed_positions: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0


def classify_loss(loss_cp: int | None, thresholds: ClassificationThresholds = DEFAULT_THRESHOLDS) -> str:
    if loss_cp is None:
        return "No evaluada"
    if loss_cp <= thresholds.acceptable_max:
        return "Buena o aceptable"
    if loss_cp <= thresholds.inaccuracy_max:
        return "Imprecisión"
    if loss_cp <= thresholds.error_max:
        return "Error"
    return "Error grave"


def _score_for_mover(
    result: AnalysisResult | None, board: chess.Board, mover: chess.Color,
) -> int | None:
    white_score = _white_score_cp(result, board)
    return None if white_score is None else white_score * (1 if mover else -1)


def _material_for(board: chess.Board, color: chess.Color) -> int:
    return sum(MATERIAL_VALUES.get(piece.piece_type, 0) for piece in board.piece_map().values()
               if piece.color == color)


def classify_move(
    before: chess.Board, after: chess.Board, move: chess.Move,
    before_result: AnalysisResult | None, after_result: AnalysisResult | None,
    loss_cp: int | None, lost_forced_mate: bool,
    thresholds: ClassificationThresholds = DEFAULT_THRESHOLDS,
) -> tuple[str, str]:
    """Return a conservative category id and a factual, score-based explanation."""
    if loss_cp is None:
        return "unrated", "No hay evaluaciones completas de ambas posiciones."
    mover = before.turn
    before_score = _score_for_mover(before_result, before, mover)
    after_score = _score_for_mover(after_result, after, mover)
    best = before_result.lines[0] if before_result and before_result.lines else None
    played_best = bool(best and best.best_move_uci == move.uci())

    # A missed win is distinct from an ordinary mistake: the mover had a clear
    # winning evaluation before moving and the resulting position no longer is.
    if before_score is not None and after_score is not None:
        if ((before_score >= 200 and after_score <= 100 and loss_cp >= 100)
                or (lost_forced_mate and before_score >= MATE_SCORE_CP - 1000)):
            return "miss", "Se perdió una ventaja ganadora o un mate que el análisis encontraba."

    if played_best and loss_cp <= 20 and before_result and after_result and best:
        # Count a sacrifice only when the engine's best reply actually wins
        # material, and the measured position remains sound for the mover.
        reply_board = after.copy(stack=False)
        if after_score is not None and after_score >= -50 and before_score is not None and before_score > -150:
            reply = after_result.lines[0].best_move_uci if after_result.lines else None
            if reply:
                try:
                    reply_move = chess.Move.from_uci(reply)
                    if reply_move in reply_board.legal_moves:
                        reply_board.push(reply_move)
                        sacrificed = _material_for(before, mover) - _material_for(reply_board, mover)
                        if sacrificed >= 200:
                            return "brilliant", "La variante principal acepta un sacrificio de material y conserva una posición al menos igualada."
                except ValueError:
                    pass

    if played_best and loss_cp <= 30:
        if before_result and len(before_result.lines) > 1:
            first, second = before_result.lines[0], before_result.lines[1]
            if first.score_cp is not None and second.score_cp is not None:
                margin = (first.score_cp - second.score_cp) * (1 if mover else -1)
                if margin >= 150:
                    return "great", "La jugada fue la primera opción y la alternativa principal evaluada es claramente inferior."
        if before_score is not None and before_score <= -200 and after_score is not None and after_score >= -50:
            return "great", "La evaluación muestra que esta jugada rescata una posición que era claramente inferior."
        if loss_cp <= 10:
            return "best", "La jugada coincide con la primera opción de Stockfish."

    if loss_cp <= 25:
        return "excellent", "La evaluación cambia muy poco con esta jugada."
    if loss_cp <= thresholds.acceptable_max:
        return "good", "La pérdida de evaluación queda dentro del margen aceptable configurado."
    if loss_cp <= thresholds.inaccuracy_max:
        return "inaccuracy", "La posición empeora según la evaluación de Stockfish."
    if loss_cp <= thresholds.error_max:
        return "mistake", "La posición pierde una cantidad importante de evaluación."
    return "blunder", "La posición pierde una cantidad muy grande de evaluación."


def _white_score_cp(result: AnalysisResult | None, board: chess.Board) -> int | None:
    """Return a comparable white-perspective score, mapping mate outside cp range."""
    if board.is_checkmate():
        # A checkmated side is to move; use one ply past mate, from White's view.
        return -MATE_SCORE_CP if board.turn == chess.WHITE else MATE_SCORE_CP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if result is None or not result.lines:
        return None
    line = result.lines[0]
    if line.mate is not None:
        distance = min(abs(line.mate), MATE_SCORE_CP - 1)
        return (MATE_SCORE_CP - distance) * (1 if line.mate >= 0 else -1)
    return line.score_cp


def move_loss_cp(
    before: chess.Board,
    after: chess.Board,
    before_result: AnalysisResult | None,
    after_result: AnalysisResult | None,
) -> tuple[int | None, bool]:
    """Compare equal-method evaluations from the mover's perspective.

    Both engine scores are first expressed from White's perspective. Multiplying
    by the mover's sign prevents a Black move from being scored as if White moved.
    A delivered checkmate is recognized even though a terminal board has no PV.
    """
    mover_sign = 1 if before.turn == chess.WHITE else -1
    before_white = _white_score_cp(before_result, before)
    after_white = _white_score_cp(after_result, after)
    if before_white is None or after_white is None:
        return None, False
    before_mover = before_white * mover_sign
    after_mover = after_white * mover_sign
    loss = max(0, before_mover - after_mover)
    lost_forced_mate = (
        before_mover > -MATE_SCORE_CP + 1000
        and after_mover <= -MATE_SCORE_CP + 1000
    )
    return loss, lost_forced_mate


def _cache_key(
    fen: str, engine_path: str, seconds: float, depth: int | None,
    multipv: int, mode: str, threads: int, hash_mb: int,
) -> tuple:
    effective_time = None if depth is not None else round(seconds, 3)
    return fen, engine_path, effective_time, depth, multipv, mode, threads, hash_mb


def analyze_game(
    game: PGNGame,
    *,
    engine_path: str,
    seconds: float = 1.5,
    depth: int | None = None,
    multipv: int = 1,
    thresholds: ClassificationThresholds = DEFAULT_THRESHOLDS,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cache: dict[tuple, AnalysisResult] | None = None,
    analyzer: Callable[..., AnalysisResult] | None = None,
    session_factory: Callable[..., UCIEngineSession] | None = None,
    mode: str = "personalizado",
    threads: int = 1,
    hash_mb: int = 16,
) -> GameAnalysisReport:
    """Analyze every unique position once, then compare each move and its reply.

    The same settings are used for every position. Exact cached FEN+engine+settings
    results can be reused safely; cancellation yields an explicit partial report.
    """
    if seconds <= 0 or (depth is not None and depth < 1) or not 1 <= multipv <= 5:
        raise ValueError("El tiempo y la profundidad deben ser positivos; las variantes deben estar entre 1 y 5.")
    if threads < 1 or hash_mb < 1:
        raise ValueError("La cantidad de hilos y memoria debe ser positiva.")
    result_cache = dict(cache or {})
    position_results: dict[str, AnalysisResult] = {}
    boards = [game.board_at(ply) for ply in range(game.move_count + 1)]
    total = len(boards)
    processed = 0
    analyzed_positions = 0
    cache_hits = 0
    cancelled = False
    started = time.monotonic()
    session = None
    session_factory = session_factory or UCIEngineSession
    try:
        for board in boards:
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            fen = board.fen()
            key = _cache_key(fen, engine_path, seconds, depth, multipv, mode, threads, hash_mb)
            if key in result_cache:
                result = result_cache[key]
                cache_hits += 1
            elif board.is_game_over(claim_draw=False):
                result = None
            else:
                if analyzer is not None:
                    result = analyzer(
                        fen, engine_path=engine_path, seconds=seconds, depth=depth,
                        multipv=multipv, cancel_event=cancel_event,
                    )
                else:
                    if session is None:
                        session = session_factory(
                            engine_path, multipv=multipv, threads=threads, hash_mb=hash_mb,
                        )
                    result = session.analyze(fen, seconds=seconds, depth=depth, cancel_event=cancel_event)
                # A stopped search may return an intermediate UCI snapshot. Do not
                # classify that position or cache it as a completed evaluation.
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break
                if result is not None:
                    result_cache[key] = result
                    analyzed_positions += 1
            if result is not None:
                position_results[fen] = result
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
    finally:
        if session is not None:
            session.close()

    reviews: list[MoveReview] = []
    for ply, san in enumerate(game.san_moves, 1):
        before, after = boards[ply - 1], boards[ply]
        before_result = position_results.get(before.fen())
        after_result = position_results.get(after.fen())
        loss, lost_mate = move_loss_cp(before, after, before_result, after_result)
        best = before_result.lines[0] if before_result and before_result.lines else None
        move = before.parse_san(san)
        category_id, explanation = classify_move(
            before, after, move, before_result, after_result, loss, lost_mate, thresholds,
        )
        _category_name, symbol = CLASSIFICATION_META[category_id]
        if category_id == "unrated":
            symbol = ""
        reviews.append(MoveReview(
            ply=ply,
            move_number=before.fullmove_number,
            player="Blancas" if before.turn == chess.WHITE else "Negras",
            move_san=san,
            before_fen=before.fen(),
            after_fen=after.fen(),
            best_move_san=best.best_move_san if best else None,
            best_move_uci=best.best_move_uci if best else None,
            loss_cp=loss,
            classification=classify_loss(loss, thresholds),
            lost_forced_mate=lost_mate,
            category_id=category_id,
            symbol=symbol,
            explanation=explanation,
            move_uci=move.uci(),
        ))
    return GameAnalysisReport(
        game=game, moves=tuple(reviews), position_results=position_results,
        cache_entries=result_cache, processed_positions=processed,
        total_positions=total, cancelled=cancelled or bool(cancel_event and cancel_event.is_set()),
        thresholds=thresholds, mode=mode, analyzed_positions=analyzed_positions,
        cache_hits=cache_hits, elapsed_seconds=time.monotonic() - started,
    )
