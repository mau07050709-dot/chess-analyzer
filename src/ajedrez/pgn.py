"""PGN file loading and immutable snapshots for simple main-line navigation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn


class PGNImportError(ValueError):
    """A PGN file is empty, unreadable, or contains a game we cannot replay."""


@dataclass(frozen=True)
class PGNGame:
    headers: dict[str, str]
    positions: tuple[chess.Board, ...]
    san_moves: tuple[str, ...]

    @property
    def move_count(self) -> int:
        return len(self.san_moves)

    def board_at(self, ply: int) -> chess.Board:
        if not 0 <= ply < len(self.positions):
            raise IndexError("El índice de jugada está fuera de la partida.")
        return self.positions[ply].copy(stack=False)

    def move_label(self, ply: int) -> str:
        """Return a numbered SAN item for a 1-based ply; index 0 is the start."""
        if not 1 <= ply <= self.move_count:
            raise IndexError("El índice de jugada está fuera de la partida.")
        board = self.positions[ply - 1]
        move_number = board.fullmove_number
        san = self.san_moves[ply - 1]
        return f"{move_number}. {san}" if board.turn == chess.WHITE else f"{move_number}... {san}"


def load_pgn_games(path: str | Path) -> tuple[PGNGame, ...]:
    """Read all main lines in a PGN file, rejecting malformed or empty input."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise PGNImportError(f"No se pudo abrir el archivo PGN: {exc}") from exc
    if not text.strip():
        raise PGNImportError("El archivo PGN está vacío.")

    import io

    stream = io.StringIO(text)
    games: list[PGNGame] = []
    game_number = 0
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception as exc:
            raise PGNImportError(f"No se pudo interpretar la partida {game_number + 1}: {exc}") from exc
        if game is None:
            break
        game_number += 1
        if game.errors:
            raise PGNImportError(f"La partida {game_number} contiene una jugada o dato inválido: {game.errors[0]}")
        try:
            board = game.board()
            if not board.is_valid():
                raise ValueError("la posición inicial no es legal")
            positions = [board.copy(stack=False)]
            san_moves: list[str] = []
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError(f"jugada ilegal: {move.uci()}")
                san_moves.append(board.san(move))
                board.push(move)
                positions.append(board.copy(stack=False))
        except (ValueError, AssertionError) as exc:
            raise PGNImportError(f"No se pudo reconstruir la partida {game_number}: {exc}") from exc
        games.append(PGNGame(dict(game.headers), tuple(positions), tuple(san_moves)))

    if not games:
        raise PGNImportError("No se encontró ninguna partida PGN válida en el archivo.")
    return tuple(games)
