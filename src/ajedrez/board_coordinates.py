"""One bijection between real chess squares and screen rows/columns.

Orientation never changes a move, piece color, FEN or side to move.
"""
import chess


def square_to_grid(square: chess.Square, orientation: str) -> tuple[int, int]:
    """Return (column, row), with screen row zero at the top."""
    if orientation not in ("white", "black"):
        raise ValueError("La orientación debe ser 'white' o 'black'.")
    file, rank = chess.square_file(square), chess.square_rank(square)
    return (file, 7-rank) if orientation == "white" else (7-file, rank)


def grid_to_square(row: int, column: int, orientation: str) -> chess.Square:
    if orientation not in ("white", "black") or not (0 <= row < 8 and 0 <= column < 8):
        raise ValueError("Coordenadas u orientación del tablero no válidas.")
    return chess.square(column, 7-row) if orientation == "white" else chess.square(7-column, row)
