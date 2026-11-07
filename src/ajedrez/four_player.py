"""Four-colour display geometry, independent of the two-player chess state.

This is a reference layout, not a rules engine or a standard FEN converter.
Coordinates are (column, row), with row zero at the top when red is below.
"""
from dataclasses import dataclass

PLAYERS = ('red', 'blue', 'yellow', 'green')
PLAYER_NAMES = {'red': 'Rojo', 'blue': 'Azul', 'yellow': 'Amarillo', 'green': 'Verde'}
PLAYER_COLORS = {'red': '#C54450', 'blue': '#438FCA', 'yellow': '#C8A329', 'green': '#559565',
                 'dead': '#A0A0A0'}


def playable(column: int, row: int) -> bool:
    return (0 <= column < 14 and 0 <= row < 14
            and (3 <= column < 11 or 3 <= row < 11))


def display_cell(column: int, row: int, bottom: str = 'red') -> tuple[int, int]:
    if not playable(column, row):
        raise ValueError('Casilla fuera del tablero en cruz.')
    turns = PLAYERS.index(bottom)
    for _ in range(turns):
        column, row = row, 13-column
    return column, row


@dataclass(frozen=True)
class FourPiece:
    player: str
    kind: str


def initial_pieces() -> dict[tuple[int, int], FourPiece]:
    """Match the supplied image, including the four king/queen placements."""
    pieces = {}
    for offset, kind in enumerate('RNBQKBNR'):
        for player, back, pawn in (
            ('red', (3+offset,13), (3+offset,12)),
            ('blue', (0,3+offset), (1,3+offset)),
            ('yellow', (10-offset,0), (10-offset,1)),
            ('green', (13,10-offset), (12,10-offset)),
        ):
            pieces[back] = FourPiece(player, kind)
            pieces[pawn] = FourPiece(player, 'P')
    return pieces
