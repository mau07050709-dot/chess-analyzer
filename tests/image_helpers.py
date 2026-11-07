"""Image fixtures that change the board view without rotating glyphs."""
from PySide6.QtGui import QImage, QPainter
from pathlib import Path
import chess


def reverse_cells(image):
    """Simulate a platform's 180-degree VIEW: glyphs stay upright."""
    output = QImage(800, 800, QImage.Format.Format_RGB32)
    painter = QPainter(output)
    for row in range(8):
        for column in range(8):
            x0, x1 = round(column * image.width() / 8), round((column + 1) * image.width() / 8)
            y0, y1 = round(row * image.height() / 8), round((row + 1) * image.height() / 8)
            tile = image.copy(x0, y0, x1-x0, y1-y0).scaled(100, 100)
            painter.drawImage((7-column)*100, (7-row)*100, tile)
    painter.end()
    return output


def board_from_real_tiles(board, orientation="white"):
    """Synthetic position built from real Chess.com tiles, not a new screenshot."""
    original = chess.Board("rnbqkb1r/pp1ppppp/2p2n2/8/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 0 3")
    source = QImage(str(Path(__file__).parent / "fixtures/chesscom_moved_position.jpg"))
    output = QImage(800,800,QImage.Format.Format_RGB32)
    painter = QPainter(output)
    def parity(square):
        return (chess.square_rank(square)+chess.square_file(square)) % 2
    for target in chess.SQUARES:
        options = [square for square in chess.SQUARES
                   if original.piece_at(square) == board.piece_at(target)]
        chosen = next((square for square in options if parity(square)==parity(target)),options[0])
        row,column = 7-chess.square_rank(chosen),chess.square_file(chosen)
        x0,x1 = round(column*source.width()/8),round((column+1)*source.width()/8)
        y0,y1 = round(row*source.height()/8),round((row+1)*source.height()/8)
        tile = source.copy(x0,y0,x1-x0,y1-y0).scaled(100,100)
        painter.drawImage(chess.square_file(target)*100,(7-chess.square_rank(target))*100,tile)
    painter.end()
    return reverse_cells(output) if orientation == "black" else output

