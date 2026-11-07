"""A lightweight painter-based chessboard, independent of external assets."""
from __future__ import annotations

import math
import atexit
from functools import lru_cache

import chess
import chess.svg
from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, Signal, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QWidget
from ..board_coordinates import square_to_grid, grid_to_square


@lru_cache(maxsize=12)
def piece_renderer(symbol: str) -> QSvgRenderer:
    """Bundled python-chess vector pieces: opaque bodies and dark outlines."""
    renderer = QSvgRenderer(QByteArray(chess.svg.piece(chess.Piece.from_symbol(symbol)).encode("utf-8")))
    # Let Qt own these shared resources, rather than leaving native objects to
    # the unpredictable order of Python's module-cache destruction at exit.
    renderer.setParent(QApplication.instance())
    return renderer


# Release native SVG resources before PySide/Shiboken's interpreter shutdown.
# Unit-test processes do not necessarily enter QApplication.exec()/aboutToQuit.
atexit.register(piece_renderer.cache_clear)


PIECE_GLYPHS = {
    (chess.KING, chess.WHITE): "♔", (chess.QUEEN, chess.WHITE): "♕",
    (chess.ROOK, chess.WHITE): "♖", (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘", (chess.PAWN, chess.WHITE): "♙",
    (chess.KING, chess.BLACK): "♚", (chess.QUEEN, chess.BLACK): "♛",
    (chess.ROOK, chess.BLACK): "♜", (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞", (chess.PAWN, chess.BLACK): "♟",
}


class BoardWidget(QWidget):
    """Paint an 8x8 board, pieces, coordinates and an optional best-move arrow."""

    square_clicked = Signal(int)
    orientation_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        app = QApplication.instance()
        if app is not None and not getattr(app, "_chess_svg_cleanup_connected", False):
            app.aboutToQuit.connect(piece_renderer.cache_clear)
            app._chess_svg_cleanup_connected = True
        self.board = chess.Board()
        self.best_move: chess.Move | None = None
        self.analysis_moves: tuple[chess.Move, ...] = ()
        self.review_move: chess.Move | None = None
        self.review_symbol = ""
        self.review_category = ""
        self.review_badge_visible = True
        self.orientation = "white"
        self.editable = False
        self.vulnerable_squares = frozenset()
        self.pinned_squares = frozenset()
        self.premove_squares = ()
        self.animation_duration_ms = 140
        self._animated_pieces = ()
        self._animation_progress = 1.0
        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._animation.valueChanged.connect(self._advance_animation)
        self._animation.finished.connect(self._finish_animation)
        self._piece_pixmaps = {}
        self._piece_raster_size = None
        self.setMinimumSize(128, 128)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setAccessibleName("Tablero de ajedrez de 8 por 8")

    def set_position(self, fen: str) -> None:
        self.set_board(chess.Board(fen))

    def set_board(self, board: chess.Board) -> None:
        changed = board.fen() != self.board.fen()
        if changed:
            self.vulnerable_squares = frozenset()
            self.pinned_squares = frozenset()
            self.premove_squares = ()
            movers = self._known_transition(self.board, board) if not self.editable else ()
            self._animation.stop()
            self._animated_pieces = movers
            self.best_move = None
            self.analysis_moves = ()
            self.review_move = None
        self.board = board.copy(stack=False)
        if changed and self._animated_pieces and self.animation_duration_ms > 0:
            self._animation_progress = 0.0
            self._animation.setDuration(self.animation_duration_ms)
            self._animation.start()
        elif changed:
            self._finish_animation()
        self.update()

    @staticmethod
    def _known_transition(previous, current):
        """Animate only the final known legal ply, never a guessed resync path."""
        if not current.move_stack:
            return ()
        before = current.copy(stack=True)
        move = before.pop()
        if before.fen() != previous.fen() or move not in before.legal_moves:
            return ()
        movers = [(move.from_square, move.to_square, current.piece_at(move.to_square))]
        if before.is_castling(move):
            rank = chess.square_rank(move.from_square)
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            movers.append((rook_from, rook_to, current.piece_at(rook_to)))
        return tuple(movers)

    def _advance_animation(self, value):
        self._animation_progress = float(value)
        self.update()  # Qt coalesces these paints; no nested event loop or sleep.

    def _finish_animation(self):
        self._animated_pieces = ()
        self._animation_progress = 1.0
        self.update()

    def _piece_pixmap(self, piece, extent):
        ratio = self.devicePixelRatioF()
        pixels = max(1, round(extent * ratio))
        size = (pixels, ratio)
        if size != self._piece_raster_size:
            self._piece_pixmaps.clear()  # Bounded to twelve glyphs at current DPI/size.
            self._piece_raster_size = size
        symbol = piece.symbol()
        if symbol not in self._piece_pixmaps:
            pixmap = QPixmap(pixels, pixels)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            piece_renderer(symbol).render(painter, QRectF(0, 0, pixels, pixels))
            painter.end()
            pixmap.setDevicePixelRatio(ratio)
            self._piece_pixmaps[symbol] = pixmap
        return self._piece_pixmaps[symbol]

    def _paint_piece(self, painter, piece, x, y, cell):
        pixmap = self._piece_pixmap(piece, cell * .92)
        painter.drawPixmap(QRectF(x + cell*.04, y + cell*.04, cell*.92, cell*.92),
                           pixmap, QRectF(0, 0, pixmap.width(), pixmap.height()))

    def set_orientation(self, orientation: str) -> None:
        if orientation not in ("white", "black"):
            raise ValueError("La orientación debe ser 'white' o 'black'.")
        changed=self.orientation != orientation
        self.orientation = orientation
        if changed:
            self._animation.stop()
            self._finish_animation()
        if changed:self.orientation_changed.emit(orientation)
        self.update()

    def set_vulnerable_squares(self, squares, *, expected_fen=None):
        """Optional live overlay. Square identities never depend on orientation."""
        if expected_fen is not None and expected_fen != self.board.fen():
            return False
        selected = frozenset(square for square in squares
                             if square in chess.SQUARES and self.board.piece_at(square)
                             and self.board.piece_type_at(square) != chess.KING)
        if selected != self.vulnerable_squares:
            self.vulnerable_squares = selected
            self.update()
        return True

    def _vulnerability_badge_rect(self, square, left, top, cell):
        column, row = self._display_coords(square)
        for origin, destination, _piece in self._animated_pieces:
            if destination == square:
                x, y = self._display_coords(origin)
                column = x + (column-x)*self._animation_progress
                row = y + (row-y)*self._animation_progress
                break
        size = max(14., min(32., cell*.30))
        return QRectF(left+(column+.82)*cell-size/2, top+(row+.18)*cell-size/2, size, size)

    def _draw_vulnerability_badges(self, painter, left, top, cell):
        for square in self.vulnerable_squares:
            rect = self._vulnerability_badge_rect(square, left, top, cell)
            coral = QColor('#ed5357')
            border = max(2., rect.width()*.15)
            painter.setPen(QPen(coral, border))
            painter.setBrush(QColor('#ffffff'))
            painter.drawEllipse(rect.adjusted(border/2, border/2, -border/2, -border/2))
            painter.setPen(QPen(coral, max(2., rect.width()*.12),
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            x, y, size = rect.x(), rect.y(), rect.width()
            painter.drawLine(QPointF(x+size*.36,y+size*.36), QPointF(x+size*.64,y+size*.64))
            painter.drawLine(QPointF(x+size*.36,y+size*.64), QPointF(x+size*.64,y+size*.36))

    def set_pinned_squares(self, squares, *, expected_fen=None):
        if expected_fen is not None and expected_fen != self.board.fen():
            return False
        selected=frozenset(square for square in squares if square in chess.SQUARES
                           and self.board.piece_at(square) and self.board.piece_type_at(square)!=chess.KING)
        if selected!=self.pinned_squares:
            self.pinned_squares=selected
            self.update()
        return True

    def set_premove_squares(self,squares):
        selected=tuple(dict.fromkeys(s for s in squares if s in chess.SQUARES))
        selected=selected if len(selected)==2 else ()
        if selected!=self.premove_squares:
            self.premove_squares=selected
            self.update()

    def _pin_badge_rect(self, square, left, top, cell):
        rect=self._vulnerability_badge_rect(square,left,top,cell)
        # Only the pin changes side; the existing red badge remains untouched.
        if square in self.vulnerable_squares:
            rect.translate(-cell*.64,0)
        return rect

    def _draw_pin_badges(self, painter, left, top, cell):
        for square in self.pinned_squares:
            rect=self._pin_badge_rect(square,left,top,cell)
            painter.setPen(QPen(QColor('#858b90'),max(1.,rect.width()*.035)))
            painter.setBrush(QColor('#363a3d'))
            painter.drawEllipse(rect.adjusted(.5,.5,-.5,-.5))
            x,y,size=rect.x(),rect.y(),rect.width()
            # Vector pushpin: head, narrowing stem, flared base and pointed tip.
            points=[(.30,.24),(.70,.24),(.70,.34),(.62,.34),(.62,.49),
                    (.75,.61),(.75,.66),(.55,.66),(.50,.88),(.45,.66),
                    (.25,.66),(.25,.61),(.38,.49),(.38,.34),(.30,.34)]
            painter.setPen(Qt.PenStyle.NoPen);painter.setBrush(QColor('#ffffff'))
            painter.drawPolygon(QPolygonF([QPointF(x+a*size,y+b*size) for a,b in points]))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.editable or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        side = min(self.width(), self.height())
        left = (self.width() - side) / 2
        top = (self.height() - side) / 2
        cell = side / 8
        file_index = int((event.position().x() - left) / cell)
        row_index = int((event.position().y() - top) / cell)
        if 0 <= file_index < 8 and 0 <= row_index < 8:
            square = grid_to_square(row_index, file_index, self.orientation)
            self.square_clicked.emit(square)

    def set_best_move(self, uci: str | None) -> None:
        try:
            move = chess.Move.from_uci(uci) if uci else None
            self.best_move = move if move and move in self.arrow_board().legal_moves else None
            self.analysis_moves = (self.best_move,) if self.best_move else ()
        except ValueError:
            self.best_move = None
            self.analysis_moves = ()
        self.update()

    def set_best_moves(self, ucis) -> None:
        """Set up to five real MultiPV moves and redraw all arrows atomically."""
        moves: list[chess.Move] = []
        seen: set[chess.Move] = set()
        legal=self.arrow_board().legal_moves
        for uci in ucis:
            try:
                move = chess.Move.from_uci(uci) if uci else None
            except ValueError:
                continue
            if move is not None and move in legal and move not in seen:
                seen.add(move)
                moves.append(move)
            if len(moves) == 5:
                break
        self.analysis_moves = tuple(moves)
        self.best_move = moves[0] if moves else None
        self.update()

    def arrow_board(self):
        from .analysis_controls import analysis_position
        return analysis_position(self.board,getattr(self,'analysis_turn',None))

    def set_review_marker(
        self, move_uci: str | None, symbol: str = "", category_id: str = "", visible: bool = True,
        allow_post_move: bool = False,
    ) -> None:
        """Highlight the reviewed move and optionally draw its classification badge."""
        try:
            move = chess.Move.from_uci(move_uci) if move_uci else None
            self.review_move = move if move and (allow_post_move or move in self.board.legal_moves) else None
        except ValueError:
            self.review_move = None
        self.review_symbol = symbol if self.review_move else ""
        self.review_category = category_id if self.review_move else ""
        self.review_badge_visible = visible
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(self.width(), self.height())
        left = (self.width() - side) / 2
        top = (self.height() - side) / 2
        cell = side / 8
        light = QColor("#eee9df")
        dark = QColor("#79917e")
        for rank in range(8):
            for file_index in range(8):
                color = light if (rank + file_index) % 2 == 0 else dark
                painter.fillRect(int(left + file_index * cell), int(top + rank * cell), math.ceil(cell), math.ceil(cell), color)

        for square in self.premove_squares:
            column,row=self._display_coords(square)
            painter.fillRect(QRectF(left+column*cell,top+row*cell,cell,cell),QColor(180,78,47,155))

        if self.review_move:
            highlight = QColor("#f5c542")
            highlight.setAlpha(100)
            for square in (self.review_move.from_square, self.review_move.to_square):
                file_index, rank_index = self._display_coords(square)
                rect = (int(left + file_index * cell), int(top + rank_index * cell), math.ceil(cell), math.ceil(cell))
                painter.fillRect(*rect, highlight)
                painter.setPen(QPen(QColor("#8a5b00"), max(2, int(cell * 0.035))))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(*rect)

        moving_destinations = {destination for _, destination, _ in self._animated_pieces}
        for square, piece in self.board.piece_map().items():
            if square in moving_destinations:
                continue
            file_index, rank_index = self._display_coords(square)
            rect_x = left + file_index * cell
            rect_y = top + rank_index * cell
            self._paint_piece(painter, piece, rect_x, rect_y, cell)

        for origin, destination, piece in self._animated_pieces:
            start_x, start_y = self._display_coords(origin)
            end_x, end_y = self._display_coords(destination)
            progress = self._animation_progress
            self._paint_piece(painter, piece,
                              left + (start_x + (end_x-start_x)*progress)*cell,
                              top + (start_y + (end_y-start_y)*progress)*cell, cell)

        arrow_colors = (QColor("#e34234"), QColor("#2878d0"), QColor("#229447"), QColor("#e88117"), QColor("#974bd0"))
        # Long strokes first: a shorter collinear alternative keeps its colored
        # head visible on top, without shifting either real square center.
        order = sorted(range(len(self.analysis_moves[:5])), key=lambda i: (
            -((chess.square_file(self.analysis_moves[i].from_square)-chess.square_file(self.analysis_moves[i].to_square))**2
              +(chess.square_rank(self.analysis_moves[i].from_square)-chess.square_rank(self.analysis_moves[i].to_square))**2), i))
        for index in order:
            move = self.analysis_moves[index]
            start = self._center(move.from_square, left, top, cell)
            end = self._center(move.to_square, left, top, cell)
            self._draw_analysis_arrow(painter, start, end, cell, arrow_colors[index], index)

        if self.review_move and self.review_symbol and self.review_badge_visible:
            _file_index, rank_index = self._display_coords(self.review_move.to_square)
            file_index, _rank = self._display_coords(self.review_move.to_square)
            badge_size = max(22, int(cell * 0.38))
            x = left + (file_index + 1) * cell - badge_size * 0.82
            y = top + rank_index * cell - badge_size * 0.18
            colors = {
                "brilliant": ("#075e46", "#ffffff"), "great": ("#185b9b", "#ffffff"),
                "best": ("#465b9b", "#ffffff"), "excellent": ("#26734d", "#ffffff"),
                "good": ("#426c55", "#ffffff"), "inaccuracy": ("#a87400", "#ffffff"),
                "mistake": ("#b65325", "#ffffff"), "blunder": ("#a52626", "#ffffff"),
                "miss": ("#762d6e", "#ffffff"), "book": ("#315d80", "#ffffff"),
            }
            background, foreground = colors.get(self.review_category, ("#343a40", "#ffffff"))
            painter.setPen(QPen(QColor("#ffffff"), max(1, int(cell * 0.015))))
            painter.setBrush(QColor(background))
            painter.drawEllipse(int(x), int(y), badge_size, badge_size)
            badge_font = QFont("Segoe UI Symbol", max(8, int(badge_size * 0.42)), QFont.Weight.Bold)
            painter.setFont(badge_font)
            painter.setPen(QColor(foreground))
            painter.drawText(int(x), int(y), badge_size, badge_size, Qt.AlignmentFlag.AlignCenter, self.review_symbol)

        coordinate_font = QFont("Segoe UI", max(8, int(cell * 0.12)), QFont.Weight.DemiBold)
        painter.setFont(coordinate_font)
        for file_index, display_file in enumerate(range(8)):
            file_name = "abcdefgh"[display_file if self.orientation == "white" else 7 - display_file]
            color = QColor("#eee9df") if file_index % 2 == 0 else QColor("#58715f")
            painter.setPen(color)
            painter.drawText(int(left + file_index * cell + 3), int(top + 7 * cell + cell * 0.78), file_name)
        self._draw_vulnerability_badges(painter, left, top, cell)
        self._draw_pin_badges(painter, left, top, cell)
        painter.end()

    def _display_coords(self, square: chess.Square) -> tuple[int, int]:
        return square_to_grid(square, self.orientation)

    def _center(self, square: chess.Square, left: float, top: float, cell: float) -> QPointF:
        file_index, rank_index = self._display_coords(square)
        return QPointF(left + (file_index + 0.5) * cell, top + (rank_index + 0.5) * cell)

    @staticmethod
    def _draw_analysis_arrow(painter: QPainter, start: QPointF, end: QPointF, cell: float,
                             color: QColor, index: int) -> None:
        color.setAlpha(220)
        pen = QPen(color, max(1.5, cell / 60 * (9, 7, 5.5, 4, 3)[index]),
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        dx, dy = end.x() - start.x(), end.y() - start.y()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, end)
        angle = math.atan2(dy, dx)
        head = cell * (0.25, 0.23, 0.21, 0.19, 0.17)[index]
        first = QPointF(end.x() - head * math.cos(angle - math.pi / 6), end.y() - head * math.sin(angle - math.pi / 6))
        second = QPointF(end.x() - head * math.cos(angle + math.pi / 6), end.y() - head * math.sin(angle + math.pi / 6))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([end, first, second]))
