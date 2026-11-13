import unittest

import chess
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from ajedrez.board_coordinates import square_to_grid
from ajedrez.capture.move_dots import without_move_dots


class MoveDotsTests(unittest.TestCase):
    cell = 60

    def make_board_image(self, orientation, *, selected=True, piece_arrived=False):
        image = QImage(self.cell * 8, self.cell * 8, QImage.Format.Format_RGB32)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(8):
            for column in range(8):
                color = "#ebecd0" if (row + column) % 2 == 0 else "#739552"
                painter.fillRect(column * self.cell, row * self.cell,
                                 self.cell, self.cell, QColor(color))

        column, row = square_to_grid(chess.E2, orientation)
        if selected:
            painter.fillRect(column * self.cell, row * self.cell,
                             self.cell, self.cell, QColor("#b9ca43"))
        if not piece_arrived:
            painter.fillRect(column * self.cell + 20, row * self.cell + 14,
                             20, 34, QColor("#fafafa"))
        painter.setBrush(QColor(0, 0, 0, 140))
        for square in (chess.E3, chess.E4):
            column, row = square_to_grid(square, orientation)
            center = QPoint(column * self.cell + 30, row * self.cell + 30)
            if piece_arrived and square == chess.E4:
                # A pawn head and its wider body are not an empty-square hint.
                painter.setBrush(QColor("#fafafa"))
                painter.drawEllipse(center - QPoint(0, 8), 9, 9)
                painter.drawRoundedRect(QRect(center.x() - 14, center.y(),
                                               28, 21), 4, 4)
            else:
                painter.drawEllipse(center, 10, 10)
        painter.end()
        return image

    def test_selected_pawn_dots_are_removed_in_both_orientations(self):
        for orientation in ("white", "black"):
            with self.subTest(orientation=orientation):
                marked = self.make_board_image(orientation)
                cleaned = without_move_dots(marked, chess.Board(), orientation)
                self.assertIsNot(cleaned, marked)
                for square in (chess.E3, chess.E4):
                    x, y = square_to_grid(square, orientation)
                    center = QPoint(x * self.cell + 30, y * self.cell + 30)
                    background = QPoint(x * self.cell + 7, y * self.cell + 7)
                    self.assertNotEqual(marked.pixelColor(center), marked.pixelColor(background))
                    self.assertEqual(cleaned.pixelColor(center), marked.pixelColor(background))
                x, y = square_to_grid(chess.E2, orientation)
                self.assertEqual(cleaned.copy(x * self.cell, y * self.cell, self.cell, self.cell),
                                 marked.copy(x * self.cell, y * self.cell, self.cell, self.cell))

    def test_incomplete_pattern_preserves_move_or_unrelated_annotation(self):
        for orientation in ("white", "black"):
            for options in ({"selected": False}, {"piece_arrived": True}):
                with self.subTest(orientation=orientation, options=options):
                    image = self.make_board_image(orientation, **options)
                    self.assertIs(without_move_dots(image, chess.Board(), orientation), image)
