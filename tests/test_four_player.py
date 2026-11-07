"""Four-player reference geometry, perspectives, and classic-mode isolation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from collections import Counter
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QColor
from ajedrez.four_player import PLAYERS, PLAYER_COLORS, playable, display_cell, initial_pieces
from ajedrez.ui.four_player_tab import FourPlayerBoard, colored_piece
from ajedrez.ui.main_window import MainWindow
from tests.qt_test_support import test_application


class FourPlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def test_cross_has_160_cells_and_four_missing_corners(self):
        self.assertEqual(sum(playable(c,r) for c in range(14) for r in range(14)),160)
        for c,r in ((0,0),(13,0),(0,13),(13,13),(-1,5),(14,5)):
            self.assertFalse(playable(c,r))

    def test_reference_has_all_four_armies_and_correct_king_queen_locations(self):
        pieces = initial_pieces()
        self.assertEqual(len(pieces),64)
        self.assertTrue(all(playable(*cell) for cell in pieces))
        for player in PLAYERS:
            kinds = Counter(p.kind for p in pieces.values() if p.player==player)
            self.assertEqual(kinds,Counter({'P':8,'R':2,'N':2,'B':2,'Q':1,'K':1}))
        for cell,player in (((7,13),'red'),((0,7),'blue'),((6,0),'yellow'),((13,6),'green')):
            self.assertEqual((pieces[cell].player,pieces[cell].kind),(player,'K'))

    def test_four_views_are_bijective_and_put_selected_player_at_bottom(self):
        cells={(c,r) for c in range(14) for r in range(14) if playable(c,r)}
        for player in PLAYERS:
            self.assertEqual({display_cell(c,r,player) for c,r in cells},cells)
        for player,cell in (('red',(7,13)),('blue',(0,7)),('yellow',(6,0)),('green',(13,6))):
            self.assertEqual(display_cell(*cell,player)[1],13)

    def test_changing_view_leaves_reference_position_intact(self):
        board = FourPlayerBoard()
        self.addCleanup(board.close)
        original = board.pieces.copy()
        board.resize(560,560)
        for player in PLAYERS:
            board.set_bottom(player)
            rect=board.cell_rect(7,13)
            self.assertEqual(rect.width(),40)
            self.assertEqual(board.pieces,original)
            self.assertFalse(board.grab().isNull())

    def test_every_piece_type_renders_with_its_player_color(self):
        for player in PLAYERS:
            for kind in 'KQRBNP':
                with self.subTest(player=player,kind=kind):
                    image=QImage(90,90,QImage.Format.Format_ARGB32)
                    image.fill(Qt.GlobalColor.transparent)
                    painter=QPainter(image)
                    colored_piece(kind,player).render(painter,QRectF(0,0,90,90))
                    painter.end()
                    color=QColor(PLAYER_COLORS[player])
                    self.assertGreater(sum(image.pixelColor(x,y)==color
                                           for x in range(90) for y in range(90)),200)

    def test_tab_does_not_invoke_stockfish_or_modify_classic_state(self):
        w=MainWindow()
        self.addCleanup(w.close)
        before=w.board_widget.board.fen()
        generation=w._live_generation
        with patch.object(w,'_start_analysis') as analyze:
            w.mode_tabs.setCurrentIndex(1)
            for index in range(4):
                w.four_player_tab.perspective.setCurrentIndex(index)
            w.mode_tabs.setCurrentIndex(0)
            analyze.assert_not_called()
        self.assertEqual(w.board_widget.board.fen(),before)
        self.assertEqual(w._live_generation,generation)
        self.assertEqual(w.mode_tabs.count(),2)
        self.assertEqual(w.four_player_tab.findChildren(QProgressBar),[])
        self.assertIn('FFA',w.four_player_tab.status.text())

if __name__=='__main__':unittest.main()
