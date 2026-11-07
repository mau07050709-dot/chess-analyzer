"""Regressions for the actual Leer tablero en pantalla rectangular selection."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import unittest
from unittest.mock import patch
from PySide6.QtCore import QRect,Qt,QTimer,QEventLoop
from PySide6.QtGui import QImage,QPainter,QColor
from ajedrez.ffa_region import locate
from ajedrez.ffa_recognition import recognize_ffa
from ajedrez.capture.screen_capture import ScreenFrame
from ajedrez.ui.four_player_tab import FourPlayerTab
from ajedrez import ffa_engine as engine
from ajedrez.four_player import playable
from tests.qt_test_support import test_application
from tests.test_ffa_recovery import reported_board,expected_board

FIX=Path(__file__).parent/'fixtures'
MOVES=('g2g4','b9d9','e14f12','m8k8','h2h4','b7c7','j14i12','n10l9','f2f3','a8b9','g13g11')


def latest():return QImage(str(FIX/'ffa_failure_region.png')).copy(0,0,1580,873)


def latest_position():
    p=engine.new_position()
    for move in MOVES:p=engine.play_move(p,move)
    return p


class FFARegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();end=QTimer();end.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None);end.timeout.connect(loop.quit)
        poll.start(5);end.start(12000)
        if not predicate():loop.exec()
        poll.stop();end.stop();self.assertTrue(predicate())

    def test_new_real_rectangular_capture_matches_all_64_pieces_and_marks(self):
        observation=recognize_ffa(latest())
        self.assertEqual(observation.pieces,engine.pieces_of(latest_position()))
        self.assertEqual(observation.bottom,'red');self.assertFalse(observation.uncertain)
        self.assertFalse(observation.transient)
        self.assertEqual({m.player for m in observation.last_moves},{'red','blue','yellow','green'})
        self.assertEqual({s.player for s in observation.clock_samples},{'red','blue','yellow','green'})
        self.assertIsNone(observation.turn_hint) # Multiple retained pairs are not a turn proof.

    def test_previous_real_image_with_page_margin_is_located_without_changing_pieces(self):
        image=QImage(str(FIX/'ffa_failure_four_moves.png')).copy(0,0,1500,906)
        result=recognize_ffa(image)
        self.assertEqual(result.board_rect,(94,29,868,868))
        self.assertEqual(result.pieces,engine.pieces_of(expected_board()))
        self.assertFalse(result.uncertain)

    def test_cached_geometry_avoids_rescan_and_does_not_add_model_runs(self):
        image=latest();first=recognize_ffa(image)
        from ajedrez.recognition import _load_classifier
        session=_load_classifier(str((Path(__file__).parents[1]/'models/chess-tiles-v2.onnx').resolve()))
        with patch('ajedrez.ffa_region._runs',side_effect=AssertionError('unexpected relocation')):
            with patch('ajedrez.ffa_recognition._load_classifier',return_value=session) as classifier:
                result=recognize_ffa(image,board_rect=first.board_rect)
                self.assertEqual(classifier.call_count,1)
        self.assertEqual(first.pieces,result.pieces)

    def test_bad_old_geometry_is_relocated_after_source_changes_size(self):
        image=QImage(str(FIX/'ffa_failure_four_moves.png')).copy(0,0,1500,906)
        result=recognize_ffa(image,board_rect=(10,20,350,350))
        self.assertEqual(result.board_rect,(94,29,868,868))
        self.assertEqual(result.pieces,engine.pieces_of(expected_board()))

    def test_missing_arm_or_large_missing_outer_edge_is_not_padded_or_classified(self):
        for image in (reported_board().copy(0,0,868,750),reported_board().copy(100,0,768,868)):
            with patch('ajedrez.ffa_recognition._load_classifier') as classifier:
                with self.assertRaisesRegex(ValueError,'14 filas y columnas'):recognize_ffa(image)
                classifier.assert_not_called()

    def test_two_separate_crosses_in_one_selection_are_ambiguous(self):
        image=QImage(1840,900,QImage.Format.Format_RGB32);image.fill(QColor('#302e2b'))
        painter=QPainter(image);painter.drawImage(0,0,reported_board());painter.drawImage(950,0,reported_board());painter.end()
        self.assertIsNone(locate(image))

    def test_two_crosses_with_different_palettes_are_also_ambiguous(self):
        image=QImage(1840,900,QImage.Format.Format_RGB32);image.fill(QColor('#302e2b'))
        painter=QPainter(image);painter.drawImage(0,0,reported_board())
        for row in range(14):
            for col in range(14):
                if playable(col,row):painter.fillRect(950+col*62,row*62,62,62,QColor('#79917e' if (col+row)%2 else '#eee9df'))
        painter.end();self.assertIsNone(locate(image))

    def test_region_with_margin_after_resizing_still_preserves_all_pieces(self):
        image=QImage(str(FIX/'ffa_failure_four_moves.png')).copy(0,0,1500,906)
        image=image.scaled(1000,604,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
        result=recognize_ffa(image)
        self.assertEqual(result.pieces,engine.pieces_of(expected_board()))
        self.assertFalse(result.uncertain)
        # The new real screenshot clips a few outer background pixels. Its
        # fitted last boundary rounds by one pixel when resized, not a row.
        result=recognize_ffa(latest().scaled(1000,553,Qt.AspectRatioMode.IgnoreAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
        self.assertEqual(result.pieces,engine.pieces_of(latest_position()))
        self.assertFalse(result.uncertain);self.assertFalse(result.transient)

    def test_no_chess_grid_does_not_reach_onnx(self):
        image=QImage(900,600,QImage.Format.Format_RGB32);image.fill(QColor('#303030'))
        with patch('ajedrez.ffa_recognition._load_classifier') as classifier:
            with self.assertRaisesRegex(ValueError,'No se encontró'):recognize_ffa(image)
            classifier.assert_not_called()

    def test_read_screen_button_region_selection_worker_and_real_analysis(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        image=latest();frame=ScreenFrame('controlled',QRect(0,0,1580,873),image,1.)
        # Indicates the known next player, not a guessed Red bootstrap.
        tab.observed_turn.setCurrentIndex(4) # Green, after the eleven real moves.
        tab.player.setCurrentIndex(tab.player.findData('green')) # Explicit analysis colour persists.
        with patch('ajedrez.ui.four_player_tab.capture_screen',return_value=frame),patch(
                'ajedrez.ui.four_player_tab.WindowAnchor.bind',return_value=None),patch(
                'ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False):
            tab.read_button.click()
            self.wait(lambda:tab.selector is not None)
            tab.selector.selected.emit(image,frame.geometry)
            self.wait(lambda:tab.board.recommendation is not None)
            self.assertEqual(tab.position.turn,3)
            self.assertEqual(tab.board.pieces,engine.pieces_of(latest_position()))
            self.assertEqual(tab.board.recommendation.player,'green')
            self.assertIsNotNone(tab._ffa_board_rect)
            self.assertTrue(tab.reading)
            self.assertEqual(tab.observed_turn.currentIndex(),0)
            tab.stop_reading()


if __name__=='__main__':unittest.main()
