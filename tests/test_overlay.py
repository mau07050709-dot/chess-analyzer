import unittest
from types import SimpleNamespace
from unittest.mock import patch
import chess
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage
from ajedrez.engine import AnalysisLine
from ajedrez.ui.main_window import MainWindow
from ajedrez.live_board import LiveBoardTracker
from tests.qt_test_support import test_application

class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=test_application()
    def setUp(self):
        # Settings changes in these tests must not persist to the user's profile.
        settings = patch('ajedrez.ui.overlay.QSettings')
        config = settings.start()
        config.return_value.value.side_effect = lambda _key, default: default
        self.addCleanup(settings.stop)
        self.w=MainWindow(); self.tab=self.w.overlay_tab
    def tearDown(self):
        self.w.close(); self.app.processEvents()
    def ready(self):
        h=self.w
        h._live_drag_active=False;h._exploring_live=False
        h.mode_tabs.setCurrentWidget(self.tab)
        h._live_monitoring=True; h._live_paused=False
        h._live_signature_confirmed=True; h._live_region_valid=True
        h._live_screen=self.app.primaryScreen()
        h._live_region=QRect(h._live_screen.geometry().topLeft(),QRect(0,0,320,320).size())
        h._live_tracker=LiveBoardTracker(chess.Board())
        h._live_analysis_cache_key=(h._effective_analysis_board().fen(),)
        h._live_displayed_lines=tuple(AnalysisLine(i+1,m,m,0,None,8,.1,(m,)) for i,m in enumerate(
                                     ('e2e4','d2d4','g1f3','c2c4','b1c3')))
        self.tab.overlay.capture_safe=True
        self.tab.sync()
    def test_drag_keeps_confirmed_arrows_and_bar_until_a_move_is_seen(self):
        from tests.test_live_board import observation

        with patch('ajedrez.ui.overlay.time.monotonic', return_value=100.0) as clock:
            self.ready()
            self.w._live_region = QRect(50, 50, 320, 320)
            self.tab.bar_mode.setCurrentIndex(self.tab.bar_mode.findData('both'))
            self.tab.sync()
            moves = self.tab.overlay.moves
            moved = chess.Board()
            moved.push_uci('e2e4')
            token = ('live_poll', self.w._live_generation)
            source = SimpleNamespace(last_move_highlight=None)

            with patch('ajedrez.ui.main_window.left_mouse_button_down', return_value=True), \
                 patch('ajedrez.ui.main_window.capture_screen') as capture:
                self.w._live_capture_tick()
                capture.assert_not_called()
                self.assertTrue(self.w._live_drag_active)
                clock.return_value = 102.0
                self.tab.sync()
                # An in-flight recognition result must not clear the Overlay
                # while the left button is still down.
                self.w._deliver_recognition(observation(moved), token, source)
                self.tab.sync()
                self.assertTrue(self.tab.overlay.isVisible())
                self.assertTrue(self.tab.overlay.bar.isVisible())
                self.assertEqual(self.tab.overlay.moves, moves)
                self.assertEqual(self.w._live_tracker.board.fen(), chess.STARTING_FEN)

            with patch('ajedrez.ui.main_window.left_mouse_button_down', return_value=False), \
                 patch('ajedrez.ui.main_window.capture_screen'), \
                 patch('ajedrez.ui.main_window.crop_region', return_value=QImage(320, 320, QImage.Format.Format_RGB32)), \
                 patch.object(self.w, '_start_recognition') as start:
                self.w._live_capture_tick()
                start.assert_called_once()
                self.assertFalse(self.w._live_drag_active)
                self.tab.sync()
                self.assertTrue(self.tab.overlay.bar.isVisible())

                # A click or cancelled drag leaves the board unchanged.
                self.w._deliver_recognition(observation(chess.Board()), token, source)
                self.tab.sync()
                self.assertEqual(self.tab.overlay.moves, moves)
                self.assertTrue(self.tab.overlay.bar.isVisible())

                # A real move discards the old position's arrows and score.
                self.w._live_signature_confirmed = False
                self.w._deliver_recognition(observation(moved), token, source)
                self.assertFalse(self.tab.overlay.isVisible())