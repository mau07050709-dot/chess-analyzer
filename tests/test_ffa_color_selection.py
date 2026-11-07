"""The user's colour selector is distinct from correction of the real turn."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch
from PySide6.QtCore import QTimer,QEventLoop
from ajedrez import ffa_engine as engine
from ajedrez.ffa_live import FFATracker
from ajedrez.four_player import PLAYERS,display_cell
from ajedrez.ui.four_player_tab import FourPlayerTab
from tests.qt_test_support import test_application
from tests.test_ffa_live import observed,real_tile_board
from tests.test_ffa_gray import image,expected

class FFAColorSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def tab(self):
        tab=FourPlayerTab();tab.milliseconds.setValue(100)
        self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        return tab

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();timeout=QTimer();timeout.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None);timeout.timeout.connect(loop.quit)
        poll.start(2);timeout.start(10000)
        if not predicate():loop.exec()
        poll.stop();timeout.stop();self.assertTrue(predicate())

    def test_choose_each_color_immediately_analyzes_it_and_places_it_below(self):
        tab=self.tab();original=tab.position.to_fen4();pieces=tab.board.pieces.copy()
        for color in ('blue','yellow','green','red'):
            tab.player.setCurrentIndex(tab.player.findData(color))
            self.wait(lambda:tab.worker is None and tab.board.recommendation is not None)
            result=tab.board.recommendation
            self.assertEqual(result.player,color)
            self.assertEqual(pieces[result.origin].player,color)
            self.assertEqual(tab.board.bottom,color)
            king=next(cell for cell,p in pieces.items() if p.player==color and p.kind=='K')
            self.assertEqual(display_cell(*king,tab.board.bottom)[1],13)
            self.assertEqual(tab.position.to_fen4(),original);self.assertEqual(tab.board.pieces,pieces)

    def test_capture_orientation_does_not_override_selected_color_or_corrupt_cells(self):
        tab=self.tab();tab.player.setCurrentIndex(tab.player.findData('yellow'))
        self.wait(lambda:tab.worker is None)
        tab.reading=True;tab.tracker=FFATracker(tab.position)
        with patch('ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False):
            for captured_bottom in PLAYERS:
                for _ in range(2):
                    tab.start_recognition(real_tile_board(tab.board.pieces,captured_bottom))
                    self.wait(lambda:tab.recognition_worker is None)
                self.wait(lambda:tab.worker is None)
                self.assertEqual(tab.perspective.currentData(),captured_bottom)
                self.assertEqual(tab.board.bottom,'yellow')
                self.assertEqual(tab.board.recommendation.player,'yellow')
                self.assertEqual(tab.board.pieces,engine.pieces_of(tab.position))

    def test_real_turn_correction_is_separate_and_does_not_reselect_my_color(self):
        tab=self.tab();tab.show()
        self.assertFalse(tab.turn_details.isVisible())
        tab.turn_details_button.click();self.assertTrue(tab.turn_details.isVisible())
        tab.player.setCurrentIndex(tab.player.findData('yellow'))
        self.wait(lambda:tab.worker is None)
        tab.reading=True;tab.tracker=FFATracker(tab.position)
        tab.observed_turn.setCurrentIndex(tab.observed_turn.findData('green'))
        self.wait(lambda:tab.worker is None)
        self.assertEqual(tab.position.turn,3)
        self.assertEqual(tab.player.currentData(),'yellow')
        self.assertEqual(tab.board.bottom,'yellow')
        self.assertEqual(tab.board.recommendation.player,'yellow')

    def test_rapid_selection_cancels_old_search_and_rejects_old_or_wrong_color(self):
        tab=self.tab();entered=threading.Event();release=threading.Event()
        self.addCleanup(release.set)
        real=engine.analyze
        def blocked(position,milliseconds,cancel,progress):
            if position.turn==1:
                entered.set();release.wait(5)
            return real(position,milliseconds,cancel,progress)
        with patch('ajedrez.ui.four_player_tab.ffa_engine.analyze',side_effect=blocked):
            tab.player.setCurrentIndex(tab.player.findData('blue'));self.assertTrue(entered.wait(2))
            old_token=tab.generation
            tab.player.setCurrentIndex(tab.player.findData('red'))
            tab.player.setCurrentIndex(tab.player.findData('green'))
            self.assertTrue(tab.worker.cancel.is_set());self.assertIsNone(tab.board.recommendation)
            release.set()
            self.wait(lambda:tab.worker is None and tab.board.recommendation is not None)
        result=tab.board.recommendation
        self.assertEqual(result.player,'green');self.assertEqual(tab.board.bottom,'green')
        wrong=replace(result,player='blue')
        tab.show_result(wrong,old_token);self.assertEqual(tab.board.recommendation,result)
        tab.show_result(wrong,tab.generation);self.assertEqual(tab.board.recommendation,result)

    def test_live_selection_and_gray_frame_keep_red_when_other_players_move(self):
        tab=self.tab();tab.player.setCurrentIndex(tab.player.findData('red'))
        self.wait(lambda:tab.worker is None)
        tab.reading=True;tab.tracker=FFATracker()
        from ajedrez.ffa_recognition import recognize_ffa
        o=recognize_ffa(image('first'))
        with patch('ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False):
            for n in range(2):tab.recognition_completed(o,tab.reading_generation,str(n).encode())
        self.wait(lambda:tab.worker is None and tab.board.recommendation is not None)
        self.assertEqual(tab.board.pieces,expected('first'))
        self.assertEqual(tab.board.recommendation.player,'red');self.assertEqual(tab.board.bottom,'red')
        self.assertEqual(tab.perspective.currentData(),'blue')

    def test_pause_and_stop_do_not_restart_a_cancelled_pending_selection(self):
        tab=self.tab();entered=threading.Event();release=threading.Event();self.addCleanup(release.set)
        real=engine.analyze
        def blocked(position,milliseconds,cancel,progress):
            entered.set();release.wait(5);return real(position,milliseconds,cancel,progress)
        with patch('ajedrez.ui.four_player_tab.ffa_engine.analyze',side_effect=blocked):
            tab.player.setCurrentIndex(tab.player.findData('blue'));self.assertTrue(entered.wait(2))
            tab.player.setCurrentIndex(tab.player.findData('yellow'));tab.stop();release.set()
            self.wait(lambda:tab.worker is None)
        self.app.processEvents();self.assertIsNone(tab.board.recommendation);self.assertIsNone(tab.worker)

if __name__=='__main__':unittest.main()
