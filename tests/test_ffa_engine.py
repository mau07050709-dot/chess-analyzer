"""Real native FFA integration, no fake engine results."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import unittest
from PySide6.QtCore import QEventLoop,QTimer
from ajedrez import ffa_engine as f
from ajedrez.four_player import PLAYERS,initial_pieces
from ajedrez.ui.four_player_tab import FourPlayerTab
from tests.qt_test_support import test_application


class FFAEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();end=QTimer();end.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None)
        end.timeout.connect(loop.quit);poll.start(3);end.start(10000)
        if not predicate():loop.exec()
        poll.stop();end.stop();self.assertTrue(predicate())

    def test_modern_initial_position_matches_image_and_fen4_roundtrip(self):
        board=f.new_position()
        self.assertEqual(f.pieces_of(board),initial_pieces())
        self.assertEqual(board,f.parse_position(board.to_fen4()))
        with self.assertRaises(ValueError):f.parse_position('not a FEN4')

    def test_real_search_for_each_color_returns_legal_move_without_mutation(self):
        b,gen,core,*_=f.modules()
        for seat,player in enumerate(PLAYERS):
            position=f.new_position();position.turn=seat;position.recompute_key()
            before=position.copy();result=f.analyze(position,200)
            self.assertEqual(result.player,player)
            legal={b.move_str(m) for m in gen.gen_legal(position)}
            self.assertIn(result.move,legal)
            self.assertGreater(result.depth,0)
            self.assertEqual(position,before)
        self.assertTrue(core.net_loaded())

    def test_play_advances_four_distinct_turns_with_real_legal_recommendations(self):
        position=f.new_position()
        for seat in range(4):
            self.assertEqual(position.turn,seat)
            result=f.analyze(position,100)
            position=f.play_move(position,result.move)
        self.assertEqual(position.turn,0)
        with self.assertRaises(ValueError):f.play_move(position,'a1a2')

    def test_ffa_allows_capture_of_opposite_color_not_a_teammate(self):
        b,*_=f.modules()
        position=f.new_position()
        self.assertTrue(position.is_enemy(b.make_piece(b.YELLOW,b.PAWN),b.RED))
        self.assertFalse(position.is_enemy(b.make_piece(b.RED,b.PAWN),b.RED))

    def test_cancelled_request_does_not_publish_and_library_is_reused(self):
        cancel=threading.Event();cancel.set();seen=[]
        with self.assertRaises(f.AnalysisCancelled):f.analyze(f.new_position(),100,cancel,seen.append)
        self.assertEqual(seen,[])
        f.analyze(f.new_position(),100)
        self.assertIs(f.modules(),f.modules())

    def test_gui_async_flecha_stale_results_and_four_orientations(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        tab.resize(1000,900);tab.show()
        ticks=[];timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(2)
        tab.start_analysis();self.wait(lambda:tab.worker is None);timer.stop()
        self.assertTrue(ticks)
        result=tab.board.recommendation;self.assertIsNotNone(result)
        self.assertIn(result.move,tab.status.text())
        for index in range(4):
            tab.perspective.setCurrentIndex(index)
            self.assertEqual(tab.board.recommendation,result)
            self.assertFalse(tab.board.grab().isNull())
        token=tab.generation
        tab.player.setCurrentIndex(2) # blue
        self.assertIsNone(tab.board.recommendation)
        tab.show_result(result,token)
        self.assertIsNone(tab.board.recommendation)
        tab.start_analysis();self.wait(lambda:tab.worker is None)
        self.assertEqual(tab.board.recommendation.player,'blue')
        tab.start_analysis();tab.stop();self.wait(lambda:tab.worker is None)
        self.assertIsNone(tab.board.recommendation)

if __name__=='__main__':unittest.main()
