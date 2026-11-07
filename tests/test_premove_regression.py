"""Real ONNX/Stockfish regression: a red ghost must never replace the game."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from pathlib import Path
import chess
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtGui import QImage
from tests.qt_test_support import test_application
from tests.image_helpers import board_from_real_tiles
from tests.test_last_move_highlight import painted_highlight
from ajedrez.live_board import LiveBoardTracker
from ajedrez.ui.main_window import MainWindow

BASE='r3k2r/pppq1ppp/2n5/5N2/1b1Pp3/8/PP1BPPBP/R2QK2R b KQkq - 0 1'
RED=(174,90,57)


class PremoveRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();timer=QTimer();timer.setInterval(3)
        timer.timeout.connect(lambda:loop.quit() if predicate() else None)
        end=QTimer();end.setSingleShot(True);end.timeout.connect(loop.quit)
        timer.start();end.start(10000)
        if not predicate():loop.exec()
        timer.stop();end.stop();self.assertTrue(predicate(),'El flujo ONNX/Stockfish no terminó')

    def window(self):
        w=MainWindow();self.addCleanup(w.close)
        board=chess.Board(BASE);w._live_monitoring=True;w._live_tracker=LiveBoardTracker(board)
        w.time_spin.setValue(.15);w._apply_live_position(board,'posición real inicial')
        self.wait(lambda:len(w.board_widget.analysis_moves)==5)
        return w,board

    def feed(self,w,image):
        for _ in range(2):
            w._live_frame_is_stable=True
            w._start_recognition(image,'white','live_poll')
            self.wait(lambda:w.recognition_worker is None)

    def ghost(self,board):
        visual=board.copy();visual.remove_piece_at(chess.H2)
        visual.set_piece_at(chess.H3,chess.Piece(chess.PAWN,True))
        return painted_highlight(visual,(chess.H2,chess.H3),color=RED)

    def test_executed_premove_preserves_real_state_then_updates_engine_without_another_move(self):
        w,board=self.window();old_request=w._live_analysis_request_id
        self.feed(w,self.ghost(board))
        self.assertEqual(w._live_tracker.board.fen(),board.fen(),'El premove no es una posición confirmada')
        self.assertTrue(w._live_tracker.turn_known)
        self.assertEqual(w._live_analysis_request_id,old_request)
        final=board.copy();final.push_uci('a7a6');final.push_uci('h2h3')
        self.feed(w,board_from_real_tiles(final))
        self.wait(lambda:w.engine_status.text().endswith('análisis finalizado') and len(w.board_widget.analysis_moves)==5)
        self.assertEqual(w._live_tracker.board.fen(),final.fen())
        self.assertEqual(w._live_tracker.board.move_stack[-2:],[chess.Move.from_uci('a7a6'),chess.Move.from_uci('h2h3')])
        self.assertTrue(w._live_tracker.turn_known)
        self.assertGreater(w._live_analysis_request_id,old_request)
        self.assertTrue(all(move in final.legal_moves for move in w.board_widget.analysis_moves))

    def test_canceled_premove_keeps_opponents_real_move_and_evaluates_without_another_move(self):
        w,board=self.window();old_request=w._live_analysis_request_id
        self.feed(w,self.ghost(board))
        self.assertEqual(w._live_tracker.board.fen(),board.fen(),'El premove cancelable no debe avanzar el estado')
        final=board.copy();final.push_uci('a7a6')
        self.feed(w,board_from_real_tiles(final))
        self.wait(lambda:w.engine_status.text().endswith('análisis finalizado') and len(w.board_widget.analysis_moves)==5)
        self.assertEqual(w._live_tracker.board.fen(),final.fen())
        self.assertEqual(w._live_tracker.board.move_stack,[chess.Move.from_uci('a7a6')])
        self.assertEqual(w._live_tracker.board.turn,chess.WHITE)
        self.assertTrue(w._live_tracker.turn_known)
        self.assertGreater(w._live_analysis_request_id,old_request)
        self.assertTrue(all(move in final.legal_moves for move in w.board_widget.analysis_moves))

    def test_real_attachment_h2_g3_does_not_become_the_analyzed_fen(self):
        w,board=self.window();request=w._live_analysis_request_id
        image=QImage(str(Path(__file__).parent/'fixtures/premove_real_h2_g3.png'))
        self.assertFalse(image.isNull())
        self.feed(w,image)
        self.assertEqual(w.fen_edit.text(),board.fen())
        self.assertEqual(w._live_tracker.board.fen(),board.fen())
        self.assertTrue(w._live_tracker.turn_known)
        self.assertEqual(w._live_analysis_request_id,request)
        self.assertEqual(len(w.board_widget.analysis_moves),5)

    def test_check_cancels_premove_and_engine_evaluates_the_real_evasions(self):
        w,board=self.window();self.feed(w,self.ghost(board))
        final=board.copy();final.push_uci('b4d2')
        self.assertTrue(final.is_check())
        self.assertNotIn(chess.Move.from_uci('h2h3'),final.legal_moves)
        self.feed(w,board_from_real_tiles(final))
        self.wait(lambda:w.engine_status.text().endswith('análisis finalizado')
                  and len(w.board_widget.analysis_moves)==min(5,final.legal_moves.count()))
        self.assertEqual(w._live_tracker.board.fen(),final.fen())
        self.assertEqual(w._premove.last_outcome,'CANCELED')
        self.assertTrue(all(move in final.legal_moves for move in w.board_widget.analysis_moves))

    def test_real_opponents_move_is_analyzed_even_before_red_overlay_disappears(self):
        w,board=self.window();self.feed(w,self.ghost(board))
        opponent=board.copy();opponent.push_uci('a7a6')
        self.feed(w,self.ghost(opponent))
        self.wait(lambda:w.engine_status.text().endswith('análisis finalizado')
                  and len(w.board_widget.analysis_moves)==5)
        self.assertEqual(w._live_tracker.board.fen(),opponent.fen())
        self.assertIsNotNone(w._premove.pending)
        self.assertTrue(all(move in opponent.legal_moves for move in w.board_widget.analysis_moves))


if __name__=='__main__':unittest.main()
