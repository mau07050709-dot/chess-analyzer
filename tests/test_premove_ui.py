"""Premove graphics, lifecycle and stale-frame boundaries in the existing UI."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import chess
from tests.qt_test_support import test_application
from tests.test_premove import Flow, SIMPLE, observation, ghost, cells_for
from ajedrez.premove import interpret_premove
from ajedrez.ui.board_widget import BoardWidget
from ajedrez.ui.main_window import MainWindow


class PremoveUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def window(self):
        w=MainWindow();self.addCleanup(w.close)
        flow=Flow(chess.Board(SIMPLE));w._live_tracker=flow.tracker;w._live_monitoring=True
        w._request_live_analysis=lambda board:None
        w._apply_live_position(flow.tracker.board,'test')
        return w

    def evidence(self,w):
        move=chess.Move.from_uci('h2h3')
        return interpret_premove(observation(ghost(w._live_tracker.board,move)),cells_for(move),
               w._live_tracker.board,w._premove.pending,epoch=w._premove.revision,
               scope=(w._live_generation,w._live_geometry_revision))

    def deliver(self,w):
        w._live_frame_is_stable=True;e=self.evidence(w)
        w._recognition_completed(e.original,('live_poll',w._live_generation),premove=e)

    def test_pending_highlights_preserve_real_board_and_arrows_in_both_views(self):
        w=self.window();before=w._live_tracker.board.fen()
        moves=list(w._live_tracker.board.legal_moves)[:5];w.board_widget.set_best_moves([m.uci() for m in moves])
        for _ in range(2):self.deliver(w)
        self.assertEqual(set(w.board_widget.premove_squares),{chess.H2,chess.H3})
        for view in ('white','black'):
            w.board_widget.set_orientation(view)
            for side in (320,640):
                w.board_widget.resize(side,side);w.board_widget.grab()
                self.assertEqual(w.board_widget.board.fen(),before)
                self.assertEqual(w.board_widget.analysis_moves,tuple(moves))
                self.assertEqual(set(w.board_widget.premove_squares),{chess.H2,chess.H3})
        self.assertTrue(w._live_tracker.turn_known)

    def test_manual_turn_changes_only_analysis_while_intent_is_pending(self):
        w=self.window();self.deliver(w);self.deliver(w)
        before=w._live_tracker.board.fen();pending=w._premove.pending
        for selection,turn in ((1,True),(2,False),(0,False)):
            w._analysis_turn_selected(selection)
            self.assertEqual(w._effective_analysis_board().turn,turn)
            self.assertEqual(w._live_tracker.board.fen(),before)
            self.assertEqual(w._premove.pending,pending)
            self.assertEqual(w.board_widget.orientation,'white')

    def test_old_generation_geometry_and_frame_cannot_replace_latest_intent(self):
        w=self.window();self.deliver(w);self.deliver(w);pending=w._premove.pending
        e=self.evidence(w)
        w._recognition_completed(e.original,('live_poll',w._live_generation-1),premove=e)
        w._live_geometry_revision+=1
        w._recognition_completed(e.original,('live_poll',w._live_generation),premove=e)
        source=SimpleNamespace(frame_id=0,last_move_highlight=None,premove_evidence=e)
        w._recognition_sequence=2
        with patch.object(w,'_recognition_completed') as completed:
            w._deliver_recognition(e.original,('live_poll',w._live_generation),source)
            completed.assert_not_called()
        self.assertEqual(w._premove.pending,pending)

    def test_pause_retains_intent_stop_and_new_game_clear_it(self):
        w=self.window();self.deliver(w);self.deliver(w);before=w._live_tracker.board.fen()
        w.live_pause_button.setEnabled(True);w.live_pause_button.click()
        self.assertTrue(w._live_paused);self.assertIsNotNone(w._premove.pending)
        self.deliver(w);self.assertEqual(w._live_tracker.board.fen(),before)
        w.live_pause_button.click();w._live_timer.stop()
        self.assertFalse(w._live_paused);self.assertIsNotNone(w._premove.pending)
        w._reset_live_game(chess.Board(),'white',True,'test')
        self.assertIsNone(w._premove.pending);self.assertEqual(w.board_widget.premove_squares,())
        w._stop_live_reading();self.assertEqual(w._premove.state,'NONE')

    def test_red_draws_beneath_pieces_and_preserves_pin_and_x_indicators(self):
        widget=BoardWidget();self.addCleanup(widget.close);board=chess.Board()
        widget.set_board(board);widget.resize(480,480)
        widget.set_pinned_squares([chess.E2]);widget.set_vulnerable_squares([chess.E2])
        before=widget.grab().toImage();widget.set_premove_squares([chess.E2,chess.E4])
        after=widget.grab().toImage()
        self.assertNotEqual(before,after)
        self.assertEqual(widget.pinned_squares,frozenset([chess.E2]))
        self.assertEqual(widget.vulnerable_squares,frozenset([chess.E2]))
        self.assertEqual(widget.board.fen(),board.fen())
        widget.set_premove_squares([]);self.assertEqual(widget.grab().toImage(),before)

    def test_voluntary_manual_correction_replaces_the_old_intent_anchor(self):
        w=self.window();self.deliver(w);self.deliver(w)
        w._live_review_pending=True
        w._recognized_position_accepted(SimpleNamespace(fen=chess.STARTING_FEN))
        w._live_timer.stop()
        self.assertIsNone(w._premove.pending)
        self.assertEqual(w.board_widget.premove_squares,())
        self.assertEqual(w._live_tracker.board.fen(),chess.STARTING_FEN)


if __name__=='__main__':unittest.main()
