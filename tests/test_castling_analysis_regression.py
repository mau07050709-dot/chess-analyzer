"""Post-castle analysis after the exact lost-rights sequence logged at 03:05."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from dataclasses import replace
from unittest.mock import patch
import chess
from PySide6.QtCore import QEventLoop, QTimer
from ajedrez.engine import AnalysisResult, find_engine
from ajedrez.live_board import LiveBoardTracker
from ajedrez.ui.main_window import MainWindow
from tests.qt_test_support import test_application
from tests.test_live_board import observation


class CastlingAnalysisRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def wait(self, predicate):
        loop = QEventLoop(); poll = QTimer(); deadline = QTimer()
        poll.setInterval(5); deadline.setSingleShot(True)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        deadline.timeout.connect(loop.quit)
        poll.start(); deadline.start(6000); loop.exec()
        poll.stop(); deadline.stop()
        self.assertTrue(predicate(), "No llegó la evaluación posterior al enroque")

    def test_real_report_unknown_rights_castle_requests_and_applies_five_lines(self):
        find_engine()
        # The actual log: a midgame reset without history, then ...d5 establishes
        # White's turn. Rights remain unknown, so the following O-O is lost.
        initial = chess.Board("r1bqk2r/pp1p1pbp/2n1p1p1/2p1P3/5P2/2NP1N2/PPP3PP/R1BQK2R w - - 0 1")
        before = initial.copy(); before.turn = chess.BLACK; before.push_uci("d7d5")
        expected = before.copy(stack=False); expected.castling_rights = chess.BB_H1
        expected.push_uci("e1g1")
        w = MainWindow()
        try:
            w._live_monitoring = True; w._live_generation = 313
            w._live_orientation = "black"; w._live_frame_is_stable = True
            w._live_tracker = LiveBoardTracker(initial, turn_known=False)
            w.time_spin.setValue(.15)
            with patch("ajedrez.ui.main_window.left_mouse_button_down", return_value=False):
                for _ in range(2):
                    w._recognition_completed(replace(observation(before), orientation="black"), ("live_poll", 313))
                self.assertTrue(w._live_tracker.turn_known)
                old_id = w._live_analysis_request_id
                old_source = w.live_analysis_worker
                # Do not wait for the previous search; castling must supersede it.
                for _ in range(2):
                    w._recognition_completed(replace(observation(expected), orientation="black"), ("live_poll", 313))
            self.assertEqual(w.board_widget.board.board_fen(), expected.board_fen())
            self.assertTrue(w._live_tracker.turn_known, "Enroque visible, pero turno provisional: se bloquea Stockfish")
            self.assertEqual(w.fen_edit.text(), expected.fen())
            self.assertEqual(w._live_tracker.board.fen(), expected.fen())
            self.assertGreater(w._live_analysis_request_id, old_id)
            self.wait(lambda: w._live_search_finalized and len(w.board_widget.analysis_moves) == 5)
            self.assertEqual(w._live_search_ticket.fen, expected.fen())
            self.assertTrue(all(m in expected.legal_moves for m in w.board_widget.analysis_moves))
            saved = (w.eval_label.text(), w.board_widget.analysis_moves)
            w._live_analysis_completed(before.fen(), AnalysisResult(before.fen(), "old", ()), old_id, old_source)
            self.assertEqual((w.eval_label.text(), w.board_widget.analysis_moves), saved)
        finally:
            w.close(); self.app.processEvents()


    def test_all_four_lost_rights_castles_update_real_stockfish_in_both_views(self):
        find_engine()
        for color in (chess.WHITE, chess.BLACK):
            for king_file in ("g", "c"):
                for orientation in ("white", "black"):
                    with self.subTest(color=color, king_file=king_file, orientation=orientation):
                        initial = chess.Board("r3k2r/pp6/8/8/8/8/PP6/R3K2R w - - 0 1")
                        before = initial.copy(); before.turn = not color
                        before.push_uci("a7a6" if color else "a2a3")
                        rank = "1" if color else "8"
                        move = chess.Move.from_uci(f"e{rank}{king_file}{rank}")
                        expected = before.copy(stack=False)
                        expected.castling_rights = chess.BB_SQUARES[chess.parse_square(("h" if king_file == "g" else "a") + rank)]
                        expected.push(move)
                        w = MainWindow()
                        try:
                            w._live_monitoring = True; w._live_generation = 314
                            w._live_orientation = orientation; w._live_frame_is_stable = True
                            w._live_tracker = LiveBoardTracker(initial, turn_known=False)
                            w.time_spin.setValue(.1)
                            with patch("ajedrez.ui.main_window.left_mouse_button_down", return_value=False):
                                for board in (before, expected):
                                    for _ in range(2):
                                        w._recognition_completed(replace(observation(board), orientation=orientation), ("live_poll", 314))
                            self.assertTrue(w._live_tracker.turn_known)
                            self.assertEqual(w.fen_edit.text(), expected.fen())
                            self.assertEqual(w.board_widget.board.fen(), expected.fen())
                            self.assertEqual(w._live_tracker.board.fen(), expected.fen())
                            self.assertEqual(w._live_tracker.board.castling_rights, 0)
                            self.wait(lambda: w._live_search_finalized and len(w.board_widget.analysis_moves) == 5)
                            self.assertTrue(all(m in expected.legal_moves for m in w.board_widget.analysis_moves))
                            self.assertEqual(w.board_widget.orientation, orientation)
                        finally:
                            w.close(); self.app.processEvents()

    def test_trusted_fen_without_rights_does_not_authorize_castling(self):
        before = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 4 10")
        tracker = LiveBoardTracker(before)
        expected = before.copy(); expected.castling_rights = chess.BB_H1; expected.push_uci("e1g1")
        self.assertIsNone(tracker._recover_observed_castle(expected.board_fen()))
        self.assertEqual(tracker.board.fen(), before.fen())

    def test_castle_probe_rejects_check_blocked_path_and_extra_changes(self):
        for fen in ("r3k2r/8/8/8/8/8/8/R3KB1R w - - 0 1",
                    "r3kr1r/8/8/8/8/8/8/R3K2R w - - 0 1"):
            with self.subTest(fen=fen):
                before = chess.Board(fen)
                tracker = LiveBoardTracker(before, turn_known=False)
                tracker._turn_candidates = {chess.WHITE}
                target = before.copy(); target.remove_piece_at(chess.E1); target.remove_piece_at(chess.H1)
                target.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
                target.set_piece_at(chess.F1, chess.Piece(chess.ROOK, chess.WHITE))
                self.assertIsNone(tracker._recover_observed_castle(target.board_fen()))
        before = chess.Board("r3k2r/pp6/8/8/8/8/PP6/R3K2R w - - 0 1")
        tracker = LiveBoardTracker(before, turn_known=False); tracker._turn_candidates = {chess.WHITE}
        target = before.copy(); target.castling_rights = chess.BB_H1; target.push_uci("e1g1")
        target.remove_piece_at(chess.A7)
        self.assertIsNone(tracker._recover_observed_castle(target.board_fen()))

    def test_known_rook_roundtrip_cannot_regain_a_castling_right(self):
        board = chess.Board("r3k2r/pp6/8/8/8/8/PP6/R3K2R w - - 0 1")
        tracker = LiveBoardTracker(board, turn_known=False)
        for uci in ("h1g1", "a7a6", "g1h1", "a6a5"):
            board.push_uci(uci)
        tracker.board = board.copy(stack=True); tracker._turn_candidates = {chess.WHITE}
        target = board.copy(stack=False); target.castling_rights = chess.BB_H1; target.push_uci("e1g1")
        self.assertIsNone(tracker._recover_observed_castle(target.board_fen()))

