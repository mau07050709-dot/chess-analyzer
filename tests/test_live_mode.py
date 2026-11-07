"""GUI-worker tests for latest-position analysis in the live-board mode."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import unittest
from unittest.mock import patch

import chess
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QImage
from PySide6.QtCore import QRect
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.engine import AnalysisLine, AnalysisResult
from ajedrez.live_board import LiveBoardUpdate
from ajedrez.recognition import RecognitionResult, SquarePrediction
from ajedrez.ui.main_window import LiveAnalysisWorker, MainWindow


def fake_result(fen: str) -> AnalysisResult:
    board = chess.Board(fen)
    move = next(iter(board.legal_moves))
    san = board.san(move)
    return AnalysisResult(fen, "fake-stockfish", (
        AnalysisLine(1, move.uci(), san, 25, None, 10, 0.01, (san,)),
    ))


class LiveModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def test_persistent_worker_discards_superseded_search_and_reuses_one_session(self):
        first_fen = chess.STARTING_FEN
        second = chess.Board()
        second.push_san("e4")
        second_fen = second.fen()
        started = threading.Event()
        created = []

        class FakeSession:
            def __init__(self, *args, **kwargs):
                created.append((args, kwargs))
                self.closed = False

            def analyze(self, fen, *, cancel_event, **_kwargs):
                if fen == first_fen:
                    started.set()
                    cancel_event.wait(2)
                return fake_result(fen)

            def close(self):
                self.closed = True

        with patch("ajedrez.ui.main_window.UCIEngineSession", FakeSession):
            worker = LiveAnalysisWorker("fake-stockfish", multipv=1)
            received = []
            loop = QEventLoop()
            worker.completed.connect(lambda fen, result, request_id: (received.append((fen, result, request_id)), loop.quit()))
            worker.start()
            first_id = worker.request_position(first_fen, 0.2, None)
            self.assertTrue(started.wait(2))
            second_id = worker.request_position(second_fen, 0.2, None)
            QTimer.singleShot(3000, loop.quit)
            loop.exec()
            worker.stop()
            self.assertTrue(worker.wait(3000))

        self.assertEqual(len(created), 1)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], second_fen)
        self.assertEqual(received[0][2], second_id)
        self.assertNotEqual(received[0][2], first_id)

    def test_live_mode_discards_a_result_for_an_old_request(self):
        window = MainWindow()
        fen = chess.STARTING_FEN
        window.fen_edit.setText(fen)
        window._live_monitoring = True
        window._live_analysis_request_id = 7
        source = object()
        window.live_analysis_worker = source
        window._bind_live_search(fen,7,source)
        window.best_move_label.setText("Mejor jugada: esperando")
        window._live_analysis_completed(fen, fake_result(fen), 6, source)
        self.assertEqual(window.best_move_label.text(), "Mejor jugada: esperando")
        window._live_analysis_completed(fen, fake_result(fen), 7, source)
        self.assertIn("Mejor jugada:", window.best_move_label.text())
        self.assertIsNotNone(window.board_widget.best_move)
        window._live_monitoring = False
        window.live_analysis_worker = None
        window.close()

    def test_live_reading_controls_pause_and_resume_without_losing_state(self):
        window = MainWindow()
        window._live_monitoring = True
        window.live_pause_button.setEnabled(True)
        window.live_stop_button.setEnabled(True)
        window._toggle_live_reading()
        self.assertTrue(window._live_paused)
        self.assertFalse(window._live_timer.isActive())
        window._toggle_live_reading()
        self.assertFalse(window._live_paused)
        self.assertTrue(window._live_timer.isActive())
        window._stop_live_reading()
        self.assertFalse(window._live_timer.isActive())
        window.close()

    def test_transient_doubt_does_not_restart_baseline_or_pause_tracking(self):
        window = MainWindow()
        board = chess.Board()
        board.push_san("e4")
        class Tracker:
            def observe(self, _result, *, visual_stable=True):
                return LiveBoardUpdate("retry", board.copy(), message="dudosa")
        window._live_monitoring = True
        window._live_paused = False
        window._live_generation = 3
        window._live_tracker = Tracker()
        window._live_tracker.board = board
        window._live_timer.start(500)
        original_fen = window.fen_edit.text()
        result = RecognitionResult(board.board_fen(), "white", tuple(
            SquarePrediction(square, piece.symbol() if piece else None, "test", 0.5)
            for square in chess.SQUARES for piece in (board.piece_at(square),)
        ))
        window._recognition_completed(result, ("live_poll", 3))
        self.assertTrue(window._live_monitoring)
        self.assertFalse(window._live_paused)
        self.assertTrue(window._live_timer.isActive())
        self.assertFalse(window._live_setup_pending)
        self.assertEqual(window.fen_edit.text(), original_fen)
        self.assertIn("reintentará", window.live_status.text())
        window._stop_live_reading()
        window.close()

    def test_standard_start_bootstraps_without_review_and_turn_is_independent_of_view(self):
        for orientation in ("white", "black"):
            window = MainWindow()
            window._live_setup_pending = True
            window._live_generation = 5
            result = RecognitionResult(chess.Board().board_fen(), orientation, ())
            with patch.object(window, "_apply_live_position") as apply_position:
                window._recognition_completed(result, ("live_setup", 5))
            self.assertIsNone(window._recognition_review)
            self.assertEqual(window._live_orientation, orientation)
            self.assertEqual(window._live_tracker.board.fen(), chess.STARTING_FEN)
            self.assertTrue(window._live_monitoring)
            self.assertEqual(apply_position.call_args.args[0].turn, chess.WHITE)
            window._stop_live_reading()
            window.close()

    def test_midgame_bootstrap_disables_unknown_castling_and_labels_uncertain_turn(self):
        board = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            board.push_san(san)
        window = MainWindow()
        window._live_setup_pending = True
        window._live_generation = 8
        result = RecognitionResult(board.board_fen(), "black", ())
        with patch.object(window, "_apply_live_position") as apply_position:
            window._recognition_completed(result, ("live_setup", 8))
        live_board = apply_position.call_args.args[0]
        message = apply_position.call_args.args[1]
        self.assertEqual(window._live_orientation, "black")
        self.assertEqual(live_board.castling_xfen(), "-")
        self.assertIn("provisional", message)
        self.assertEqual(live_board.turn, chess.WHITE)  # explicit conservative default
        self.assertIsNone(window._recognition_review)
        window._stop_live_reading()
        window.close()

    def test_provisional_turn_is_not_sent_to_stockfish(self):
        window = MainWindow()
        board = chess.Board()
        board.push_san("e4")
        with patch.object(window, "_request_live_analysis") as analyze:
            window._apply_live_position(board, "turno sin verificar", turn_known=False)
        analyze.assert_not_called()
        self.assertIn("provisional", window.eval_label.text())
        self.assertEqual(window.board_widget.analysis_moves, ())
        window.close()

    def test_capture_signature_skips_identical_frames_but_changes_for_new_pixels(self):
        from pathlib import Path
        fixture_dir = Path(__file__).parent / "fixtures"
        image = QImage(str(fixture_dir / "chesscom_moved_position.jpg"))
        other = QImage(str(fixture_dir / "chesscom_review_capture_506x507.png"))
        first = MainWindow._image_signature(image)
        self.assertEqual(first, MainWindow._image_signature(image.copy()))
        self.assertNotEqual(first, MainWindow._image_signature(other))

    def test_same_pending_frame_is_reclassified_for_stability_but_confirmed_frame_is_skipped(self):
        window = MainWindow()
        image = QImage(128, 128, QImage.Format.Format_RGB32)
        image.fill(0xFFEEEEEE)
        window._live_monitoring = True
        window._live_screen = object()
        window._live_region = QRect(0, 0, 128, 128)
        window._live_orientation = "white"
        window._live_last_signature = None
        started = []
        with patch("ajedrez.ui.main_window.capture_screen", return_value=object()), \
             patch("ajedrez.ui.main_window.crop_region", return_value=image), \
             patch.object(window, "_start_recognition", side_effect=lambda *args: started.append(args)):
            window._live_capture_tick()
            window._live_capture_tick()
            self.assertEqual(len(started), 2)
            window._live_signature_confirmed = True
            window._live_capture_tick()
            self.assertEqual(len(started), 2)
        window._stop_live_reading()
        window.close()

    def test_held_mouse_ignores_drag_frames_and_keeps_last_board(self):
        window = MainWindow()
        confirmed = chess.Board()
        window._live_monitoring = True
        window._live_screen = object()
        window._live_region = QRect(0, 0, 128, 128)
        window._live_tracker = type("Tracker", (), {"discard_pending_candidate": lambda self: None})()
        window._live_tracker.board = confirmed
        window._live_signature_confirmed = True
        started = []
        with patch("ajedrez.ui.main_window.left_mouse_button_down", return_value=True), \
             patch("ajedrez.ui.main_window.capture_screen") as capture, \
             patch.object(window, "_start_recognition", side_effect=lambda *args: started.append(args)):
            window._live_capture_tick()
        capture.assert_not_called()
        self.assertEqual(started, [])
        self.assertEqual(window._live_tracker.board.fen(), chess.STARTING_FEN)
        self.assertTrue(window._live_drag_active)
        window._stop_live_reading()
        window.close()


if __name__ == "__main__":
    unittest.main()
