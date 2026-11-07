"""Off-screen GUI smoke tests, including a real asynchronous engine request."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

import chess
from PySide6.QtCore import QEventLoop, QPoint, QTimer
from PySide6.QtGui import QImage, QPainter
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.ui.board_widget import BoardWidget
from ajedrez.ui.main_window import MainWindow
from ajedrez.ui.theme import APP_STYLESHEET, install_theme
from ajedrez.engine import AnalysisLine, AnalysisResult
from ajedrez.recognition import RecognitionResult, SquarePrediction


class DesktopUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def test_board_starts_with_initial_position_and_renders(self):
        board = BoardWidget()
        self.assertEqual(board.board.fen(), chess.STARTING_FEN)
        self.assertEqual(len(board.board.piece_map()), 32)
        image = QImage(480, 480, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        board.resize(480, 480)
        board.render(painter, QPoint())
        painter.end()
        self.assertFalse(image.isNull())

    def test_multipv_arrows_update_together_and_clear_old_moves(self):
        board = BoardWidget()
        board.resize(480, 480)
        board.set_best_moves(("e2e4", "d2d4", "g1f3"))
        self.assertEqual(tuple(move.uci() for move in board.analysis_moves),
                         ("e2e4", "d2d4", "g1f3"))
        self.assertEqual(board.best_move.uci(), "e2e4")
        image = QImage(480, 480, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        board.render(painter, QPoint())
        painter.end()
        self.assertFalse(image.isNull())
        board.set_best_moves(("d2d4",))
        self.assertEqual(tuple(move.uci() for move in board.analysis_moves), ("d2d4",))
        board.set_best_moves(())
        self.assertEqual(board.analysis_moves, ())
        self.assertIsNone(board.best_move)

    def test_review_move_marker_can_be_toggled_and_is_painted(self):
        board = BoardWidget()
        board.resize(480, 480)
        board.set_review_marker("e2e4", "??", "blunder", True)
        self.assertEqual(board.review_move.uci(), "e2e4")
        self.assertEqual(board.review_symbol, "??")
        self.assertTrue(board.review_badge_visible)
        image = QImage(480, 480, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        board.render(painter, QPoint())
        painter.end()
        self.assertFalse(image.isNull())
        board.set_review_marker("e2e4", "??", "blunder", False)
        self.assertFalse(board.review_badge_visible)

    def test_shared_theme_covers_button_states_dialogs_and_requested_labels(self):
        install_theme(self.app)
        window = MainWindow()
        self.assertEqual(window.open_pgn_button.text(), "Abrir PGN…")
        self.assertEqual(window.pgn_start_button.text(), "|< Inicio")
        self.assertEqual(window.pgn_prev_button.text(), "< Anterior")
        self.assertEqual(window.pgn_next_button.text(), "Siguiente >")
        self.assertEqual(window.pgn_end_button.text(), "Final")
        self.assertEqual(window.analyze_button.text(), "Analizar posición")
        for state in ("QPushButton:hover", "QPushButton:pressed", "QPushButton:focus", "QPushButton:disabled",
                      "QPushButton#primaryButton:disabled", "QLineEdit:focus", "QComboBox:focus"):
            self.assertIn(state, APP_STYLESHEET)
        self.assertIn("color: #F2F2F2", APP_STYLESHEET)
        self.assertIn("background: #343638", APP_STYLESHEET)
        window.close()

    def test_window_runs_engine_analysis_without_blocking_event_loop(self):
        window = MainWindow()
        window.time_spin.setValue(0.3)
        window.multipv_spin.setValue(1)
        result_loop = QEventLoop()
        window._start_analysis()
        self.assertIsNotNone(window.worker)
        worker = window.worker
        worker.completed.connect(lambda *_: result_loop.quit())
        worker.failed.connect(lambda *_: result_loop.quit())
        QTimer.singleShot(10000, result_loop.quit)
        result_loop.exec()
        self.assertIn("Mejor jugada:", window.best_move_label.text())
        self.assertNotEqual(window.board_widget.best_move, None)
        self.app.processEvents()
        self.assertTrue(window.analyze_button.isEnabled())
        window.close()
        self.app.processEvents()

    def test_stop_button_cancels_an_active_search(self):
        window = MainWindow()
        window.depth_check.setChecked(True)
        window.depth_spin.setValue(40)
        window.multipv_spin.setValue(1)
        result_loop = QEventLoop()
        window._start_analysis()
        worker = window.worker
        self.assertIsNotNone(worker)
        worker.completed.connect(lambda *_: result_loop.quit())
        worker.failed.connect(lambda *_: result_loop.quit())
        QTimer.singleShot(100, window._stop_analysis)
        QTimer.singleShot(10000, result_loop.quit)
        result_loop.exec()
        self.app.processEvents()
        self.assertEqual(window.engine_status.text(), "Análisis detenido")
        self.assertTrue(window.analyze_button.isEnabled())
        window.close()
        self.app.processEvents()

    def test_recognition_acceptance_closes_dialog_and_starts_analysis(self):
        window = MainWindow()
        board = chess.Board()
        board.push_san("e4")
        expected_fen = " ".join((board.fen().split()[0], "b", "KQkq", "-", "0", "1"))
        predictions = tuple(SquarePrediction(square, None, "empty", 1.0) for square in chess.SQUARES)
        result = RecognitionResult(board.fen().split()[0], "white", predictions)
        analysis_result = AnalysisResult(
            expected_fen,
            window.engine_path or "stockfish",
            (AnalysisLine(1, "e7e5", "e5", 18, None, 8, 0.01, ("e5",)),),
        )
        with patch("ajedrez.ui.main_window.analyze_position", return_value=analysis_result) as analyze:
            window._open_recognition_review(result)
            dialog = window._recognition_review
            self.assertIsNotNone(dialog)

            loop = QEventLoop()

            def accept_and_watch_worker():
                dialog.turn.setCurrentIndex(dialog.turn.findData("b"))
                dialog.castling.setCurrentIndex(dialog.castling.findData("KQkq"))
                dialog.en_passant.setCurrentIndex(dialog.en_passant.findData("-"))
                dialog.halfmove.setText("0")
                dialog.fullmove.setText("1")
                dialog.confirm_check.setChecked(True)
                dialog.analyze_button.click()
                self.assertIsNone(window._recognition_review)
                self.assertFalse(dialog.isVisible())
                self.assertEqual(window.fen_edit.text(), expected_fen)
                worker = window.worker
                self.assertIsNotNone(worker)
                worker.finished.connect(loop.quit)

            QTimer.singleShot(0, accept_and_watch_worker)
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            self.app.processEvents()
            analyze.assert_called_once()
            self.assertEqual(analyze.call_args.args[0], expected_fen)
            self.assertEqual(window.best_move_label.text(), "Mejor jugada: e5")
            self.assertIsNotNone(window.board_widget.best_move)
        window.close()
        self.app.processEvents()

    def test_evaluation_value_and_bar_are_visible_for_cp_and_mate(self):
        window = MainWindow()
        window.show()
        self.app.processEvents()
        self.assertEqual(window.analyze_button.text(), "Analizar posición")
        self.assertEqual(window.stop_button.text(), "Detener análisis")
        self.assertTrue(window.eval_label.isVisible())
        self.assertTrue(window.eval_bar.isVisible())

        window.eval_bar.resize(34, 220)
        cp_line = AnalysisLine(1, "e2e4", "e4", 135, None, 12, 0.5, ("e4",))
        window._render_lines((cp_line,))
        self.assertIn("+1.35 peones", window.eval_label.text())
        self.assertIn("ventaja blanca", window.eval_label.text())
        self.assertIn("desde las blancas", window.eval_label.text())
        black_fraction, white_fraction = window.eval_bar.segment_fractions()
        self.assertGreater(white_fraction, 0.5)
        self.assertAlmostEqual(black_fraction + white_fraction, 1.0)
        self.app.processEvents()
        pixels = window.eval_bar.grab().toImage()
        self.assertEqual(pixels.pixelColor(3, round(pixels.height()*.05)).name(), "#20252a")
        self.assertEqual(pixels.pixelColor(3, round(pixels.height()*.95)).name(), "#f7f7f4")

        black_line = AnalysisLine(1, "e7e5", "e5", -225, None, 12, 0.5, ("e5",))
        window._render_lines((black_line,))
        self.assertIn("-2.25 peones", window.eval_label.text())
        self.assertIn("ventaja negra", window.eval_label.text())
        black_fraction, white_fraction = window.eval_bar.segment_fractions()
        self.assertGreater(black_fraction, 0.5)
        self.assertLess(white_fraction, 0.5)

        equal_line = AnalysisLine(1, "e2e4", "e4", 3, None, 12, 0.5, ("e4",))
        window._render_lines((equal_line,))
        self.assertEqual(window.eval_bar.value(), 50)
        self.assertEqual(window.eval_bar.segment_fractions(), (0.5, 0.5))
        self.assertIn("+0.03 peones", window.eval_label.text())

        mate_line = AnalysisLine(1, "e2e4", "e4", None, -4, 12, 0.5, ("e4",))
        window._render_lines((mate_line,))
        self.assertIn("Mate en 4 para negras", window.eval_label.text())
        self.assertEqual(window.eval_bar.value(), 0)
        self.assertEqual(window.eval_bar.segment_fractions(), (1.0, 0.0))
        mate_white_line = AnalysisLine(1, "e2e4", "e4", None, 3, 12, 0.5, ("e4",))
        window._render_lines((mate_white_line,))
        self.assertIn("Mate en 3 para blancas", window.eval_label.text())
        self.assertEqual(window.eval_bar.segment_fractions(), (0.0, 1.0))
        window.close()
        self.app.processEvents()

if __name__ == "__main__":
    unittest.main()
