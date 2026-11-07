import unittest
from pathlib import Path
import statistics
from unittest.mock import patch
import chess
from PySide6.QtGui import QColor, QImage
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.recognition import (
    compose_full_fen,
    recognize_position,
    recognize_position_auto_orientation,
    RecognitionResult,
)
from tests.image_helpers import reverse_cells
from ajedrez import recognition
from ajedrez.ui.capture_dialog import RecognitionReviewDialog


REAL_CHESSCOM_PLACEMENT = "rnbqkb1r/pp1ppppp/2p2n2/8/2P5/2N5/PP1PPPPP/R1BQKBNR"


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()
    def test_real_chesscom_capture_matches_known_position_without_calibration_leakage(self):
        image_path = Path(__file__).parent / "fixtures" / "chesscom_moved_position.jpg"
        image = QImage(str(image_path))
        self.assertFalse(image.isNull(), f"No se pudo abrir la captura de prueba: {image_path}")
        detected = recognize_position(image, "white")
        self.assertEqual(detected.piece_placement, REAL_CHESSCOM_PLACEMENT)
        self.assertEqual(detected.uncertain_squares, ())
        self.assertGreater(detected.minimum_confidence, 0.70)
        self.assertIsNotNone(detected.timings)
        self.assertGreater(detected.timings.inference_ms, 0)

    def test_latest_real_review_screenshot_matches_the_same_board_at_ui_scale(self):
        image_path = Path(__file__).parent / "fixtures" / "chesscom_review_capture_506x507.png"
        image = QImage(str(image_path))
        self.assertFalse(image.isNull())
        detected = recognize_position(image, "white")
        self.assertEqual(detected.piece_placement, REAL_CHESSCOM_PLACEMENT)
        self.assertEqual(detected.uncertain_squares, ())

    def test_real_tiles_reordered_for_black_view_map_back_to_same_position(self):
        image = QImage(str(Path(__file__).parent / "fixtures" / "chesscom_moved_position.jpg"))
        rotated = reverse_cells(image)
        detected = recognize_position(rotated, "black")
        self.assertEqual(detected.piece_placement, REAL_CHESSCOM_PLACEMENT)

    def test_auto_orientation_detects_black_view_with_upright_real_tiles(self):
        image = QImage(str(Path(__file__).parent / "fixtures" / "chesscom_moved_position.jpg"))
        rotated = reverse_cells(image)
        detected = recognize_position_auto_orientation(rotated)
        self.assertEqual(detected.orientation, "black")
        self.assertEqual(detected.piece_placement, REAL_CHESSCOM_PLACEMENT)

    def test_auto_orientation_uses_starting_placement_to_identify_black_view(self):
        start = RecognitionResult(chess.STARTING_FEN.split()[0], "black", ())
        wrong = RecognitionResult("RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr", "white", ())
        with patch.object(recognition, "recognize_position", side_effect=(wrong, start)):
            selected = recognize_position_auto_orientation(QImage(128, 128, QImage.Format.Format_RGB32))
        self.assertEqual(selected.orientation, "black")
        self.assertEqual(selected.piece_placement, chess.Board().board_fen())

    def test_auto_orientation_keeps_white_view_when_start_is_canonical(self):
        start = RecognitionResult(chess.Board().board_fen(), "white", ())
        wrong = RecognitionResult("RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr", "black", ())
        with patch.object(recognition, "recognize_position", side_effect=(start, wrong)):
            selected = recognize_position_auto_orientation(QImage(128, 128, QImage.Format.Format_RGB32))
        self.assertEqual(selected.orientation, "white")

    def test_warm_recognition_benchmark_reports_each_real_fixture_stage(self):
        fixture_dir = Path(__file__).parent / "fixtures"
        paths = (fixture_dir / "chesscom_moved_position.jpg",
                 fixture_dir / "chesscom_review_capture_506x507.png")
        for path in paths:
            image = QImage(str(path))
            recognize_position(image)  # warm the ONNX session before measuring
            samples = [recognize_position(image).timings for _ in range(7)]
            total = statistics.median(item.total_ms for item in samples)
            image_prep = statistics.median(item.image_prepare_ms for item in samples)
            split = statistics.median(item.board_split_ms for item in samples)
            inference = statistics.median(item.inference_ms for item in samples)
            rebuild = statistics.median(item.reconstruction_ms for item in samples)
            print(f"BENCH {path.name}: total={total:.2f}ms board_detection=0ms "
                  f"image_prep={image_prep:.2f}ms split={split:.2f}ms "
                  f"onnx={inference:.2f}ms reconstruction={rebuild:.2f}ms")
            self.assertLess(total, 1000.0)

    def test_onnx_session_is_reused_across_recognition_calls(self):
        image = QImage(str(Path(__file__).parent / "fixtures" / "chesscom_moved_position.jpg"))
        model_path = Path("models/chess-tiles-v2.onnx").resolve()
        first = recognition._load_classifier(str(model_path))
        recognize_position(image, model_path=model_path)
        recognize_position(image, model_path=model_path)
        second = recognition._load_classifier(str(model_path))
        self.assertIs(first, second)

    def test_placement_is_not_silently_promoted_to_full_fen(self):
        placement = chess.STARTING_FEN.split()[0]
        with self.assertRaises(ValueError):
            compose_full_fen(placement, "", "", "", "", "")
        fen = compose_full_fen(placement, "w", "KQkq", "-", "0", "1")
        self.assertEqual(chess.Board(fen).fen(), chess.STARTING_FEN)

    def test_invalid_board_crop_is_rejected(self):
        image = QImage(400, 300, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        with self.assertRaisesRegex(ValueError, "cuadrado"):
            recognize_position(image)

    def test_review_dialog_shows_recognition_and_leaves_hidden_fen_fields_blank(self):
        result = recognize_position(QImage(str(Path(__file__).parent / "fixtures" / "chesscom_moved_position.jpg")))
        dialog = RecognitionReviewDialog(result)
        self.assertEqual(dialog.board.board.fen().split()[0], REAL_CHESSCOM_PLACEMENT)
        self.assertIsNone(dialog.turn.currentData())
        self.assertIsNone(dialog.castling.currentData())
        self.assertIsNone(dialog.en_passant.currentData())
        self.assertEqual(dialog.halfmove.text(), "")
        self.assertEqual(dialog.fullmove.text(), "")
        dialog.close()

    def test_review_numeric_fields_remain_readable_inputs_and_feed_fen_counters(self):
        placement = chess.STARTING_FEN.split()[0]
        result = RecognitionResult(placement, "white", ())
        dialog = RecognitionReviewDialog(result)
        dialog.turn.setCurrentIndex(dialog.turn.findData("w"))
        dialog.castling.setCurrentIndex(dialog.castling.findData("KQkq"))
        dialog.en_passant.setCurrentIndex(dialog.en_passant.findData("-"))
        dialog.halfmove.setText("17")
        dialog.fullmove.setText("42")
        self.assertEqual(dialog.halfmove.text(), "17")
        self.assertEqual(dialog.fullmove.text(), "42")
        self.assertEqual(dialog.halfmove.validator().validate("17", 2)[0].name, "Acceptable")
        self.assertEqual(dialog.fullmove.validator().validate("42", 2)[0].name, "Acceptable")
        dialog.confirm_check.setChecked(True)
        dialog._accept_fen()
        self.assertEqual(dialog.fen, f"{placement} w KQkq - 17 42")
        self.assertFalse(dialog.isVisible())
        dialog.close()

    def test_live_review_prefills_metadata_from_last_confirmed_position(self):
        result = recognize_position(QImage(str(Path(__file__).parent / "fixtures" / "chesscom_moved_position.jpg")))
        board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 17 42")
        dialog = RecognitionReviewDialog(result, initial_board=board)
        self.assertEqual(dialog.turn.currentData(), "w")
        self.assertEqual(dialog.castling.currentData(), "KQkq")
        self.assertEqual(dialog.en_passant.currentData(), "d6")
        self.assertEqual(dialog.halfmove.text(), "17")
        self.assertEqual(dialog.fullmove.text(), "42")
        dialog.close()

    def test_invalid_piece_position_keeps_review_open_with_actionable_error(self):
        board = chess.Board()
        board.remove_piece_at(chess.E8)
        result = RecognitionResult(board.fen().split()[0], "white", ())
        dialog = RecognitionReviewDialog(result)
        dialog.turn.setCurrentIndex(dialog.turn.findData("w"))
        dialog.castling.setCurrentIndex(dialog.castling.findData("KQkq"))
        dialog.en_passant.setCurrentIndex(dialog.en_passant.findData("-"))
        dialog.halfmove.setText("0")
        dialog.fullmove.setText("1")
        dialog.show()
        self.app.processEvents()
        dialog.confirm_check.setChecked(True)
        dialog.analyze_button.click()
        self.app.processEvents()
        self.assertTrue(dialog.isVisible())
        self.assertIn("posición no es legal", dialog.error_label.text())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
