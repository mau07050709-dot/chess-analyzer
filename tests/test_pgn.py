"""PGN import and navigation regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

import chess
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.engine import AnalysisLine, AnalysisResult
from ajedrez.pgn import PGNImportError, load_pgn_games
from ajedrez.ui.main_window import MainWindow


VALID_PGN = '''[Event "Mini match"]
[Site "Local"]
[Date "2026.10.23"]
[Round "1"]
[White "Ada"]
[Black "Turing"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
'''


class PGNImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def _write(self, text: str) -> Path:
        temp = tempfile.NamedTemporaryFile("w", suffix=".pgn", encoding="utf-8", delete=False)
        temp.write(text)
        temp.close()
        self.addCleanup(Path(temp.name).unlink, missing_ok=True)
        return Path(temp.name)

    def test_loads_metadata_mainline_and_navigable_positions(self):
        game = load_pgn_games(self._write(VALID_PGN))[0]
        self.assertEqual(game.headers["White"], "Ada")
        self.assertEqual(game.move_count, 4)
        self.assertEqual(game.move_label(1), "1. e4")
        self.assertEqual(game.move_label(2), "1... e5")
        self.assertEqual(game.board_at(0).fen(), chess.STARTING_FEN)
        board = game.board_at(2)
        self.assertEqual(board.piece_at(chess.E4), chess.Piece(chess.PAWN, chess.WHITE))
        self.assertEqual(board.piece_at(chess.E5), chess.Piece(chess.PAWN, chess.BLACK))
        self.assertEqual(board.turn, chess.WHITE)

    def test_custom_initial_fen_preserves_rights_ep_and_move_counters(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 7 23"
        text = f'''[Event "Custom"]\n[SetUp "1"]\n[FEN "{fen}"]\n[White "White"]\n[Black "Black"]\n[Result "*"]\n\n23... O-O *\n'''
        game = load_pgn_games(self._write(text))[0]
        start = game.board_at(0)
        self.assertEqual(start.fen(), fen)
        end = game.board_at(1)
        self.assertEqual(end.fullmove_number, 24)
        self.assertFalse(end.has_kingside_castling_rights(chess.BLACK))
        self.assertEqual(end.king(chess.BLACK), chess.G8)

    def test_promotion_and_en_passant_fields_survive_navigation(self):
        promotion_fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
        promotion_pgn = f'''[SetUp "1"]\n[FEN "{promotion_fen}"]\n[Result "*"]\n\n1. a8=Q+ *\n'''
        promotion = load_pgn_games(self._write(promotion_pgn))[0]
        self.assertEqual(promotion.board_at(1).piece_at(chess.A8), chess.Piece(chess.QUEEN, chess.WHITE))

        ep_fen = "4k3/8/8/8/3pP3/8/8/4K3 b - e3 4 20"
        ep_pgn = f'''[SetUp "1"]\n[FEN "{ep_fen}"]\n[Result "*"]\n\n20... dxe3 *\n'''
        en_passant = load_pgn_games(self._write(ep_pgn))[0]
        self.assertEqual(en_passant.board_at(0).ep_square, chess.E3)
        self.assertEqual(en_passant.board_at(1).piece_at(chess.E3), chess.Piece(chess.PAWN, chess.BLACK))

    def test_empty_and_invalid_pgn_report_clear_errors(self):
        with self.assertRaisesRegex(PGNImportError, "vacío"):
            load_pgn_games(self._write(" \n"))
        invalid = '[Event "Broken"]\n[Result "*"]\n\n1. e5 *\n'
        with self.assertRaisesRegex(PGNImportError, "inválid"):
            load_pgn_games(self._write(invalid))

    def test_multiple_games_are_returned_for_selection(self):
        games = load_pgn_games(self._write(VALID_PGN + "\n" + VALID_PGN.replace("Ada", "Grace")))
        self.assertEqual(len(games), 2)
        self.assertEqual(games[1].headers["White"], "Grace")

    def test_ui_selects_a_move_and_updates_board_and_turn(self):
        window = MainWindow()
        game = load_pgn_games(self._write(VALID_PGN))[0]
        window._set_pgn_game(game, "mini.pgn")
        window.eval_label.setText("Evaluación: resultado anterior")
        window.pgn_moves.setCurrentRow(2)
        self.assertEqual(window._pgn_ply, 2)
        self.assertEqual(window.board_widget.board.fen(), game.board_at(2).fen())
        self.assertEqual(window.pgn_turn_label.text(), "Turno: Blancas")
        self.assertNotEqual(window.eval_label.text(), "Evaluación: resultado anterior")
        self.assertIn("analiza esta posición", window.eval_label.text())
        window._navigate_pgn(0)
        self.assertEqual(window.board_widget.board.fen(), chess.STARTING_FEN)
        window._navigate_pgn(game.move_count)
        self.assertEqual(window.board_widget.board.fen(), game.board_at(4).fen())
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_old_analysis_snapshots_are_ignored_after_position_changes(self):
        window = MainWindow()
        window._analysis_generation = 3
        window.best_move_label.setText("Mejor jugada: posición nueva")
        line = AnalysisLine(1, "e2e4", "e4", 25, None, 8, 0.1, ("e4",))
        result = AnalysisResult(chess.STARTING_FEN, window.engine_path or "stockfish", (line,))
        window._show_progress((line,), 2)
        window._analysis_completed(result, False, 2)
        window._analysis_failed("fallo antiguo", 2)
        self.assertEqual(window.best_move_label.text(), "Mejor jugada: posición nueva")
        window.close()
        window.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
