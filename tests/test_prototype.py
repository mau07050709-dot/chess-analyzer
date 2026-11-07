"""Pruebas básicas de validación de entrada y descubrimiento de Stockfish."""
import unittest
from unittest.mock import patch

import chess
from ajedrez import engine


class PositionTests(unittest.TestCase):
    def test_starting_position_is_valid(self):
        self.assertTrue(chess.Board(chess.STARTING_FEN).is_valid())

    def test_san_sequence_updates_position(self):
        board = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6"):
            board.push_san(san)
        self.assertEqual(board.fullmove_number, 3)
        self.assertTrue(board.is_valid())

    def test_missing_engine_has_actionable_message(self):
        with patch.object(engine.Path, "is_file", return_value=False), patch.object(engine.shutil, "which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "STOCKFISH_EXECUTABLE"):
                engine.find_engine()


if __name__ == "__main__":
    unittest.main()
