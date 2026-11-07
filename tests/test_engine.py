"""UCI integration checks for the reusable engine module."""
import unittest

import chess

from ajedrez.engine import UCIEngineSession, _convert_lines, _parse_uci_info, analyze_position, find_engine


class EngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine_path = find_engine()
        except FileNotFoundError:
            raise unittest.SkipTest("Stockfish no está disponible para la prueba UCI")

    def test_stockfish_returns_legal_lines_for_fen(self):
        result = analyze_position(chess.STARTING_FEN, engine_path=self.engine_path, depth=5, multipv=2)
        self.assertEqual(len(result.lines), 2)
        board = chess.Board(result.fen)
        self.assertTrue(all(line.best_move_uci for line in result.lines))
        self.assertTrue(all(chess.Move.from_uci(line.best_move_uci) in board.legal_moves for line in result.lines))
        self.assertTrue(all(line.depth and line.depth >= 1 for line in result.lines))
        self.assertIn("desde las blancas", result.lines[0].evaluation_text)

    def test_persistent_session_returns_three_coherent_multipv_lines(self):
        progress = []
        with UCIEngineSession(self.engine_path, multipv=3, threads=4, hash_mb=256) as session:
            result = session.analyze(
                chess.STARTING_FEN, depth=8,
                progress_callback=lambda lines: progress.append(lines),
            )
        self.assertEqual(len(result.lines), 3)
        self.assertEqual(tuple(line.index for line in result.lines), (1, 2, 3))
        self.assertEqual(len({line.depth for line in result.lines}), 1)
        board = chess.Board(result.fen)
        self.assertTrue(all(line.best_move_uci and chess.Move.from_uci(line.best_move_uci) in board.legal_moves
                            for line in result.lines))
        self.assertTrue(any(len(lines) == 3 for lines in progress))

    def test_evaluation_is_normalized_to_white_perspective_for_cp_and_mate(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        black_view = chess.engine.PovScore(chess.engine.Cp(125), chess.BLACK)
        line = _convert_lines(board, [{"score": black_view, "pv": [move], "depth": 8}])[0]
        self.assertEqual(line.score_cp, -125)
        self.assertIn("-1.25 peones", line.evaluation_text)
        self.assertIn("ventaja negra", line.evaluation_text)

        mate_from_black = chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)
        mate_line = _convert_lines(board, [{"score": mate_from_black, "pv": [move], "depth": 8}])[0]
        self.assertEqual(mate_line.mate, -3)
        self.assertIn("Mate en 3 para negras", mate_line.evaluation_text)

    def test_invalid_explicit_engine_path_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "No existe el ejecutable"):
            find_engine("Z:/stockfish-does-not-exist.exe")

    def test_uci_info_is_converted_to_white_perspective_and_legal_san(self):
        board = chess.Board()
        board.push_san("e4")
        parsed = _parse_uci_info(board, "info depth 12 multipv 1 score cp 40 time 23 pv e7e5 g1f3")
        self.assertIsNotNone(parsed)
        index, line = parsed
        self.assertEqual(index, 1)
        self.assertEqual(line.score_cp, -40)
        self.assertEqual(line.pv_san, ("e5", "Nf3"))


if __name__ == "__main__":
    unittest.main()
