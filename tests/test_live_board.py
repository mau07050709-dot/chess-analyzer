"""Tests for stable, legal screen-position tracking."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import chess

from ajedrez.live_board import LiveBoardTracker, legal_turn_candidates
from ajedrez.recognition import RecognitionResult, SquarePrediction


def observation(board: chess.Board, confidence: float = 0.99) -> RecognitionResult:
    predictions = tuple(SquarePrediction(square, piece.symbol() if piece else None,
                                         piece.symbol() if piece else "empty", confidence)
                       for square in chess.SQUARES
                       for piece in (board.piece_at(square),))
    return RecognitionResult(board.board_fen(), "white", predictions, 0.70)


class LiveBoardTrackerTests(unittest.TestCase):
    def test_identical_position_is_ignored(self):
        tracker = LiveBoardTracker(chess.Board())
        first = tracker.observe(observation(chess.Board()))
        second = tracker.observe(observation(chess.Board()))
        self.assertEqual(first.status, "unchanged")
        self.assertEqual(second.status, "unchanged")

    def test_legal_move_waits_for_two_stable_observations_and_updates_full_fen(self):
        tracker = LiveBoardTracker(chess.Board())
        moved = chess.Board()
        moved.push_san("e4")
        self.assertEqual(tracker.observe(observation(moved)).status, "waiting")
        update = tracker.observe(observation(moved))
        self.assertEqual(update.status, "updated")
        self.assertEqual(update.move.uci(), "e2e4")
        self.assertEqual(update.board.fen(), moved.fen())
        self.assertEqual(update.board.turn, chess.BLACK)

    def test_unstable_drag_frame_cannot_commit_a_move(self):
        tracker = LiveBoardTracker(chess.Board())
        moved = chess.Board(); moved.push_san("e4")
        update = tracker.observe(observation(moved), visual_stable=False)
        self.assertEqual(update.status, "waiting")
        self.assertEqual(tracker.board.fen(), chess.STARTING_FEN)
        update = tracker.observe(observation(moved), visual_stable=True)
        self.assertEqual(update.status, "updated")

    def test_premove_or_cancelled_drag_keeps_the_confirmed_board_and_turn(self):
        tracker = LiveBoardTracker(chess.Board())
        for _ in range(3):
            update = tracker.observe(observation(chess.Board()), visual_stable=True)
        self.assertEqual(update.status, "unchanged")
        self.assertEqual(update.board.fen(), chess.STARTING_FEN)

    def test_provisional_turn_is_resolved_by_a_legal_transition(self):
        before = chess.Board(); before.push_san("e4")
        candidates = legal_turn_candidates(before.board_fen())
        self.assertEqual({candidate.turn for candidate in candidates}, {chess.WHITE, chess.BLACK})
        tracker = LiveBoardTracker(candidates[0], turn_known=False)
        self.assertFalse(tracker.turn_known)
        after = before.copy(); after.push_san("e5")
        tracker.observe(observation(after))
        update = tracker.observe(observation(after))
        self.assertEqual(update.status, "updated")
        self.assertTrue(tracker.turn_known)
        # The starting position was a midgame placement, so castling rights
        # were not observable and must not be invented during re-anchoring.
        self.assertEqual(update.board.board_fen(), after.board_fen())
        self.assertEqual(update.board.turn, after.turn)
        self.assertEqual(update.board.castling_xfen(), "-")

    def test_missed_single_opponent_reply_recovers_short_legal_sequence(self):
        tracker = LiveBoardTracker(chess.Board())
        two_moves = chess.Board()
        two_moves.push_san("e4")
        two_moves.push_san("e5")
        self.assertEqual(tracker.observe(observation(two_moves)).status, "waiting")
        update = tracker.observe(observation(two_moves))
        self.assertEqual(update.status, "updated")
        self.assertEqual(tuple(move.uci() for move in update.moves), ("e2e4", "e7e5"))
        self.assertEqual(update.board.fen(), two_moves.fen())

    def test_three_missed_plies_preserve_metadata_shared_by_all_paths(self):
        tracker = LiveBoardTracker(chess.Board())
        observed = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            observed.push_san(san)
        tracker.observe(observation(observed))
        update = tracker.observe(observation(observed))
        # Nf3 may have preceded e4. Do not invent an order, but retain the
        # turn and castling rights justified by every legal explanation.
        self.assertEqual(update.status, "resynced")
        self.assertEqual(update.board.board_fen(), observed.board_fen())
        self.assertEqual(update.board.castling_xfen(), "KQkq")
        self.assertTrue(tracker.turn_known)
        self.assertEqual(update.board.turn, observed.turn)

    def test_legal_visual_position_resynchronizes_without_fabricating_history(self):
        tracker = LiveBoardTracker(chess.Board())
        observed = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"):
            observed.push_san(san)
        # A valid position beyond the bounded history is accepted from stable,
        # high-confidence squares but unknown metadata is conservatively reset.
        tracker.max_recovery_plies = 1
        tracker.observe(observation(observed))
        update = tracker.observe(observation(observed))
        self.assertEqual(update.status, "resynced")
        self.assertEqual(update.board.board_fen(), observed.board_fen())
        self.assertEqual(update.board.castling_xfen(), "-")
        self.assertEqual(update.board.ep_square, None)
        self.assertEqual(update.board.halfmove_clock, 0)
        self.assertIn("desconocido", update.message)

    def test_uncertain_cells_retry_and_keep_full_confirmed_state(self):
        initial = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 17 42")
        tracker = LiveBoardTracker(initial)
        uncertain = observation(initial, 0.4)
        update = tracker.observe(uncertain)
        self.assertEqual(update.status, "retry")
        self.assertEqual(update.board.fen(), initial.fen())
        self.assertIn("Casillas dudosas", update.message)
        for _ in range(3):
            update = tracker.observe(uncertain)
        self.assertEqual(update.status, "review")
        self.assertEqual(update.board.fen(), initial.fen())

    def test_uncertain_capture_can_resolve_without_resetting_confirmed_metadata(self):
        initial = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 17 42")
        tracker = LiveBoardTracker(initial)
        uncertain = observation(initial, 0.4)
        self.assertEqual(tracker.observe(uncertain).status, "retry")
        stable = tracker.observe(observation(initial, 0.99))
        self.assertEqual(stable.status, "unchanged")
        self.assertEqual(stable.board.fen(), initial.fen())

    def test_castling_transition_preserves_python_chess_state(self):
        start = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3", "Bc5"):
            start.push_san(san)
        tracker = LiveBoardTracker(start)
        expected = start.copy()
        expected.push_san("O-O")
        tracker.observe(observation(expected))
        update = tracker.observe(observation(expected))
        self.assertEqual(update.status, "updated")
        self.assertEqual(update.move.uci(), "e1g1")
        self.assertEqual(update.board.fen(), expected.fen())

    def test_en_passant_and_promotion_preserve_full_python_chess_state(self):
        start = chess.Board()
        for san in ("e4", "a6", "e5", "d5"):
            start.push_san(san)
        tracker = LiveBoardTracker(start)
        expected = start.copy(); expected.push_san("exd6")
        tracker.observe(observation(expected))
        update = tracker.observe(observation(expected))
        self.assertEqual(update.status, "updated")
        self.assertEqual(update.board.fen(), expected.fen())

        start = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        tracker = LiveBoardTracker(start)
        expected = start.copy(); expected.push_san("a8=Q+")
        tracker.observe(observation(expected))
        update = tracker.observe(observation(expected))
        self.assertEqual(update.status, "updated")
        self.assertEqual(update.board.fen(), expected.fen())

    def test_impossible_position_never_replaces_last_confirmed_state(self):
        tracker = LiveBoardTracker(chess.Board())
        impossible = chess.Board(None)
        impossible.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.WHITE))
        impossible.set_piece_at(chess.E5, chess.Piece(chess.KING, chess.BLACK))
        for _ in range(2):
            update = tracker.observe(observation(impossible))
        self.assertEqual(update.status, "review")
        self.assertEqual(update.board.fen(), chess.STARTING_FEN)

    def test_ambiguous_legal_explanations_resync_with_conservative_metadata(self):
        tracker = LiveBoardTracker(chess.Board())
        observed = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            observed.push_san(san)
        tracker.observe(observation(observed))
        paths = [
            tuple(chess.Move.from_uci(uci) for uci in ("e2e4", "e7e5", "g1f3")),
            tuple(chess.Move.from_uci(uci) for uci in ("g1f3", "e7e5", "e2e4")),
        ]
        with patch.object(tracker, "_find_paths", return_value=paths):
            update = tracker.observe(observation(observed))
        self.assertEqual(update.status, "resynced")
        self.assertEqual(update.board.board_fen(), observed.board_fen())
        self.assertEqual(update.board.castling_xfen(), "KQkq")
        self.assertTrue(tracker.turn_known)
        self.assertEqual(update.board.turn, observed.turn)
        self.assertIn("historial desconocido", update.message)


if __name__ == "__main__":
    unittest.main()
