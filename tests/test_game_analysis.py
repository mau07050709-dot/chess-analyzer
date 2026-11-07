"""Unit and integration tests for whole-PGN review and mover-relative loss."""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import chess
from PySide6.QtCore import QEventLoop, QTimer
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.engine import AnalysisLine, AnalysisResult, find_engine
from ajedrez.game_analysis import (
    CLASSIFICATION_META,
    ClassificationThresholds,
    analyze_game,
    classify_move,
    classify_loss,
    move_loss_cp,
)
from ajedrez.pgn import load_pgn_games
from ajedrez.ui.main_window import MainWindow
from ajedrez.ui.game_report_dialog import ClassificationSettingsDialog, GameReportDialog


PGN = '''[Event "Review test"]
[White "Ada"]
[Black "Turing"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
'''


def make_game(text: str = PGN):
    handle = tempfile.NamedTemporaryFile("w", suffix=".pgn", encoding="utf-8", delete=False)
    try:
        handle.write(text)
        handle.close()
        return load_pgn_games(handle.name)[0]
    finally:
        Path(handle.name).unlink(missing_ok=True)


def result_for(board: chess.Board, cp: int | None, mate: int | None = None) -> AnalysisResult:
    move = next(iter(board.legal_moves), None)
    san = board.san(move) if move else "sin jugada"
    line = AnalysisLine(1, move.uci() if move else None, san, cp, mate, 8, 0.02, (san,) if move else ())
    return AnalysisResult(board.fen(), "test-engine", (line,))


class GameAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def test_white_and_black_moves_use_the_mover_perspective(self):
        game = make_game()
        scores = (0, 200, 400, 350, 350)
        by_fen = {game.board_at(index).fen(): result_for(game.board_at(index), score)
                  for index, score in enumerate(scores)}
        calls = []

        def fake_analyze(fen, **_kwargs):
            calls.append(fen)
            return by_fen[fen]

        report = analyze_game(game, engine_path="stockfish", analyzer=fake_analyze, depth=8)
        self.assertEqual(len(report.moves), 4)
        self.assertEqual([move.player for move in report.moves], ["Blancas", "Negras", "Blancas", "Negras"])
        self.assertEqual(report.moves[0].loss_cp, 0)
        self.assertEqual(report.moves[0].classification, "Buena o aceptable")
        self.assertEqual(report.moves[1].loss_cp, 200)
        self.assertEqual(report.moves[1].classification, "Error")
        self.assertEqual(report.moves[2].loss_cp, 50)
        self.assertEqual(report.moves[2].classification, "Buena o aceptable")
        self.assertEqual(report.moves[3].loss_cp, 0)
        self.assertEqual(len(calls), 5)
        self.assertEqual(report.processed_positions, report.total_positions)

    def test_scores_that_favor_black_are_not_sign_flipped_into_white_blunders(self):
        game = make_game(PGN.replace("1. e4 e5 2. Nf3 Nc6", "1. e4 e5"))
        white_scores = (-100, -250, -200)
        by_fen = {game.board_at(index).fen(): result_for(game.board_at(index), score)
                  for index, score in enumerate(white_scores)}
        report = analyze_game(game, engine_path="stockfish",
                              analyzer=lambda fen, **_kwargs: by_fen[fen], depth=6)
        # White worsens an already Black-favored position; Black's reply improves it.
        self.assertEqual(report.moves[0].loss_cp, 150)
        self.assertEqual(report.moves[0].classification, "Imprecisión")
        self.assertEqual(report.moves[1].loss_cp, 50)
        self.assertEqual(report.moves[1].classification, "Buena o aceptable")

    def test_black_move_category_uses_the_black_mover_perspective(self):
        before = chess.Board()
        before.push_san("e4")
        move = before.parse_san("e5")
        after = before.copy()
        after.push(move)
        best = AnalysisResult(
            before.fen(), "test", (AnalysisLine(1, move.uci(), "e5", 100, None, 12, 0.1, ("e5",)),)
        )
        category, _reason = classify_move(
            before, after, move, best, result_for(after, 250), 150, False,
        )
        self.assertEqual(category, "inaccuracy")

    def test_losing_a_forced_mate_is_classified_as_a_missed_opportunity(self):
        before = chess.Board()
        move = before.parse_san("e4")
        after = before.copy()
        after.push(move)
        category, _reason = classify_move(
            before, after, move, result_for(before, None, 2), result_for(after, None, -2),
            199_996, True,
        )
        self.assertEqual(category, "miss")

    def test_classification_thresholds_are_configurable_and_cover_each_band(self):
        thresholds = ClassificationThresholds(20, 80, 200)
        self.assertEqual(classify_loss(None, thresholds), "No evaluada")
        self.assertEqual(classify_loss(20, thresholds), "Buena o aceptable")
        self.assertEqual(classify_loss(21, thresholds), "Imprecisión")
        self.assertEqual(classify_loss(81, thresholds), "Error")
        self.assertEqual(classify_loss(201, thresholds), "Error grave")
        with self.assertRaises(ValueError):
            ClassificationThresholds(100, 50, 200)

    def test_all_ordinary_move_categories_have_stable_ids_and_symbols(self):
        before = chess.Board()
        move = before.parse_san("e4")
        after = before.copy()
        after.push(move)
        before_result = AnalysisResult(
            before.fen(), "test", (AnalysisLine(1, move.uci(), "e4", 0, None, 8, 0.02, ("e4",)),)
        )
        after_result = result_for(after, 0)
        expected = ((0, "best"), (20, "excellent"), (40, "good"),
                    (100, "inaccuracy"), (200, "mistake"), (400, "blunder"))
        for loss, category in expected:
            with self.subTest(category=category):
                found, _reason = classify_move(before, after, move, before_result,
                                               after_result, loss, False)
                self.assertEqual(found, category)
                self.assertTrue(CLASSIFICATION_META[found][1])

    def test_best_move_alone_is_not_brilliant_or_great(self):
        before = chess.Board()
        move = before.parse_san("e4")
        after = before.copy()
        after.push(move)
        before_result = AnalysisResult(
            before.fen(), "test", (AnalysisLine(1, move.uci(), "e4", 0, None, 12, 0.1, ("e4",)),)
        )
        category, _ = classify_move(before, after, move, before_result,
                                   result_for(after, 0), 0, False)
        self.assertEqual(category, "best")

    def test_brilliant_requires_a_sound_engine_line_that_confirms_material_sacrifice(self):
        before = chess.Board("6k1/6p1/8/7Q/8/8/8/6K1 w - - 0 1")
        move = chess.Move.from_uci("h5h7")
        self.assertIn(move, before.legal_moves)
        after = before.copy()
        after.push(move)
        capture = chess.Move.from_uci("g8h7")
        self.assertIn(capture, after.legal_moves)
        before_result = AnalysisResult(
            before.fen(), "test", (AnalysisLine(1, move.uci(), "Qh7+", 0, None, 20, 0.1, ("Qh7+",)),)
        )
        after_result = AnalysisResult(
            after.fen(), "test", (AnalysisLine(1, capture.uci(), "Kxh7", 0, None, 20, 0.1, ("Kxh7",)),)
        )
        category, reason = classify_move(before, after, move, before_result, after_result, 0, False)
        self.assertEqual(category, "brilliant")
        self.assertIn("sacrificio", reason)

    def test_great_move_needs_a_large_engine_gap_to_next_line(self):
        before = chess.Board()
        move = before.parse_san("e4")
        after = before.copy()
        after.push(move)
        lines = (
            AnalysisLine(1, move.uci(), "e4", 40, None, 16, 0.1, ("e4",)),
            AnalysisLine(2, "d2d4", "d4", -150, None, 16, 0.1, ("d4",)),
        )
        category, reason = classify_move(
            before, after, move, AnalysisResult(before.fen(), "test", lines),
            result_for(after, 0), 0, False,
        )
        self.assertEqual(category, "great")
        self.assertIn("alternativa", reason)

    def test_missed_winning_opportunity_is_distinct_from_an_ordinary_blunder(self):
        before = chess.Board()
        move = before.parse_san("e4")
        after = before.copy()
        after.push(move)
        category, reason = classify_move(
            before, after, move, result_for(before, 350), result_for(after, 50), 300, False,
        )
        self.assertEqual(category, "miss")
        self.assertIn("ventaja ganadora", reason)

    def test_book_is_not_assigned_without_an_opening_source(self):
        game = make_game()
        report = analyze_game(
            game, engine_path="stockfish", analyzer=lambda fen, **_kwargs: result_for(chess.Board(fen), 0)
        )
        self.assertNotIn("book", {move.category_id for move in report.moves})

    def test_threshold_settings_dialog_returns_user_selected_bands(self):
        dialog = ClassificationSettingsDialog(ClassificationThresholds())
        dialog.acceptable.setValue(0.75)
        dialog.inaccuracy.setValue(1.25)
        dialog.error.setValue(2.75)
        dialog._accept_values()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        self.assertEqual(dialog.thresholds, ClassificationThresholds(75, 125, 275))
        dialog.deleteLater()
        self.app.processEvents()

    def test_mate_scores_and_terminal_checkmate_are_compared_without_sign_reversal(self):
        before = chess.Board()
        after = before.copy()
        after.push_san("f3")
        white_mate = result_for(before, None, 3)
        black_mate = result_for(after, None, -2)
        loss, lost_mate = move_loss_cp(before, after, white_mate, black_mate)
        self.assertGreaterEqual(loss, 199_000)
        self.assertTrue(lost_mate)
        loss_from_equal, lost_mate_from_equal = move_loss_cp(
            before, after, result_for(before, 0), black_mate
        )
        self.assertGreaterEqual(loss_from_equal, 99_000)
        self.assertTrue(lost_mate_from_equal)

        fool = chess.Board()
        for san in ("f3", "e5", "g4"):
            fool.push_san(san)
        before_terminal = fool.copy(stack=False)
        before_mate = result_for(fool, 50)
        fool.push_san("Qh4#")
        loss, lost_mate = move_loss_cp(before_terminal, fool,
                                       before_mate, None)
        # The terminal position checkmates White; Black's move is a winning result.
        self.assertEqual(loss, 0)
        self.assertFalse(lost_mate)

    def test_cancel_returns_partial_report_and_marks_unprocessed_moves(self):
        game = make_game()
        cancel = threading.Event()
        first_board = game.board_at(0)

        def fake_analyze(fen, **_kwargs):
            return result_for(first_board, 0)

        def on_progress(done, _total):
            if done == 1:
                cancel.set()

        report = analyze_game(game, engine_path="stockfish", analyzer=fake_analyze,
                              cancel_event=cancel, progress_callback=on_progress)
        self.assertTrue(report.cancelled)
        self.assertEqual(report.processed_positions, 1)
        self.assertEqual(report.moves[0].classification, "No evaluada")
        self.assertEqual(report.moves[0].category_id, "unrated")
        self.assertEqual(report.moves[0].symbol, "")

    def test_cancelled_intermediate_engine_snapshot_is_not_cached_or_classified(self):
        game = make_game()
        cancel = threading.Event()

        def cancel_during_search(fen, **_kwargs):
            cancel.set()
            return result_for(chess.Board(fen), 900)

        report = analyze_game(game, engine_path="stockfish", analyzer=cancel_during_search,
                              cancel_event=cancel)
        self.assertTrue(report.cancelled)
        self.assertEqual(report.processed_positions, 0)
        self.assertEqual(report.cache_entries, {})
        self.assertTrue(all(move.classification == "No evaluada" for move in report.moves))

    def test_engine_errors_are_not_hidden(self):
        game = make_game()

        def fail(*_args, **_kwargs):
            raise RuntimeError("motor desconectado")

        with self.assertRaisesRegex(RuntimeError, "motor desconectado"):
            analyze_game(game, engine_path="stockfish", analyzer=fail)

    def test_exact_cached_results_are_reused_without_reanalysis(self):
        game = make_game()
        calls = []

        def fake_analyze(fen, **_kwargs):
            calls.append(fen)
            board = chess.Board(fen)
            return result_for(board, 0)

        first = analyze_game(game, engine_path="stockfish", analyzer=fake_analyze, depth=8)
        first_call_count = len(calls)
        second = analyze_game(game, engine_path="stockfish", analyzer=fake_analyze, depth=8,
                              cache=first.cache_entries)
        self.assertEqual(first_call_count, 5)
        self.assertEqual(len(calls), first_call_count)
        self.assertEqual(second.processed_positions, 5)

    def test_one_persistent_session_is_reused_for_all_uncached_positions(self):
        game = make_game()
        created = []
        analyzed = []

        class FakeSession:
            def __init__(self, engine_path, **options):
                created.append((engine_path, options))
                self.closed = False

            def analyze(self, fen, **_options):
                analyzed.append(fen)
                return result_for(chess.Board(fen), 0)

            def close(self):
                self.closed = True

        report = analyze_game(
            game, engine_path="stockfish", depth=8, mode="deep", threads=4, hash_mb=256,
            session_factory=FakeSession,
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(len(analyzed), 5)
        self.assertEqual(created[0][1], {"multipv": 1, "threads": 4, "hash_mb": 256})
        self.assertEqual(report.analyzed_positions, 5)
        self.assertEqual(report.cache_hits, 0)

    def test_persistent_session_closes_after_cancel_and_engine_error(self):
        game = make_game()
        cancel = threading.Event()
        sessions = []

        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                self.closed = False
                sessions.append(self)

            def analyze(self, fen, **_kwargs):
                cancel.set()
                return result_for(chess.Board(fen), 0)

            def close(self):
                self.closed = True

        report = analyze_game(game, engine_path="stockfish", cancel_event=cancel,
                              session_factory=FakeSession)
        self.assertTrue(report.cancelled)
        self.assertEqual(report.analyzed_positions, 0)
        self.assertTrue(sessions[-1].closed)

        class BrokenSession(FakeSession):
            def analyze(self, *_args, **_kwargs):
                raise RuntimeError("uci roto")

        cancel.clear()
        with self.assertRaisesRegex(RuntimeError, "uci roto"):
            analyze_game(game, engine_path="stockfish", session_factory=BrokenSession)
        self.assertTrue(sessions[-1].closed)

    def test_cache_isolated_by_analysis_mode_and_resource_settings(self):
        game = make_game()
        calls = []

        def fake_analyze(fen, **_kwargs):
            calls.append(fen)
            return result_for(chess.Board(fen), 0)

        first = analyze_game(game, engine_path="stockfish", analyzer=fake_analyze,
                             depth=8, mode="deep", threads=4, hash_mb=256)
        analyze_game(game, engine_path="stockfish", analyzer=fake_analyze,
                     depth=8, mode="deep", threads=4, hash_mb=256, cache=first.cache_entries)
        self.assertEqual(len(calls), 5)
        changed_mode = analyze_game(game, engine_path="stockfish", analyzer=fake_analyze,
                                    depth=8, mode="quick", threads=4, hash_mb=256,
                                    cache=first.cache_entries)
        self.assertEqual(len(calls), 10)
        self.assertEqual(changed_mode.cache_hits, 0)

    def test_game_analysis_ui_exposes_quick_and_deep_profiles(self):
        window = MainWindow()
        self.assertEqual(window.game_mode_combo.currentData(), "quick")
        self.assertEqual(window.depth_spin.value(), 18)
        self.assertFalse(window.depth_spin.isEnabled())
        window.game_mode_combo.setCurrentIndex(1)
        self.assertTrue(window.depth_spin.isEnabled())
        window.game_mode_combo.setCurrentIndex(0)
        self.assertFalse(window.depth_spin.isEnabled())
        window.close()

    def test_selecting_report_move_navigates_to_its_before_position_and_recommendation(self):
        game = make_game()
        report = analyze_game(
            game, engine_path="stockfish", analyzer=lambda fen, **_kwargs: result_for(chess.Board(fen), 25)
        )
        window = MainWindow()
        window._set_pgn_game(game)
        window._game_analysis_report = report
        window._game_report_move_selected(2)
        self.assertEqual(window.board_widget.board.fen(), game.board_at(1).fen())
        self.assertEqual(window.pgn_move_label.text(), "Antes de 1... e5")
        self.assertEqual(window.board_widget.review_move.uci(), report.moves[1].move_uci)
        self.assertEqual(window.board_widget.review_symbol, report.moves[1].symbol)
        best = report.position_results[game.board_at(1).fen()].lines[0]
        self.assertEqual(window.best_move_label.text(), f"Mejor jugada: {best.best_move_san}")
        window._game_report_move_selected(3)
        self.assertEqual(window.board_widget.review_move.uci(), report.moves[2].move_uci)
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_review_badge_stays_synchronized_when_main_navigation_continues_after_dialog_close(self):
        game = make_game()
        report = analyze_game(
            game, engine_path="stockfish", analyzer=lambda fen, **_kwargs: result_for(chess.Board(fen), 0)
        )
        window = MainWindow()
        window._set_pgn_game(game)
        window._game_analysis_report = report
        dialog = GameReportDialog(report, window)
        dialog.move_selected.connect(window._game_report_move_selected)
        dialog.badge_visibility_changed.connect(window._set_classification_badges_visible)
        dialog.closed.connect(lambda source=dialog: window._game_report_dialog_closed(source))
        window._game_report_dialog = dialog
        dialog.table.selectRow(1)  # move e5, the board is shown before e5
        self.app.processEvents()
        self.assertEqual(window._pgn_ply, 1)
        self.assertEqual(window.board_widget.board.fen(), game.board_at(1).fen())
        self.assertEqual(window.board_widget.review_move.uci(), report.moves[1].move_uci)
        dialog.badge_checkbox.setChecked(False)
        self.assertFalse(window.board_widget.review_badge_visible)
        dialog.close()
        self.app.processEvents()
        self.assertIsNone(window._game_report_dialog)
        window._navigate_pgn(window._pgn_ply + 1)  # after e5: keep the e5 classification
        self.assertEqual(window.board_widget.board.fen(), game.board_at(2).fen())
        self.assertEqual(window.board_widget.review_move.uci(), report.moves[1].move_uci)
        self.assertFalse(window.board_widget.review_badge_visible)
        window._navigate_pgn(window._pgn_ply - 1)  # after e4: now show e4 instead
        self.assertEqual(window.board_widget.board.fen(), game.board_at(1).fen())
        self.assertEqual(window.board_widget.review_move.uci(), report.moves[0].move_uci)
        self.assertFalse(window.board_widget.review_badge_visible)
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_report_table_selection_emits_the_selected_ply(self):
        game = make_game()
        report = analyze_game(
            game, engine_path="stockfish", analyzer=lambda fen, **_kwargs: result_for(chess.Board(fen), 0)
        )
        dialog = GameReportDialog(report)
        selected = []
        dialog.move_selected.connect(selected.append)
        dialog.table.selectRow(2)
        self.app.processEvents()
        self.assertEqual(selected[-1], 3)
        self.assertIn("Variante principal", dialog.explanation.text())
        visibility = []
        dialog.badge_visibility_changed.connect(visibility.append)
        dialog.badge_checkbox.setChecked(False)
        self.assertEqual(visibility[-1], False)
        self.assertIn("✓", dialog.table.item(0, 3).text())
        self.assertNotEqual(dialog.table.item(0, 3).foreground().color().name(), "#ffffff")
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_completion_from_previous_game_is_discarded(self):
        game_one = make_game()
        report = analyze_game(
            game_one, engine_path="stockfish", analyzer=lambda fen, **_kwargs: result_for(chess.Board(fen), 0)
        )
        game_two = make_game(PGN.replace("1. e4 e5 2. Nf3 Nc6", "1. d4 d5"))
        window = MainWindow()
        window._set_pgn_game(game_one)
        stale_generation = window._game_analysis_generation
        window._set_pgn_game(game_two)
        window._game_analysis_completed(report, False, stale_generation)
        self.assertIsNone(window._game_analysis_report)
        self.assertIsNone(window._game_report_dialog)
        self.assertEqual(window._pgn_game, game_two)
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_gui_runs_full_game_analysis_asynchronously_with_real_stockfish(self):
        try:
            find_engine()
        except FileNotFoundError:
            self.skipTest("Stockfish no está disponible")
        game = make_game(PGN.replace("1. e4 e5 2. Nf3 Nc6", "1. e4 e5"))
        window = MainWindow()
        window._set_pgn_game(game)
        window.depth_check.setChecked(True)
        window.depth_spin.setValue(3)
        window.multipv_spin.setValue(1)
        loop = QEventLoop()
        window._start_game_analysis()
        worker = window.game_worker
        self.assertIsNotNone(worker)
        worker.completed.connect(lambda *_: loop.quit())
        worker.failed.connect(lambda *_: loop.quit())
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        self.app.processEvents()
        self.assertIsNotNone(window._game_analysis_report)
        self.assertEqual(window._game_analysis_report.processed_positions, 3)
        self.assertEqual(window.game_progress.value(), 100)
        self.assertIsNotNone(window._game_report_dialog)
        if window._game_report_dialog:
            window._game_report_dialog.close()
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_gui_cancel_button_marks_partial_report_and_returns(self):
        game = make_game()
        window = MainWindow()
        window._set_pgn_game(game)
        loop = QEventLoop()

        def wait_then_review(selected_game, **kwargs):
            kwargs["cancel_event"].wait(3)
            return analyze_game(
                selected_game,
                analyzer=lambda fen, **_opts: result_for(chess.Board(fen), 0),
                **kwargs,
            )

        with patch("ajedrez.ui.main_window.analyze_game", side_effect=wait_then_review):
            window._start_game_analysis()
            worker = window.game_worker
            self.assertIsNotNone(worker)
            worker.completed.connect(lambda *_: loop.quit())
            worker.failed.connect(lambda *_: loop.quit())
            QTimer.singleShot(0, window._stop_game_analysis)
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
        self.app.processEvents()
        self.assertIsNotNone(window._game_analysis_report)
        self.assertTrue(window._game_analysis_report.cancelled)
        self.assertIn("cancelado", window.game_analysis_status.text().lower())
        if window._game_report_dialog:
            window._game_report_dialog.close()
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_stockfish_integration_analyzes_positions_in_a_short_game(self):
        try:
            engine = find_engine()
        except FileNotFoundError:
            self.skipTest("Stockfish no está disponible")
        game = make_game(PGN.replace("1. e4 e5 2. Nf3 Nc6", "1. e4 e5"))
        report = analyze_game(game, engine_path=engine, depth=4, multipv=1)
        self.assertEqual(report.processed_positions, 3)
        self.assertEqual(len(report.moves), 2)
        self.assertTrue(all(move.loss_cp is not None for move in report.moves))
        self.assertTrue(all(move.best_move_san for move in report.moves))


if __name__ == "__main__":
    unittest.main()
