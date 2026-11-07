"""End-to-end orientation and castling regressions using the local model/engine."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import unittest
import queue
import threading
from types import SimpleNamespace
from unittest.mock import patch

import chess
from PySide6.QtCore import QEventLoop, QPointF, QTimer
from PySide6.QtGui import QBrush, QColor, QImage, QPainter
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.engine import UCIEngineSession, find_engine
from ajedrez.board_coordinates import square_to_grid, grid_to_square
from ajedrez.live_board import LiveBoardTracker
from ajedrez.recognition import RecognitionResult, recognize_position, recognize_position_auto_orientation
from ajedrez.ui.board_widget import BoardWidget
from ajedrez.ui.main_window import MainWindow, LiveAnalysisWorker

PLACEMENT = "rnbqkb1r/pp1ppppp/2p2n2/8/2P5/2N5/PP1PPPPP/R1BQKBNR"


from tests.image_helpers import reverse_cells, board_from_real_tiles


def observation(board, orientation="white"):
    return RecognitionResult(board.board_fen(), orientation, ())


class Phase10Regressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def test_black_view_keeps_real_piece_glyphs_upright(self):
        image = QImage(str(Path(__file__).parent / "fixtures/chesscom_moved_position.jpg"))
        reversed_image = reverse_cells(image)
        result = recognize_position(reversed_image, "black")
        self.assertEqual(result.piece_placement, PLACEMENT)
        self.assertEqual(result.uncertain_squares, ())
        auto = recognize_position_auto_orientation(reversed_image)
        self.assertEqual(auto.orientation, "black")
        self.assertEqual(auto.piece_placement, PLACEMENT)

    def test_missed_plies_keep_the_rights_proved_by_all_legal_histories(self):
        board = chess.Board()
        tracker = LiveBoardTracker(board)
        for san in ("e4", "e5", "Nf3"):
            board.push_san(san)
        tracker.observe(observation(board))
        update = tracker.observe(observation(board))
        self.assertEqual(update.board.castling_xfen(), "KQkq")
        self.assertEqual(update.board.turn, board.turn)
        self.assertTrue(tracker.turn_known)
        # Continue all the way through castling, not only an isolated helper.
        for san in ("Nc6", "Bc4", "Nf6", "d3", "Bc5", "O-O"):
            board.push_san(san)
            tracker.observe(observation(board))
            update = tracker.observe(observation(board))
        self.assertEqual(update.status, "updated")
        self.assertEqual(update.move.uci(), "e1g1")
        self.assertTrue(tracker.turn_known)
        self.assertEqual(update.board.castling_xfen(), "kq")

    def test_all_castles_ignore_partial_frames_and_commit_one_ply(self):
        for turn in ("w", "b"):
            for target_file in (chess.G1, chess.C1):
                for orientation in ("white", "black"):
                    with self.subTest(turn=turn, target_file=target_file, orientation=orientation):
                        board = chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 7 12")
                        origin = chess.E1 if turn == "w" else chess.E8
                        destination = target_file if turn == "w" else target_file+56
                        move = chess.Move(origin,destination)
                        self.assertIn(move, board.legal_moves)
                        tracker = LiveBoardTracker(board)
                        # King settled; rook still in its old square.
                        half = board.copy()
                        half.remove_piece_at(origin)
                        half.set_piece_at(destination, chess.Piece(chess.KING, board.turn))
                        for _ in range(3):
                            tracker.observe(observation(half, orientation))
                        self.assertEqual(tracker.board.fen(), board.fen())
                        final = board.copy(); final.push(move)
                        tracker.observe(observation(final, orientation))
                        update = tracker.observe(observation(final, orientation))
                        self.assertEqual(update.status,"updated")
                        self.assertEqual(update.moves,(move,))
                        self.assertEqual(update.board.fen(),final.fen())
                        self.assertTrue(tracker.turn_known)

    def test_rook_half_castle_does_not_block_a_real_rook_move_forever(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        rook = board.copy(); rook.push_uci("h1f1")
        tracker = LiveBoardTracker(board)
        for _ in range(2):
            self.assertEqual(tracker.observe(observation(rook)).status,"waiting")
        self.assertEqual(tracker.board.fen(),board.fen())
        final = board.copy(); final.push_uci("e1g1")
        tracker.observe(observation(final))
        self.assertEqual(tracker.observe(observation(final)).board.fen(), final.fen())
        tracker = LiveBoardTracker(board)
        for _ in range(3):
            update = tracker.observe(observation(rook))
        self.assertEqual(update.status,"waiting")
        reply=rook.copy();reply.push_san("Kd7")
        tracker.observe(observation(reply))
        self.assertEqual(tracker.observe(observation(reply)).board.fen(),reply.fen())

    def test_coordinate_bijection_and_legal_special_arrows_survive_view_changes(self):
        for orientation in ("white","black"):
            for square in chess.SQUARES:
                column,row = square_to_grid(square,orientation)
                self.assertEqual(grid_to_square(row,column,orientation),square)
        cases = [(chess.STARTING_FEN,"e2e4"),(chess.STARTING_FEN,"g1f3"),
                 ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1","e7e5"),
                 ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1","e4d5"),
                 ("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1","d5e4")]
        for turn in ("w","b"):
            for target in ("g","c"):
                rank = "1" if turn == "w" else "8"
                cases.append((f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1",f"e{rank}{target}{rank}"))
        widget = BoardWidget(); widget.resize(480,480)
        for fen,uci in cases:
            board = chess.Board(fen); move = chess.Move.from_uci(uci)
            self.assertIn(move,board.legal_moves)
            widget.set_board(board);widget.set_best_move(uci)
            for orientation in ("white","black"):
                widget.set_orientation(orientation)
                self.assertEqual(widget.best_move,move)
                for square in (move.from_square,move.to_square):
                    col,row = square_to_grid(square,orientation)
                    self.assertEqual(widget._center(square,0,0,60),QPointF((col+.5)*60,(row+.5)*60))

    def _wait_for(self, predicate):
        loop = QEventLoop()
        timer = QTimer();timer.setInterval(5)
        timer.timeout.connect(lambda: loop.quit() if predicate() else None)
        deadline = QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
        timer.start();deadline.start(6000);loop.exec();timer.stop();deadline.stop()
        self.assertTrue(predicate(),"Stockfish/Qt no completó la actualización esperada")

    def test_live_ui_evaluates_each_castle_without_another_move(self):
        try:
            find_engine()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        for turn in ("w","b"):
            for target in ("g","c"):
                for orientation in ("white","black"):
                    with self.subTest(turn=turn,target=target,orientation=orientation):
                        window = MainWindow()
                        try:
                            board = chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 7 12")
                            rank = "1" if turn == "w" else "8"
                            move = chess.Move.from_uci(f"e{rank}{target}{rank}")
                            window._live_tracker = LiveBoardTracker(board)
                            window._live_monitoring=True;window._live_generation=71
                            window._live_orientation=orientation;window._live_frame_is_stable=True
                            window.time_spin.setValue(.15)
                            window._apply_live_position(board,"prueba antes del enroque")
                            self._wait_for(lambda: window._live_first_eval_logged_for_fen == board.fen())
                            old_id = window._live_analysis_request_id
                            board.push(move)
                            result = recognize_position(board_from_real_tiles(board,orientation),orientation)
                            self.assertEqual(result.piece_placement,board.board_fen())
                            self.assertEqual(result.uncertain_squares,())
                            window._recognition_completed(result,("live_poll",71))
                            window._recognition_completed(result,("live_poll",71))
                            self.assertEqual(window.fen_edit.text(),board.fen())
                            self.assertNotEqual(old_id,window._live_analysis_request_id)
                            self._wait_for(lambda: window.engine_status.text().endswith("análisis finalizado"))
                            self.assertEqual(window.board_widget.board.fen(),board.fen())
                            self.assertTrue(window.board_widget.analysis_moves)
                            self.assertTrue(all(m in board.legal_moves for m in window.board_widget.analysis_moves))
                            self.assertNotIn("pendiente",window.eval_label.text())
                            # An old completion cannot erase these current arrows/evaluation.
                            current = (window.eval_label.text(),window.board_widget.analysis_moves)
                            from ajedrez.engine import AnalysisResult
                            window._live_analysis_completed(chess.STARTING_FEN,AnalysisResult(chess.STARTING_FEN,"old",()),old_id,window.live_analysis_worker)
                            self.assertEqual((window.eval_label.text(),window.board_widget.analysis_moves),current)
                        finally:
                            window.close();self.app.processEvents()

    def test_real_stockfish_recommends_castling_and_it_is_drawn_in_both_views(self):
        try:
            engine=find_engine()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        board=chess.Board()
        for san in ("e4","e5","Nf3","Nc6","Bc4","Nf6","d3","Bc5"):
            board.push_san(san)
        with UCIEngineSession(engine,multipv=5,threads=4,hash_mb=256) as session:
            result=session.analyze(board.fen(),depth=10)
        self.assertIn("e1g1",[line.best_move_uci for line in result.lines])
        self.assertEqual(next(line.best_move_san for line in result.lines if line.best_move_uci=="e1g1"),"O-O")
        widget=BoardWidget();widget.set_board(board)
        for orientation in ("white","black"):
            widget.set_orientation(orientation)
            widget.set_best_moves(line.best_move_uci for line in result.lines)
            self.assertIn(chess.Move.from_uci("e1g1"),widget.analysis_moves)

    def test_live_pipeline_retains_evaluation_after_missed_moves_then_castling(self):
        try:
            find_engine()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        window=MainWindow()
        try:
            board=chess.Board()
            window._live_tracker=LiveBoardTracker(board)
            window._live_monitoring=True;window._live_generation=44
            window._live_orientation="black";window._live_frame_is_stable=True
            window.time_spin.setValue(.15)
            window._apply_live_position(board,"inicio de prueba")
            self._wait_for(lambda: window._live_first_eval_logged_for_fen == board.fen())
            for group in (("e4","e5","Nf3"),("Nc6",),("Bc4",),("Nf6",),("d3",),("Bc5",),("O-O",)):
                for san in group:
                    board.push_san(san)
                result=recognize_position(board_from_real_tiles(board,"black"),"black")
                self.assertEqual(result.piece_placement,board.board_fen())
                for _ in range(2):
                    window._recognition_completed(result,("live_poll",44))
                self.assertTrue(window._live_tracker.turn_known)
                self.assertEqual(window._live_tracker.board.turn,board.turn)
                self.assertEqual(window._live_tracker.board.castling_rights,board.castling_rights)
            self._wait_for(lambda: window.engine_status.text().endswith("análisis finalizado"))
            self.assertEqual(window.board_widget.board.board_fen(),board.board_fen())
            self.assertEqual(window.board_widget.board.turn,chess.BLACK)
            self.assertTrue(window.board_widget.analysis_moves)
            self.assertNotIn("pendiente",window.eval_label.text())
        finally:
            window.close();self.app.processEvents()

    def test_white_pieces_have_a_solid_body_on_both_square_colors(self):
        widget = BoardWidget()
        widget.resize(480, 480)
        image = widget.grab().toImage()
        for square in (chess.D2, chess.E2):
            col, row = widget._display_coords(square)
            pixels = [image.pixelColor(col*60+x, row*60+y) for x in range(60) for y in range(60)]
            white = sum(c.red() >= 248 and c.green() >= 248 and c.blue() >= 248 for c in pixels)
            dark = sum(c.red() < 65 and c.green() < 65 and c.blue() < 65 for c in pixels)
            self.assertGreater(white, 200)
            self.assertGreater(dark, 40)

    def test_straight_arrows_do_not_inherit_a_filled_brush(self):
        class Painter:
            def __init__(self):
                self.brush = QBrush(QColor("red"))
                self.path_brush = None
            def setPen(self, pen): pass
            def setBrush(self, brush): self.brush = QBrush(brush)
            def drawLine(self, start, end): self.path_brush = self.brush
            def drawPolygon(self, polygon): pass
        painter = Painter()
        BoardWidget._draw_analysis_arrow(painter, QPointF(30,30), QPointF(90,150),60,QColor("blue"),1)
        from PySide6.QtCore import Qt
        self.assertEqual(painter.path_brush.style(), Qt.BrushStyle.NoBrush)

    def test_straight_arrows_preserve_the_five_scaled_stroke_widths(self):
        class Painter:
            def setPen(self, pen): self.pen=pen
            def setBrush(self, brush): pass
            def drawLine(self, start, end): self.width=self.pen.widthF();self.line=(start,end)
            def drawPolygon(self, polygon): pass
        widths=(9,7,5.5,4,3)
        for index in range(5):
            for cell in (60,90):
                painter=Painter()
                BoardWidget._draw_analysis_arrow(painter,QPointF(30,150),QPointF(30,90),cell,QColor("green"),index)
                self.assertAlmostEqual(painter.width,widths[index]*cell/60)
                self.assertEqual(painter.line,(QPointF(30,150),QPointF(30,90)))

    def test_incomplete_and_repeated_depth_multipv_batches_never_mix(self):
        # Unit protocol fixture; integration tests below use the real engine.
        session=object.__new__(UCIEngineSession)
        session._closed=False;session.engine_path="unit-protocol";session.multipv=5
        session.write_lock=threading.Lock();session.process=SimpleNamespace(poll=lambda:None,stdin=None)
        session.messages=queue.Queue();session._send=lambda command:None
        moves=("e2e4","d2d4","g1f3","b1c3","c2c4")
        for index,move in enumerate(moves,1):
            session.messages.put(f"info depth 8 multipv {index} score cp 20 pv {move}")
        session.messages.put("info string incomplete housekeeping")
        session.messages.put("info depth 8 multipv 1 score cp 25 pv d2d4")
        session.messages.put("info depth 8 multipv 2 score cp 24 pv g1f3")
        session.messages.put("bestmove d2d4")
        progress=[]
        result=session.analyze(chess.STARTING_FEN,depth=8,progress_callback=progress.append)
        self.assertEqual(tuple(line.best_move_uci for line in result.lines),moves)
        for lines in progress:
            self.assertEqual(len({line.best_move_uci for line in lines}),len(lines))
            self.assertEqual(len({line.depth for line in lines}),1)
            self.assertEqual(tuple(line.index for line in lines),tuple(range(1,len(lines)+1)))

    def test_first_complete_multipv_batch_is_not_hidden_by_gui_throttling(self):
        from ajedrez.engine import AnalysisLine
        worker=LiveAnalysisWorker("unused-unit-test",5)
        lines=tuple(AnalysisLine(index,move,move,0,None,8,.01,(move,))
                    for index,move in enumerate(("e2e4","d2d4","g1f3","b1c3","c2c4"),1))
        received=[]
        worker.progress.connect(lambda fen,items,token: received.append(len(items)))
        worker.request_position(chess.STARTING_FEN,.2,None)
        with patch("ajedrez.ui.main_window.time.monotonic",return_value=100):
            worker._emit_progress(chess.STARTING_FEN,lines[:1],1)
            worker._emit_progress(chess.STARTING_FEN,lines,1)
            worker._emit_progress(chess.STARTING_FEN,lines,1)
        self.assertEqual(received,[1,5])

    def test_multipv_five_returns_only_the_single_legal_reply(self):
        board=chess.Board("7k/8/5K2/8/8/8/8/7R b - - 0 1")
        self.assertEqual(board.legal_moves.count(),1)
        try:
            engine=find_engine()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        with UCIEngineSession(engine,multipv=5) as session:
            result=session.analyze(board.fen(),depth=6)
        self.assertEqual(len(result.lines),1)
        self.assertIn(chess.Move.from_uci(result.lines[0].best_move_uci),board.legal_moves)

    def test_real_recognition_stockfish_and_arrow_endpoints_in_all_four_combinations(self):
        try:
            engine = find_engine()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        image = QImage(str(Path(__file__).parent / "fixtures/chesscom_moved_position.jpg"))
        widget = BoardWidget()
        widget.resize(480,480)
        with UCIEngineSession(engine, multipv=5, threads=4, hash_mb=256) as session:
            for turn in ("w", "b"):
                reference = chess.Board(f"{PLACEMENT} {turn} KQkq - 0 3")
                result = session.analyze(reference.fen(), depth=7)
                view_moves = []
                for orientation in ("white", "black"):
                    with self.subTest(turn=turn, orientation=orientation):
                        recognized = recognize_position(image if orientation == "white" else reverse_cells(image), orientation)
                        self.assertEqual(recognized.piece_placement, PLACEMENT)
                        board = chess.Board(f"{recognized.piece_placement} {turn} KQkq - 0 3")
                        self.assertEqual(result.fen, board.fen())
                        self.assertEqual(len(result.lines), 5)
                        widget.set_board(board)
                        widget.set_orientation(orientation)
                        widget.set_best_moves(line.best_move_uci for line in result.lines)
                        drawn = {}
                        def record(painter, start, end, cell, color, index):
                            drawn[index] = (start, end)
                        with patch.object(widget, "_draw_analysis_arrow", side_effect=record):
                            widget.grab()
                        self.assertEqual(len(drawn), 5)
                        for index, line in enumerate(result.lines):
                            move = chess.Move.from_uci(line.best_move_uci)
                            self.assertIn(move, board.legal_moves)
                            for square, point in zip((move.from_square,move.to_square), drawn[index]):
                                file, rank = chess.square_file(square), chess.square_rank(square)
                                column, row = ((file,7-rank) if orientation == "white" else (7-file,rank))
                                self.assertEqual(point, QPointF((column+.5)*60,(row+.5)*60))
                        self.assertEqual(widget.board.fen(), board.fen())
                        view_moves.append(widget.analysis_moves)
                self.assertEqual(view_moves[0],view_moves[1])
