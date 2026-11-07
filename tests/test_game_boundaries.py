"""Continuous same-region observation across game boundaries and board flips."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import unittest
from unittest.mock import patch
import chess
from pathlib import Path
from PySide6.QtCore import QRect,QEventLoop,QTimer
from ajedrez.live_board import LiveBoardTracker
from ajedrez.recognition import recognize_position, remap_recognition
from ajedrez.ui.main_window import MainWindow
from tests.qt_test_support import test_application
from tests.image_helpers import board_from_real_tiles
from tests.test_live_board import observation


def advanced_board():
    board=chess.Board()
    for san in ("e4","e5","Nf3","Nc6","Bc4","Nf6","d3","Bc5","O-O","d6","Nc3","O-O"):
        board.push_san(san)
    return board


class GameBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def make_window(self,orientation="white"):
        window=MainWindow();board=advanced_board()
        window._live_monitoring=True;window._live_generation=7
        window._live_frame_is_stable=True;window._live_orientation=orientation
        window._live_tracker=LiveBoardTracker(board)
        window.board_widget.set_board(board);window.board_widget.set_orientation(orientation)
        window.fen_edit.setText(board.fen())
        window._live_screen=object();window._live_region=QRect(20,30,800,800)
        self.addCleanup(window.close)
        return window

    def feed(self,window,result,count=2):
        for _ in range(count):
            window._recognition_completed(result,("live_poll",window._live_generation))

    def feed_image(self,window,board,view,count=2):
        image=board_from_real_tiles(board,view)
        for index in range(count):
            result=recognize_position(image,window._live_orientation)
            window._live_frame_is_stable=index>0
            window._recognition_completed(result,("live_poll",window._live_generation))
        window._live_timer.stop()  # Unit fixture has no real monitor; region remains selected.

    def wait_for(self,predicate):
        loop=QEventLoop();timer=QTimer();timer.setInterval(5)
        timer.timeout.connect(lambda:loop.quit() if predicate() else None)
        deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
        timer.start();deadline.start(8000);loop.exec();timer.stop();deadline.stop()
        self.assertTrue(predicate(),"La nueva partida no recibió el análisis de su FEN")

    def test_new_game_opposite_orientation_resets_state_without_reselecting_region(self):
        window=self.make_window();region=window._live_region;screen=window._live_screen
        # Pixels from the black view decoded using the OLD white mapping.
        result=recognize_position(board_from_real_tiles(chess.Board(),"black"),"white")
        with patch.object(window,"_request_live_analysis") as analyze:
            self.feed(window,result)
        self.assertEqual(window._live_orientation,"black")
        self.assertEqual(window._live_tracker.board.fen(),chess.STARTING_FEN)
        self.assertTrue(window._live_tracker.turn_known)
        self.assertEqual(window._live_tracker.board.move_stack,[])
        self.assertIs(window._live_screen,screen);self.assertEqual(window._live_region,region)
        self.assertTrue(window._live_monitoring);analyze.assert_called_once()
        self.assertIsNone(window._recognition_review)

    def test_same_game_flip_preserves_history_turn_and_castling_metadata(self):
        window=self.make_window();before=window._live_tracker.board.copy(stack=True)
        result=recognize_position(board_from_real_tiles(before,"black"),"white")
        with patch.object(window,"_request_live_analysis"):
            self.feed(window,result)
        self.assertEqual(window._live_orientation,"black")
        self.assertEqual(window._live_tracker.board.fen(),before.fen())
        self.assertEqual(window._live_tracker.board.move_stack,before.move_stack)
        self.assertEqual(window.board_widget.orientation,"black")

    def test_same_orientation_new_initial_position_restores_official_start_fen(self):
        window=self.make_window()
        result=observation(chess.Board())
        with patch.object(window,"_request_live_analysis"):
            self.feed(window,result)
        self.assertEqual(window._live_tracker.board.fen(),chess.STARTING_FEN)
        self.assertTrue(window._live_tracker.turn_known)
        self.assertEqual(window._live_orientation,"white")

    def test_black_to_white_new_game_also_preserves_capture_selection(self):
        window=self.make_window("black");screen=window._live_screen;region=window._live_region
        with patch.object(window,"_request_live_analysis"):
            self.feed_image(window,chess.Board(),"white")
        self.assertEqual(window._live_tracker.board.fen(),chess.STARTING_FEN)
        self.assertEqual(window._live_orientation,"white")
        self.assertIs(window._live_screen,screen);self.assertEqual(window._live_region,region)

    def test_view_and_side_to_move_remain_independent_in_all_four_combinations(self):
        for view in ("white","black"):
            for turn in (chess.WHITE,chess.BLACK):
                with self.subTest(view=view,turn=turn):
                    window=self.make_window(view);board=advanced_board()
                    if turn==chess.BLACK:board.push_san("a3")
                    window._live_tracker.reset(board)
                    window.fen_edit.setText(board.fen());window.board_widget.set_board(board)
                    epoch=window._live_game_epoch;generation=window._live_generation
                    newview="black" if view=="white" else "white"
                    with patch.object(window,"_request_live_analysis"):
                        self.feed_image(window,board,newview)
                    self.assertEqual(window._live_tracker.board.fen(),board.fen())
                    self.assertEqual(window._live_tracker.board.turn,turn)
                    self.assertEqual(window._live_game_epoch,epoch)
                    self.assertGreater(window._live_generation,generation)
                    self.assertEqual(window._live_orientation,newview)
                    window.close()

    def test_flip_and_missed_move_are_explained_from_the_existing_history(self):
        window=self.make_window();before=window._live_tracker.board.copy(stack=True)
        after=before.copy();after.push_san("a3")
        with patch.object(window,"_request_live_analysis"):
            self.feed_image(window,after,"black")
        self.assertEqual(window._live_tracker.board.fen(),after.fen())
        self.assertEqual(window._live_tracker.board.move_stack,after.move_stack)
        self.assertEqual(window._live_orientation,"black")

    def test_isolated_start_or_doubtful_rotation_does_not_reset_the_game(self):
        window=self.make_window();before=window._live_tracker.board.fen();epoch=window._live_game_epoch
        with patch.object(window,"_request_live_analysis") as analyze:
            self.feed(window,observation(chess.Board()),count=1)
            self.feed(window,observation(advanced_board()),count=1)
            self.assertEqual(window._live_tracker.board.fen(),before)
            inverted=remap_recognition(observation(advanced_board()),"black")
            # This models raw pieces from a flipped raster decoded with the old view.
            from ajedrez.recognition import RecognitionResult
            raw=RecognitionResult(inverted.piece_placement,"white",inverted.predictions)
            self.feed(window,raw,count=1)
            self.feed(window,observation(advanced_board(),.2),count=1)
            self.feed(window,raw,count=1)
            self.assertEqual(window._live_orientation,"white")
            self.assertEqual(window._live_game_epoch,epoch)
            self.assertEqual(window._live_tracker.board.fen(),before)
            self.feed(window,raw,count=1)
            self.assertEqual(window._live_orientation,"black")
            self.assertEqual(window._live_tracker.board.fen(),before)
            analyze.assert_called_once()

    def test_new_game_without_initial_frame_keeps_history_unknown_until_a_move(self):
        window=self.make_window()
        sparse=chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 42 80")
        window._live_tracker.reset(sparse);window.fen_edit.setText(sparse.fen());window.board_widget.set_board(sparse)
        new=chess.Board()
        for san in ("e4","e5","Nf3"):new.push_san(san)
        with patch.object(window,"_request_live_analysis") as analyze:
            self.feed_image(window,new,"black",count=3)
            self.assertEqual(window._live_orientation,"black")
            self.assertEqual(window._live_tracker.board.board_fen(),new.board_fen())
            self.assertFalse(window._live_tracker.turn_known)
            self.assertEqual(window._live_tracker.board.castling_xfen(),"-")
            self.assertIsNone(window._live_tracker.board.ep_square)
            self.assertEqual(window._live_tracker.board.fullmove_number,1)
            self.assertEqual(window._live_tracker.board.move_stack,[])
            self.assertEqual(window.board_widget.analysis_moves,())
            analyze.assert_not_called()
            new.push_san("Nc6")
            self.feed_image(window,new,"black")
            self.assertTrue(window._live_tracker.turn_known)
            self.assertEqual(window._live_tracker.board.turn,chess.WHITE)
            self.assertEqual(window._live_tracker.board.castling_xfen(),"-")
            analyze.assert_called_once()

    def test_check_can_prove_the_turn_in_an_unlinked_game_without_inventing_castling(self):
        window=self.make_window();sparse=chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 0 50")
        window._live_tracker.reset(sparse);window.fen_edit.setText(sparse.fen());window.board_widget.set_board(sparse)
        new=chess.Board()
        for san in ("e4","d6","Bb5+"):new.push_san(san)
        with patch.object(window,"_request_live_analysis") as analyze:
            self.feed_image(window,new,"white",count=3)
        self.assertTrue(window._live_tracker.turn_known)
        self.assertEqual(window._live_tracker.board.turn,chess.BLACK)
        self.assertEqual(window._live_tracker.board.castling_xfen(),"-")
        analyze.assert_called_once()

    def test_new_game_discards_pending_castle_hypotheses_and_all_old_metadata(self):
        window=self.make_window();old=chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 17 42")
        window._live_tracker.reset(old);window.fen_edit.setText(old.fen());window.board_widget.set_board(old)
        partial=old.copy();partial.push_uci("h1f1")
        for _ in range(3):window._live_tracker.observe(observation(partial))
        self.assertTrue(window._live_tracker._pending_transition_roots)
        with patch.object(window,"_request_live_analysis"):
            self.feed_image(window,chess.Board(),"black")
        self.assertEqual(window._live_tracker.board.fen(),chess.STARTING_FEN)
        self.assertEqual(window._live_tracker._pending_transition_roots,[])
        self.assertEqual(len(window._live_tracker._recent_confirmed),0)
        self.assertEqual(window._live_tracker._candidate_count,0)

    def test_legal_knight_return_to_initial_layout_is_not_a_new_game(self):
        window=self.make_window();board=chess.Board()
        for san in ("Nf3","Nf6","Ng1"):board.push_san(san)
        window._live_tracker.reset(board);window.fen_edit.setText(board.fen());window.board_widget.set_board(board)
        epoch=window._live_game_epoch
        board.push_san("Ng8")
        with patch.object(window,"_request_live_analysis"):
            self.feed(window,observation(board))
        self.assertEqual(window._live_tracker.board.fen(),board.fen())
        self.assertEqual(window._live_tracker.board.fullmove_number,3)
        self.assertEqual(window._live_tracker.board.halfmove_clock,4)
        self.assertEqual(window._live_game_epoch,epoch)

    def test_view_comparison_never_initializes_onnx_or_mutates_tracker_candidates(self):
        from ajedrez.recognition import remap_recognition
        from ajedrez.live_session import LiveSessionWatch
        board=advanced_board();tracker=LiveBoardTracker(board)
        moved=board.copy();moved.push_san("a3");tracker.observe(observation(moved))
        state=(tracker.board.fen(),tracker._candidate,tracker._candidate_count)
        raw=remap_recognition(observation(board),"black")
        from ajedrez.recognition import RecognitionResult
        raw=RecognitionResult(raw.piece_placement,"white",raw.predictions)
        with patch("ajedrez.recognition._load_classifier",side_effect=AssertionError("No debe ejecutar ONNX otra vez")):
            decision=LiveSessionWatch().inspect(raw,tracker,"white",visual_stable=True)
        self.assertEqual(decision.event,"waiting")
        self.assertEqual((tracker.board.fen(),tracker._candidate,tracker._candidate_count),state)

    def test_pending_castling_continuation_is_not_mistaken_for_another_game(self):
        window=self.make_window();board=chess.Board("r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 10")
        window._live_tracker.reset(board);window.fen_edit.setText(board.fen());window.board_widget.set_board(board)
        epoch=window._live_game_epoch
        board.push_san("Rf1")
        with patch.object(window,"_request_live_analysis"):
            self.feed(window,observation(board),count=4)
            for san in ("a6","a3","b6"):board.push_san(san)
            self.feed(window,observation(board))
        self.assertEqual(window._live_tracker.board.fen(),board.fen())
        self.assertEqual(window._live_game_epoch,epoch)

    def test_incomplete_rotation_frames_and_late_old_recognition_do_not_reset(self):
        window=self.make_window();before=window._live_tracker.board.fen()
        epoch=window._live_game_epoch;generation=window._live_generation
        incomplete=chess.Board(None)
        incomplete.set_piece_at(chess.E4,chess.Piece(chess.QUEEN,chess.WHITE))
        with patch.object(window,"_request_live_analysis") as analyze:
            self.feed(window,observation(incomplete),count=4)
            self.assertEqual(window._live_tracker.board.fen(),before)
            self.assertEqual(window._live_game_epoch,epoch)
            analyze.assert_not_called()
            self.feed_image(window,chess.Board(),"black")
            current=window._live_tracker.board.fen()
            window._recognition_completed(observation(advanced_board()),("live_poll",generation))
            self.assertEqual(window._live_tracker.board.fen(),current)
            self.assertEqual(window._live_orientation,"black")
            analyze.assert_called_once()

    def test_multiple_games_and_flips_reuse_real_uci_and_onnx_and_discard_old_results(self):
        from ajedrez.engine import UCIEngineSession,find_engine
        from ajedrez import recognition
        try:engine=find_engine()
        except FileNotFoundError as exc:self.skipTest(str(exc))
        sessions=[];game_commands=[];analyses=[]
        class TracedSession(UCIEngineSession):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);sessions.append(self)
            def new_game(self):
                game_commands.append(self.process.pid);super().new_game()
            def analyze(self,fen,**kwargs):
                result=super().analyze(fen,**kwargs);analyses.append(result);return result
        window=self.make_window();screen=window._live_screen;region=window._live_region
        mate=chess.Board("1n1Rkb1r/p4ppp/4q3/4p1B1/4P3/8/PPP2PPP/2K5 b k - 1 17")
        self.assertTrue(mate.is_checkmate())
        window._live_tracker.reset(mate);window.fen_edit.setText(mate.fen());window.board_widget.set_board(mate)
        model=recognition._load_classifier(str(Path('models/chess-tiles-v2.onnx').resolve()))
        window.time_spin.setValue(.15)
        def ready():
            return bool(window._live_search_ticket and window._live_search_finalized
                        and window._live_search_ticket.fen==window.fen_edit.text())
        with patch("ajedrez.ui.main_window.UCIEngineSession",TracedSession):
            try:
                window._apply_live_position(window._live_tracker.board,"partida previa")
                self.wait_for(ready)
                worker=window.live_analysis_worker;pid=sessions[0].process.pid
                old_generation=window._live_generation
                first_new_payload=None
                for round_number,view in enumerate(("black","white","white"),1):
                    # The previous game is advanced (both sides have castled).
                    if round_number>1:
                        progressed=chess.Board();progressed.push_san("e4")
                        for san in ("e5","Nf3","Nc6","Bc4","Nf6","d3","Bc5","O-O","d6","Nc3","O-O"):
                            progressed.push_san(san)
                            self.feed_image(window,progressed,window._live_orientation)
                    self.feed_image(window,chess.Board(),view)
                    self.wait_for(ready)
                    self.assertEqual(window._live_tracker.board.fen(),chess.STARTING_FEN)
                    self.assertEqual(window._live_orientation,view)
                    self.assertEqual(len(window.board_widget.analysis_moves),5)
                    self.assertTrue(all(move in window._live_tracker.board.legal_moves for move in window.board_widget.analysis_moves))
                    self.assertIs(window.live_analysis_worker,worker)
                    self.assertEqual(sessions[0].process.pid,pid)
                    self.assertIs(window._live_screen,screen);self.assertEqual(window._live_region,region)
                    current=(window.eval_label.text(),window.board_widget.analysis_moves)
                    if first_new_payload is None:
                        first_new_payload=(window._live_analysis_request_id,analyses[-1])
                    else:
                        old_id,old_result=first_new_payload
                        window._live_analysis_completed(old_result.fen,old_result,old_id,worker)
                        self.assertEqual((window.eval_label.text(),window.board_widget.analysis_moves),current)
                    # A move and a pure view flip while a search is still running.
                    moved=chess.Board();moved.push_san("e4")
                    self.feed_image(window,moved,view)
                    opposite="white" if view=="black" else "black"
                    fen=window._live_tracker.board.fen();epoch=window._live_game_epoch
                    self.feed_image(window,moved,opposite)
                    self.wait_for(ready)
                    self.assertEqual(window._live_tracker.board.fen(),fen)
                    self.assertEqual(window._live_game_epoch,epoch)
                    self.assertEqual(window._live_tracker.board.turn,chess.BLACK)
                self.assertGreater(window._live_generation,old_generation)
                self.assertEqual(len(sessions),1)
                self.assertEqual(len(game_commands),4) # prior game + 3 new games, not view flips
                self.assertIs(model,recognition._load_classifier(str(Path('models/chess-tiles-v2.onnx').resolve())))
            finally:window.close();self.app.processEvents()
