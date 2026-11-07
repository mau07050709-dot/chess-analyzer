import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from unittest.mock import patch
import chess
from PySide6.QtCore import QEventLoop,QTimer,QRect
from ajedrez.ui.main_window import MainWindow
from ajedrez.ui.analysis_controls import EvaluationBar,analysis_position
from ajedrez.live_board import LiveBoardTracker
from ajedrez.engine import UCIEngineSession,AnalysisLine,AnalysisResult
from tests.qt_test_support import test_application
from tests.test_live_board import observation
from tests.test_game_boundaries import advanced_board


class AnalysisControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def window(self,board=None,live=False,known=True):
        w=MainWindow();self.addCleanup(w.close)
        board=board or chess.Board()
        w.board_widget.set_board(board);w.fen_edit.setText(board.fen())
        if live:
            w._live_tracker=LiveBoardTracker(board,turn_known=known)
            w._live_monitoring=True;w._live_frame_is_stable=True
            w._live_screen=object();w._live_region=QRect(0,0,800,800)
        return w

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();poll.setInterval(5)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None)
        deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
        poll.start();deadline.start(8000);loop.exec();poll.stop();deadline.stop()
        self.assertTrue(predicate(),'No llegó un análisis vigente')

    def test_effective_fen_copies_without_touching_history_or_special_rights(self):
        board=chess.Board();board.push_san('e4')
        before=(board.fen(en_passant='fen'),list(board.move_stack))
        auto=analysis_position(board,None);self.assertEqual(auto.fen(en_passant='fen'),before[0])
        white=analysis_position(board,chess.WHITE)
        self.assertEqual(white.board_fen(),board.board_fen());self.assertIsNone(white.ep_square)
        self.assertEqual(white.castling_rights,board.castling_rights)
        self.assertEqual(white.fullmove_number,board.fullmove_number)
        black=analysis_position(board,chess.BLACK);self.assertEqual(black.ep_square,chess.E3)
        self.assertEqual((board.fen(en_passant='fen'),board.move_stack),before)

    def test_auto_default_and_selection_only_requests_analysis_without_tracker_reset(self):
        w=self.window(advanced_board(),live=True)
        tracker=w._live_tracker;before=(tracker.board.fen(),list(tracker.board.move_stack))
        state=(w._live_generation,w._live_game_epoch,w._live_orientation,w._live_region)
        self.assertTrue(w.auto_turn_button.isChecked());self.assertIsNone(w._analysis_turn)
        with patch.object(w,'_request_live_analysis') as request,patch.object(tracker,'reset') as reset:
            w.auto_turn_button.click();request.assert_not_called()
            for button,turn in ((w.white_turn_button,True),(w.black_turn_button,False),(w.auto_turn_button,None)):
                button.click();self.assertIs(w._analysis_turn,turn)
                button.click()  # Selecting the already active option is a no-op.
                self.assertEqual((tracker.board.fen(),tracker.board.move_stack),before)
                self.assertEqual((w._live_generation,w._live_game_epoch,w._live_orientation,w._live_region),state)
                self.assertEqual(w.fen_edit.text(),before[0]);self.assertEqual(w.board_widget.board.fen(),before[0])
            self.assertEqual(request.call_count,3);reset.assert_not_called()

    def test_provisional_turn_can_be_overridden_without_confirming_tracker(self):
        w=self.window(advanced_board(),live=True,known=False)
        tracker=w._live_tracker;before=tracker.board.fen()
        with patch.object(w,'_request_live_analysis') as request,patch.object(w,'_open_recognition_review') as review:
            w.black_turn_button.click();request.assert_called_once()
            self.assertFalse(tracker.turn_known);self.assertEqual(tracker.board.fen(),before)
            w.auto_turn_button.click();self.assertEqual(request.call_count,1)
            self.assertFalse(tracker.turn_known);review.assert_not_called()

    def test_tracking_retries_moves_and_new_games_stay_independent_of_override(self):
        w=self.window(advanced_board(),live=True);w._live_orientation='white'
        with patch.object(w,'_request_live_analysis'),patch.object(w,'_open_recognition_review') as review:
            w.black_turn_button.click();board=w._live_tracker.board.copy(stack=True)
            doubtful=observation(board,confidence=.3)
            w._recognition_completed(doubtful,('live_poll',w._live_generation))
            self.assertEqual(w._live_tracker.board.fen(),board.fen())
            board.push_san('a3')
            for _ in range(2):w._recognition_completed(observation(board),('live_poll',w._live_generation))
            self.assertEqual(w._live_tracker.board.fen(),board.fen())
            for _ in range(2):w._recognition_completed(observation(chess.Board()),('live_poll',w._live_generation))
            self.assertEqual(w._live_tracker.board.fen(),chess.STARTING_FEN)
            self.assertIs(w._analysis_turn,chess.BLACK);self.assertTrue(w.black_turn_button.isChecked())
            review.assert_not_called()
        w._live_timer.stop()

    def test_old_turn_result_and_old_standard_worker_progress_are_discarded(self):
        w=self.window(live=True)
        class Source:multipv=5
        source=Source();w.live_analysis_worker=source
        old=AnalysisLine(1,'e2e4','e4',500,None,8,.1,('e4',))
        w._bind_live_search(chess.STARTING_FEN,10,source)
        generation=w._analysis_generation
        # Avoid calling a fake source as a real worker; retain its identity only
        # when exercising result validation after selection.
        w.live_analysis_worker=None
        with patch.object(w,'_request_live_analysis'):
            w.black_turn_button.click()
        w.live_analysis_worker=source
        w._bind_live_search(w._effective_analysis_board().fen(),11,source)
        w.eval_label.setText('actual')
        w._live_analysis_completed(chess.STARTING_FEN,AnalysisResult(chess.STARTING_FEN,'fake',(old,)),10,source)
        w._show_progress((old,),generation)
        self.assertEqual(w.eval_label.text(),'actual');self.assertEqual(w.board_widget.analysis_moves,())
        effective=w._effective_analysis_board().fen()
        new=AnalysisLine(1,'e7e5','e5',200,None,8,.1,('e5',))
        w._live_analysis_completed(effective,AnalysisResult(effective,'fake',(new,)),11,source)
        self.assertIn('+2.00 peones',w.eval_label.text())
        self.assertGreater(w.eval_bar.segment_fractions()[1],.5)
        self.assertEqual(w.board_widget.analysis_moves,(chess.Move.from_uci('e7e5'),))
        w.live_analysis_worker=None

    def test_orientation_flip_under_manual_turn_preserves_real_side_and_history(self):
        from ajedrez.recognition import remap_recognition,RecognitionResult
        board=advanced_board();board.push_san('a3')
        w=self.window(board,live=True)
        before=(board.fen(),list(board.move_stack));epoch=w._live_game_epoch
        mapped=remap_recognition(observation(board),'black')
        pixels_in_old_view=RecognitionResult(mapped.piece_placement,'white',mapped.predictions)
        with patch.object(w,'_request_live_analysis'):
            w.white_turn_button.click()
            for _ in range(2):w._recognition_completed(pixels_in_old_view,('live_poll',w._live_generation))
        self.assertEqual(w._live_orientation,'black');self.assertEqual(w.eval_bar.orientation,'black')
        self.assertEqual((w._live_tracker.board.fen(),w._live_tracker.board.move_stack),before)
        self.assertEqual(w._live_tracker.board.turn,chess.BLACK)
        self.assertEqual(w._effective_analysis_board().turn,chess.WHITE)
        self.assertEqual(w._live_game_epoch,epoch);w._live_timer.stop()

    def test_pgn_navigation_keeps_recorded_game_and_manual_analysis_separate(self):
        from ajedrez.pgn import PGNGame
        board=chess.Board();positions=[board.copy()]
        for san in ('e4','e5'):board.push_san(san);positions.append(board.copy())
        game=PGNGame({},tuple(positions),('e4','e5'));w=self.window();w._set_pgn_game(game)
        fens=tuple(p.fen() for p in game.positions)
        with patch.object(w,'_request_live_analysis') as request:
            w.black_turn_button.click();w._navigate_pgn(2)
            w._start_analysis()
            self.assertEqual(w.board_widget.board.fen(),game.board_at(2).fen())
            self.assertEqual(w._effective_analysis_board().turn,chess.BLACK)
            w.auto_turn_button.click()
            self.assertEqual(w._effective_analysis_board().turn,chess.WHITE)
            self.assertGreaterEqual(request.call_count,3)
        self.assertIs(w._pgn_game,game);self.assertEqual(tuple(p.fen() for p in game.positions),fens)

    def test_real_engine_switches_colors_reusing_one_session_and_original_position(self):
        sessions=[]
        class CountedSession(UCIEngineSession):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);sessions.append(self)
        w=self.window();w.time_spin.setValue(.15);w.multipv_spin.setValue(5)
        w.board_widget.set_orientation('black')
        before=w.fen_edit.text()
        with patch('ajedrez.ui.main_window.UCIEngineSession',CountedSession):
            for button,color in ((w.white_turn_button,True),(w.black_turn_button,False),
                                 (w.white_turn_button,True),(w.auto_turn_button,True)):
                button.click()
                self.wait(lambda:w._live_search_finalized and bool(w.board_widget.analysis_moves))
                self.assertEqual(len(w.board_widget.analysis_moves),5)
                for move in w.board_widget.analysis_moves:
                    self.assertEqual(w.board_widget.board.piece_at(move.from_square).color,color)
                    self.assertIn(move,w._effective_analysis_board().legal_moves)
                self.assertEqual(w.board_widget.orientation,'black')
                self.assertEqual(w.board_widget.board.fen(),before);self.assertEqual(w.fen_edit.text(),before)
            self.assertEqual(len(sessions),1)

    def test_selector_waits_for_running_single_analysis_and_uses_latest_selection(self):
        w=self.window();w.time_spin.setValue(.3);w.multipv_spin.setValue(5)
        w._start_analysis();old_worker=w.worker
        w.white_turn_button.click();w.black_turn_button.click()
        self.assertTrue(w._selector_pending)
        self.wait(lambda:w.worker is None and w._live_search_finalized and bool(w.board_widget.analysis_moves))
        self.assertTrue(all(w.board_widget.board.piece_at(m.from_square).color==chess.BLACK for m in w.board_widget.analysis_moves))
        self.assertIsNotNone(old_worker)

    def test_vertical_colors_and_scores_follow_orientation_not_side_to_move(self):
        bar=EvaluationBar();bar.resize(34,400);bar.show();self.addCleanup(bar.close)
        for orientation in ('white','black'):
            bar.set_orientation(orientation)
            for cp in (0,50,200,-150,500):
                bar.set_score(cp);self.app.processEvents()
                self.assertEqual(bar.score_label.text(),f'{cp/100:+.1f}' if cp else '0.0')
                self.assertEqual(bar.value()>50,cp>0)
                pixels=bar.grab().toImage()
                top=bar.BLACK_COLOR if orientation=='white' else bar.WHITE_COLOR
                bottom=bar.WHITE_COLOR if orientation=='white' else bar.BLACK_COLOR
                self.assertEqual(pixels.pixelColor(3,3).name(),top)
                self.assertEqual(pixels.pixelColor(3,396).name(),bottom)
            for mate in (-4,3):
                bar.set_score(None,mate);self.app.processEvents()
                self.assertEqual(bar.segment_fractions(),(1.,0.) if mate<0 else (0.,1.))
                self.assertIn(f'M{abs(mate)}',bar.score_label.text())

    def test_mapping_is_smooth_bounded_monotone_and_height_matches_board_on_resize(self):
        bar=EvaluationBar();previous=0
        for cp in range(-10000,10001,25):
            bar.set_score(cp);self.assertGreaterEqual(bar.value(),previous)
            self.assertTrue(0<=bar.value()<=100);previous=bar.value()
        bar.set_score(50);self.assertLessEqual(bar.value(),57)
        w=self.window();w.show()
        for width,height in ((1100,900),(1600,1050),(1200,800)):
            w.resize(width,height);self.app.processEvents()
            self.assertEqual(w.eval_bar.height(),w.board_widget.height())
            self.assertEqual(w.board_widget.height(),w.board_widget.width())
            self.assertEqual(w.eval_bar.y(),w.board_widget.y())
            self.assertEqual(w.board_widget.x()-w.eval_bar.geometry().right(),11)
        before=w.board_widget.board.fen()
        with patch.object(w,'_request_live_analysis') as request:
            w.board_widget.set_orientation('black');w.eval_bar.set_score(-150)
            self.assertEqual(w.eval_bar.orientation,'black')
            self.assertEqual(w.board_widget.board.fen(),before);request.assert_not_called()
