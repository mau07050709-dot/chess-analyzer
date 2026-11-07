"""Real-image and controlled-image regressions for complementary turn recovery."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
import chess
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtGui import QImage,QColor,QPainter
from ajedrez.board_coordinates import square_to_grid
from ajedrez.capture.board_region import rgb
from ajedrez.last_move_highlight import (detect_highlights,HighlightedCells,resolve_last_move,HighlightTurnRecovery)
from ajedrez.live_board import LiveBoardTracker
from ajedrez.recognition import recognize_position
from ajedrez.ui.main_window import MainWindow,PositionRecognitionWorker
from ajedrez.engine import AnalysisResult
from tests.image_helpers import board_from_real_tiles,reverse_cells
from tests.qt_test_support import test_application
from tests.test_live_board import observation

FIXTURES=Path(__file__).parent/'fixtures'
REFERENCE='r2q1rk1/p3np1p/1pnpp1p1/8/2P5/2P1P1P1/P3NP1P/R1BQ1RK1'


def cells_for(move,orientation="white"):
    return HighlightedCells(tuple((square_to_grid(q,orientation)[1],square_to_grid(q,orientation)[0])
                                 for q in (move.from_square,move.to_square)))


def painted_highlight(board,squares,orientation="white",color=None):
    """Synthetic variation from real tiles; never claimed as a real screenshot."""
    pixels=rgb(board_from_real_tiles(board,orientation))
    old=np.asarray(((235,236,208),(115,149,82),(185,202,67),(245,246,130),(246,249,94)),dtype=np.float32)
    for row in range(8):
        for col in range(8):
            tile=pixels[row*100:(row+1)*100,col*100:(col+1)*100]
            mask=np.linalg.norm(tile.astype(np.float32)[:,:,None,:]-old[None,None,:,:],axis=3).min(axis=2)<32
            square=chess.square(col,7-row) if orientation=="white" else chess.square(7-col,row)
            parity=(row+col)%2
            background=((235,236,208),(115,149,82))[parity]
            if square in squares:background=color or ((245,246,130),(185,202,67))[parity]
            tile[mask]=background
    return QImage(pixels.data,800,800,pixels.strides[0],QImage.Format.Format_RGB888).copy()


class LastMoveHighlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();deadline=QTimer()
        poll.setInterval(5);deadline.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None);deadline.timeout.connect(loop.quit)
        poll.start();deadline.start(8000);loop.exec();poll.stop();deadline.stop()
        self.assertTrue(predicate(),"No llegó la evaluación con turno recuperado")

    def test_attached_real_image_identifies_black_knight_b8_c6(self):
        image=QImage(str(FIXTURES/'phase13_3_last_black_knight.png'))
        highlight=detect_highlights(image)
        self.assertEqual(set(highlight.squares("white")),{chess.B8,chess.C6})
        evidence=resolve_last_move(chess.Board(REFERENCE+' b - - 0 1'),highlight,"white")
        self.assertEqual((evidence.origin,evidence.destination,evidence.mover,evidence.next_turn),
                         (chess.B8,chess.C6,chess.BLACK,chess.WHITE))
        # The supplied image has an outer margin. An ordinary selected board
        # crop removes it; no piece probabilities/thresholds are changed.
        recognized=recognize_position(image.copy(5,0,816,810))
        self.assertEqual(recognized.piece_placement,REFERENCE);self.assertEqual(recognized.uncertain_squares,())

    def test_two_other_real_captures_identify_black_knight_g8_f6(self):
        for name in ('chesscom_moved_position.jpg','chesscom_review_capture_506x507.png'):
            with self.subTest(name=name):
                image=QImage(str(FIXTURES/name));h=detect_highlights(image);r=recognize_position(image)
                b=chess.Board(r.piece_placement+' b - - 0 1')
                e=resolve_last_move(b,h,"white")
                self.assertEqual((e.origin,e.destination,e.next_turn),(chess.G8,chess.F6,chess.WHITE))

    def test_white_last_move_and_both_orientations_at_multiple_sizes(self):
        board=chess.Board();move=board.parse_san('e4');board.push(move)
        for orientation in ("white","black"):
            image=painted_highlight(board,(move.from_square,move.to_square),orientation)
            for side in (256,480,800):
                with self.subTest(orientation=orientation,side=side):
                    h=detect_highlights(image.scaled(side,side));e=resolve_last_move(board,h,orientation)
                    self.assertIsNotNone(e);self.assertEqual(e.next_turn,chess.BLACK)
                    self.assertEqual((e.origin,e.destination),(chess.E2,chess.E4))

    def test_real_image_orientation_flip_keeps_piece_and_turn_meanings(self):
        image=QImage(str(FIXTURES/'chesscom_moved_position.jpg'))
        board=chess.Board(recognize_position(image).piece_placement+' w - - 0 1')
        h=detect_highlights(reverse_cells(image));e=resolve_last_move(board,h,"black")
        self.assertEqual((e.origin,e.destination,e.next_turn),(chess.G8,chess.F6,chess.WHITE))

    def test_color_tolerance_does_not_depend_on_one_rgb(self):
        board=chess.Board();board.push_san('e4')
        for color in ((237,239,119),(194,211,75),(210,213,92)):
            image=painted_highlight(board,(chess.E2,chess.E4),color=color)
            self.assertEqual(set(detect_highlights(image).squares("white")),{chess.E2,chess.E4})

    def test_red_premove_single_selection_and_extra_yellow_square_are_rejected(self):
        board=chess.Board();board.push_san('e4')
        for squares,color in (((chess.E2,chess.E4),(200,60,60)),((chess.E4,),None),
                              ((chess.E2,chess.E4,chess.A3),None)):
            with self.subTest(squares=squares,color=color):
                self.assertIsNone(detect_highlights(painted_highlight(board,squares,color=color)))

    def test_center_dots_and_user_arrows_do_not_cover_the_background_corners(self):
        board=chess.Board();image=painted_highlight(board,())
        painter=QPainter(image);painter.setBrush(QColor('yellow'))
        painter.drawEllipse(430,430,40,40);painter.drawRect(445,240,10,350);painter.end()
        self.assertIsNone(detect_highlights(image))

    def test_yellow_board_theme_is_not_a_last_move(self):
        image=QImage(800,800,QImage.Format.Format_RGB32);image.fill(QColor(200,205,70))
        self.assertIsNone(detect_highlights(image))

    def test_invalid_move_two_occupied_and_two_empty_squares_are_ambiguous(self):
        board=chess.Board()
        for uci in ('a3b3','a2b2','a3a2'):
            self.assertIsNone(resolve_last_move(board,cells_for(chess.Move.from_uci(uci)),"white"))

    def test_unchanged_yellow_premove_or_stale_markers_cannot_override_an_anchor(self):
        board=chess.Board();board.push_san('Nf3')
        # A proposed f3-g5 still has the piece at f3. Reverse geometry g5-f3
        # alone would be plausible, so the unchanged confirmed anchor is vital.
        hint=cells_for(chess.Move.from_uci('f3g5'))
        self.assertIsNone(resolve_last_move(board,hint,"white",anchor=board))

    def test_captures_pawns_en_passant_and_promotions_for_both_colors(self):
        cases=(('4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1','e4d5'),
               ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1','e5d6'),
               ('4k3/6P1/8/8/8/8/8/4K3 w - - 0 1','g7g8q'))
        for fen,uci in cases:
            original=chess.Board(fen);move=chess.Move.from_uci(uci)
            for mirror in (False,True):
                board=original.mirror() if mirror else original.copy()
                m=chess.Move(chess.square_mirror(move.from_square),chess.square_mirror(move.to_square),promotion=move.promotion) if mirror else move
                self.assertIn(m,board.legal_moves);mover=board.turn;board.push(m)
                saved=board.fen(en_passant='fen')
                e=resolve_last_move(board,cells_for(m),"white")
                self.assertIsNotNone(e);self.assertEqual(e.next_turn,not mover)
                self.assertEqual(board.fen(en_passant='fen'),saved)

    def test_all_four_castles_use_king_color_without_restoring_rights(self):
        for turn in ('w','b'):
            for target in ('g','c'):
                board=chess.Board(f'r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 7 12')
                rank='1' if turn=='w' else '8';move=chess.Move.from_uci('e'+rank+target+rank)
                mover=board.turn;board.push(move);board.castling_rights=0
                saved=board.fen()
                e=resolve_last_move(board,cells_for(move),"white")
                self.assertIsNotNone(e);self.assertEqual(e.next_turn,not mover)
                self.assertEqual(board.fen(),saved);self.assertEqual(board.castling_rights,0)

    def test_temporal_gate_requires_two_stable_matching_captures(self):
        board=chess.Board();board.push_san('e4');hint=cells_for(chess.Move.from_uci('e2e4'))
        gate=HighlightTurnRecovery()
        self.assertIsNone(gate.observe(board,hint,"white",(1,1),visual_stable=False))
        self.assertIsNone(gate.observe(board,hint,"white",(1,1),visual_stable=False))
        self.assertIsNotNone(gate.observe(board,hint,"white",(1,1),visual_stable=True))

    def test_new_geometry_position_or_absent_markers_discard_pending_evidence(self):
        board=chess.Board();board.push_san('e4');h=cells_for(chess.Move.from_uci('e2e4'));gate=HighlightTurnRecovery()
        self.assertIsNone(gate.observe(board,h,"white",(1,1),visual_stable=True))
        self.assertIsNone(gate.observe(board,h,"white",(1,2),visual_stable=True))
        self.assertIsNone(gate.observe(board,None,"white",(1,2),visual_stable=True));self.assertFalse(gate.pending)
        after=board.copy();after.push_san('e5');new=cells_for(chess.Move.from_uci('e7e5'))
        self.assertIsNone(gate.observe(after,new,"white",(1,2),visual_stable=True))

    def test_confirm_only_changes_unknown_turn_and_keeps_metadata_and_history(self):
        board=chess.Board();board.push_san('e4')
        tracker=LiveBoardTracker(board,turn_known=False)
        saved=(tracker.board.board_fen(),tracker.board.castling_rights,tracker.board.ep_square,
               tracker.board.halfmove_clock,tracker.board.fullmove_number,list(tracker.board.move_stack))
        self.assertTrue(tracker.confirm_visual_turn(chess.BLACK,board.fen()))
        self.assertEqual((tracker.board.board_fen(),tracker.board.castling_rights,tracker.board.ep_square,
                          tracker.board.halfmove_clock,tracker.board.fullmove_number,list(tracker.board.move_stack)),saved)
        self.assertFalse(tracker.confirm_visual_turn(chess.WHITE,tracker.board.fen()))
        other=LiveBoardTracker(chess.Board(),turn_known=False)
        self.assertFalse(other.confirm_visual_turn(chess.BLACK,'old FEN'))

    def window_for(self,board,orientation='white'):
        w=MainWindow();self.addCleanup(w.close)
        w._live_monitoring=True;w._live_generation=33;w._live_frame_is_stable=True
        w._live_orientation=orientation;w._live_tracker=LiveBoardTracker(board,turn_known=False)
        w.board_widget.set_board(board);w.fen_edit.setText(board.fen());w.time_spin.setValue(.1)
        return w

    def test_real_onnx_and_stockfish_resume_five_arrows_without_another_move(self):
        original=QImage(str(FIXTURES/'phase13_3_last_black_knight.png')).copy(5,0,816,810)
        for orientation in ('white','black'):
            with self.subTest(orientation=orientation):
                image=original if orientation=='white' else reverse_cells(original)
                r=recognize_position(image,orientation);self.assertEqual(r.uncertain_squares,())
                h=detect_highlights(image);board=chess.Board(r.piece_placement+' b - - 9 18')
                w=self.window_for(board,orientation)
                try:
                    with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
                        w._start_recognition(image,orientation,'live_poll')
                        self.wait(lambda:w.recognition_worker is None)
                        self.assertFalse(w._live_tracker.turn_known);self.assertFalse(w._live_signature_confirmed)
                        w._start_recognition(image,orientation,'live_poll')
                        self.wait(lambda:w.recognition_worker is None)
                    self.assertTrue(w._live_tracker.turn_known);self.assertEqual(w._live_tracker.board.turn,chess.WHITE)
                    self.assertEqual(w._live_tracker.board.castling_rights,0);self.assertIsNone(w._live_tracker.board.ep_square)
                    self.assertEqual(w._live_tracker.board.halfmove_clock,9);self.assertEqual(w._live_tracker.board.fullmove_number,18)
                    self.assertEqual(w._live_tracker.board.move_stack,[])
                    self.wait(lambda:w._live_search_finalized and len(w.board_widget.analysis_moves)==5)
                    current=w._live_tracker.board
                    self.assertEqual(w._live_search_ticket.fen,current.fen())
                    self.assertEqual(w.fen_edit.text(),current.fen())
                    self.assertTrue(all(move in current.legal_moves for move in w.board_widget.analysis_moves))
                    saved=(w.eval_label.text(),w.board_widget.analysis_moves)
                    w._live_analysis_completed(board.fen(),AnalysisResult(board.fen(),'old',()),0,w.live_analysis_worker)
                    self.assertEqual((w.eval_label.text(),w.board_widget.analysis_moves),saved)
                finally:w.close();self.app.processEvents()

    def test_initial_midgame_bootstrap_recovers_on_next_stable_capture_without_dialog(self):
        image=QImage(str(FIXTURES/'chesscom_moved_position.jpg'));r=recognize_position(image);h=detect_highlights(image)
        w=MainWindow();self.addCleanup(w.close);w._live_generation=3;w._live_frame_is_stable=True
        with patch.object(w,'_request_live_analysis') as request,patch.object(w,'_open_recognition_review') as review, \
             patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
            w._recognition_completed(r,('live_setup',3),h)
            self.assertFalse(w._live_tracker.turn_known);self.assertFalse(w._live_signature_confirmed)
            w._recognition_completed(r,('live_poll',w._live_generation),h)
            self.assertTrue(w._live_tracker.turn_known);self.assertEqual(w._live_tracker.board.turn,chess.WHITE)
            request.assert_called_once();review.assert_not_called()

    def test_skipped_many_plies_recovers_from_real_highlight_without_fake_history(self):
        image=QImage(str(FIXTURES/'phase13_3_last_black_knight.png')).copy(5,0,816,810)
        r=recognize_position(image);h=detect_highlights(image)
        w=MainWindow();self.addCleanup(w.close);w._live_monitoring=True;w._live_generation=5
        w._live_frame_is_stable=True;w._live_tracker=LiveBoardTracker(chess.Board())
        with patch.object(w,'_request_live_analysis') as request,patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
            for _ in range(5):w._recognition_completed(r,('live_poll',w._live_generation),h)
            self.assertEqual(w._live_tracker.board.board_fen(),REFERENCE)
            self.assertTrue(w._live_tracker.turn_known);self.assertEqual(w._live_tracker.board.turn,chess.WHITE)
            self.assertEqual(w._live_tracker.board.move_stack,[]);self.assertEqual(w._live_tracker.board.castling_rights,0)
            self.assertGreaterEqual(request.call_count,1)

    def test_known_turn_and_manual_override_are_never_changed_by_highlights(self):
        board=chess.Board();board.push_san('e4');r=observation(board);h=cells_for(chess.Move.from_uci('e2e4'))
        w=self.window_for(board)
        for turn in (chess.WHITE,chess.BLACK):
            with patch.object(w,'_request_live_analysis'):
                (w.white_turn_button if turn else w.black_turn_button).click()
            saved=w._live_tracker.board.fen()
            with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
                for _ in range(3):w._recognition_completed(r,('live_poll',33),h)
            self.assertEqual(w._analysis_turn,turn);self.assertFalse(w._live_tracker.turn_known)
            self.assertEqual(w._live_tracker.board.fen(),saved)
        w._live_tracker=LiveBoardTracker(board);w._analysis_turn=None
        with patch.object(w,'_request_live_analysis') as request:
            w._try_highlight_turn_recovery(r,h);request.assert_not_called()
        self.assertEqual(w._live_tracker.board.turn,chess.BLACK)

    def test_return_to_auto_arms_recovery_without_resetting_tracker(self):
        board=chess.Board();board.push_san('e4');w=self.window_for(board);tracker=w._live_tracker
        with patch.object(w,'_request_live_analysis'):
            w.white_turn_button.click();w._live_signature_confirmed=True;w.auto_turn_button.click()
        self.assertIs(w._live_tracker,tracker);self.assertFalse(w._live_signature_confirmed)
        self.assertIsNone(w._analysis_turn);self.assertFalse(tracker.turn_known)

    def test_stale_generation_geometry_and_pause_cannot_confirm_turn(self):
        board=chess.Board();board.push_san('e4');r=observation(board);h=cells_for(chess.Move.from_uci('e2e4'));w=self.window_for(board)
        with patch.object(w,'_request_live_analysis') as request,patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
            w._recognition_completed(r,('live_poll',32),h)
            self.assertFalse(w._highlight_turn_recovery.pending)
            w._recognition_completed(r,('live_poll',33),h);self.assertFalse(w._live_tracker.turn_known)
            w._live_geometry_revision+=1
            w._recognition_completed(r,('live_poll',33),h);self.assertFalse(w._live_tracker.turn_known)
            w._live_paused=True;w._recognition_completed(r,('live_poll',33),h)
            self.assertFalse(w._live_tracker.turn_known);request.assert_not_called()

    def test_no_highlight_keeps_traditional_recovery_and_unknown_image_skip(self):
        board=chess.Board();board.push_san('e4');w=self.window_for(board);w._live_tracker.board.turn=chess.WHITE
        with patch.object(w,'_request_live_analysis') as request,patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
            w._recognition_completed(observation(w._live_tracker.board),('live_poll',33))
            self.assertFalse(w._live_tracker.turn_known);self.assertTrue(w._live_signature_confirmed)
            after=board.copy();after.push_san('e5')
            for _ in range(2):w._recognition_completed(observation(after),('live_poll',33))
            self.assertTrue(w._live_tracker.turn_known);self.assertEqual(w._live_tracker.board.turn,chess.WHITE)
            request.assert_called_once()

    def test_worker_runs_onnx_once_and_skips_highlights_in_static_or_known_unchanged_mode(self):
        image=QImage(str(FIXTURES/'chesscom_moved_position.jpg'));r=recognize_position(image)
        for enabled,known,expected in ((False,None,0),(True,r.piece_placement,0),(True,None,1)):
            worker=PositionRecognitionWorker(image,'white',recover_turn=enabled,known_placement=known)
            with patch('ajedrez.ui.main_window.recognize_position',return_value=r) as onnx, \
                 patch('ajedrez.ui.main_window.detect_highlights',wraps=detect_highlights) as detector:
                worker.run();onnx.assert_called_once();self.assertEqual(detector.call_count,expected)

    def test_detector_benchmark_after_warmup(self):
        image=QImage(str(FIXTURES/'phase13_3_last_black_knight.png'));detect_highlights(image)
        times=[]
        for _ in range(30):
            start=time.perf_counter();self.assertIsNotNone(detect_highlights(image));times.append((time.perf_counter()-start)*1000)
        print('LAST_MOVE_BENCH average_ms=',sum(times)/len(times),'p95_ms=',sorted(times)[28])

    def test_nonpawn_capture_can_restore_a_predecessor_blocker_only_in_the_probe(self):
        board=chess.Board('8/8/7k/8/8/5N2/3p4/2B1K3 w - - 0 1')
        move=chess.Move.from_uci('f3d2');self.assertIn(move,board.legal_moves);board.push(move)
        saved=board.fen();e=resolve_last_move(board,cells_for(move),'white')
        self.assertIsNotNone(e);self.assertEqual(e.next_turn,chess.BLACK);self.assertEqual(board.fen(),saved)

    def test_doubtful_pieces_and_drag_frames_never_supply_confirmation(self):
        board=chess.Board();board.push_san('e4');w=self.window_for(board)
        hint=cells_for(chess.Move.from_uci('e2e4'))
        with patch.object(w,'_request_live_analysis') as request:
            w._try_highlight_turn_recovery(observation(board,confidence=.2),hint)
            self.assertFalse(w._highlight_turn_recovery.pending)
            with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=True):
                w._recognition_completed(observation(board),('live_poll',33),hint)
            self.assertFalse(w._live_tracker.turn_known);request.assert_not_called()

    def test_manual_worker_finishing_after_return_to_auto_rearms_one_fresh_read(self):
        from types import SimpleNamespace
        board=chess.Board();board.push_san('e4');w=self.window_for(board)
        worker=SimpleNamespace(recover_turn=False,deleteLater=lambda:None)
        w.recognition_worker=worker
        with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
            w._recognition_completed(observation(board),('live_poll',33))
        self.assertTrue(w._live_signature_confirmed)
        w._recognition_worker_finished(worker)
        self.assertFalse(w._live_signature_confirmed)
        self.assertIsNone(w.recognition_worker);self.assertFalse(w._live_tracker.turn_known)
