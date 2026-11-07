"""Regression tests for visual-only transitions and equivalent fast image hashes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import hashlib
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import chess
from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtGui import QImage, QColor
from tests.qt_test_support import test_application
from ajedrez.ui.board_widget import BoardWidget, piece_renderer
from ajedrez.ui.main_window import MainWindow, PositionRecognitionWorker
from ajedrez.recognition import recognize_position, recognize_position_auto_orientation, remap_recognition, _load_classifier


def old_signature(image):
    pixels=bytearray()
    for y in range(64):
        sy=min(image.height()-1,(2*y+1)*image.height()//128)
        for x in range(64):
            sx=min(image.width()-1,(2*x+1)*image.width()//128)
            pixels.extend(int(image.pixel(sx,sy)).to_bytes(4,'little'))
    return hashlib.blake2b(pixels,digest_size=12).digest()


class LatencyAndAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def new_board(self, before, uci, orientation='white'):
        widget=BoardWidget();widget.resize(480,480)
        widget.set_orientation(orientation);widget.set_board(before)
        after=before.copy(stack=True);after.push_uci(uci)
        widget.set_board(after)
        self.addCleanup(widget.close)
        return widget,after

    def test_vectorized_signature_is_byte_identical_on_real_images_and_formats(self):
        image=QImage(str(Path(__file__).parent/'fixtures/chesscom_moved_position.jpg'))
        for format in (QImage.Format.Format_RGB32,QImage.Format.Format_ARGB32,
                       QImage.Format.Format_ARGB32_Premultiplied,QImage.Format.Format_RGBA8888,
                       QImage.Format.Format_RGB888,QImage.Format.Format_Grayscale8):
            for width in (257,506,800):
                with self.subTest(format=format,width=width):
                    sample=image.scaled(width,width+1).convertToFormat(format)
                    self.assertEqual(MainWindow._image_signature(sample),old_signature(sample))
        sample=QImage(257,258,QImage.Format.Format_ARGB32_Premultiplied)
        sample.fill(QColor(80,100,120,90))
        self.assertEqual(MainWindow._image_signature(sample),old_signature(sample))

    def test_signature_detects_exact_same_sample_changes(self):
        image=QImage(256,256,QImage.Format.Format_RGB32);image.fill(QColor('white'))
        before=MainWindow._image_signature(image)
        image.setPixelColor(2,2,QColor('black'))
        self.assertNotEqual(before,MainWindow._image_signature(image))
        self.assertEqual(MainWindow._image_signature(image),old_signature(image))

    def test_normal_move_updates_logic_before_animation_and_keeps_current_arrows(self):
        widget,after=self.new_board(chess.Board(),'e2e4')
        self.assertEqual(widget.board.fen(),after.fen())
        self.assertEqual(widget._animation.state().name,'Running')
        self.assertEqual(widget._animated_pieces,((chess.E2,chess.E4,chess.Piece(chess.PAWN,True)),))
        legal=list(after.legal_moves)[:5]
        widget.set_best_moves(m.uci() for m in legal)
        widget._animation.setCurrentTime(70)
        self.assertAlmostEqual(widget._animation_progress,.5)
        self.assertEqual(widget.analysis_moves,tuple(legal))
        self.assertEqual(widget.board.fen(),after.fen())

    def test_capture_has_no_duplicate_or_captured_piece(self):
        before=chess.Board('4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1')
        widget,after=self.new_board(before,'e4d5')
        self.assertEqual(widget.board.piece_at(chess.D5),chess.Piece(chess.PAWN,True))
        self.assertIsNone(widget.board.piece_at(chess.E4))
        self.assertEqual(len(widget._animated_pieces),1)
        widget.grab()
        self.assertEqual(widget.board.fen(),after.fen())

    def test_four_castles_animate_king_and_rook_in_both_views(self):
        for turn in ('w','b'):
            for file in ('g','c'):
                for view in ('white','black'):
                    rank='1' if turn=='w' else '8'
                    widget,after=self.new_board(chess.Board(f'r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1'),f'e{rank}{file}{rank}',view)
                    self.assertEqual(len(widget._animated_pieces),2)
                    self.assertEqual(widget.board.fen(),after.fen())
                    self.assertEqual({p.piece_type for _,_,p in widget._animated_pieces},{chess.KING,chess.ROOK})
                    self.assertEqual(widget.board.turn,turn!='w')

    def test_promotions_render_resulting_piece_for_both_colors(self):
        for fen,uci,symbol in [('4k3/P7/8/8/8/8/8/4K3 w - - 0 1','a7a8q','Q'),
                               ('4k3/8/8/8/8/8/p7/4K3 b - - 0 1','a2a1n','n')]:
            widget,after=self.new_board(chess.Board(fen),uci)
            self.assertEqual(widget._animated_pieces[0][2].symbol(),symbol)
            self.assertEqual(widget.board.fen(),after.fen());widget.grab()

    def test_en_passant_does_not_leave_a_ghost_pawn(self):
        widget,after=self.new_board(chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),'e5d6')
        self.assertIsNone(widget.board.piece_at(chess.D5))
        self.assertIsNone(widget.board.piece_at(chess.E5))
        self.assertEqual(widget._animated_pieces[0][:2],(chess.E5,chess.D6))
        widget.grab();self.assertEqual(widget.board.fen(),after.fen())

    def test_fast_consecutive_moves_replace_animation_without_queue(self):
        widget,board=self.new_board(chess.Board(),'e2e4')
        widget._animation.setCurrentTime(50)
        board.push_uci('e7e5');widget.set_board(board)
        self.assertEqual(widget._animated_pieces[0][:2],(chess.E7,chess.E5))
        self.assertEqual(len(widget._animated_pieces),1)
        self.assertEqual(widget._animation_progress,0)
        self.assertEqual(widget.board.fen(),board.fen())
        widget._animation.setCurrentTime(widget.animation_duration_ms)
        self.assertEqual(widget._animated_pieces,())
        self.assertEqual(widget.board.fen(),board.fen())

    def test_unknown_resync_or_multiple_plies_never_invent_a_path(self):
        widget,board=self.new_board(chess.Board(),'e2e4')
        board.push_uci('e7e5');board.push_uci('g1f3')
        widget.set_board(board)
        self.assertEqual(widget._animated_pieces,())
        self.assertEqual(widget._animation.state().name,'Stopped')
        single=board.copy();single.push_uci('b8c6')
        widget.set_board(single.copy(stack=False))
        self.assertEqual(widget._animated_pieces,())

    def test_same_position_does_not_restart_animation(self):
        widget,board=self.new_board(chess.Board(),'e2e4')
        widget._animation.setCurrentTime(70);progress=widget._animation_progress
        widget.set_board(board)
        self.assertEqual(widget._animation_progress,progress)

    def test_view_change_cancels_motion_but_keeps_legal_arrows_and_fen(self):
        widget,board=self.new_board(chess.Board(),'e2e4')
        widget.set_best_move('e7e5');widget.set_orientation('black')
        self.assertEqual(widget._animated_pieces,())
        self.assertEqual(widget.best_move,chess.Move.from_uci('e7e5'))
        self.assertEqual(widget._display_coords(chess.E7),(3,6))
        self.assertEqual(widget.board.fen(),board.fen())

    def test_piece_rasters_are_cached_across_frames_and_bounded_on_resize(self):
        widget=BoardWidget();widget.resize(480,480);self.addCleanup(widget.close)
        with patch('ajedrez.ui.board_widget.piece_renderer',wraps=piece_renderer) as render:
            widget.grab();first=render.call_count
            for _ in range(5):widget.grab()
            self.assertEqual(first,12);self.assertEqual(render.call_count,first)
            widget.resize(560,560);widget.grab()
            self.assertEqual(render.call_count,24)
            self.assertEqual(len(widget._piece_pixmaps),12)

    def test_editor_does_not_animate_piece_corrections(self):
        widget=BoardWidget();widget.editable=True;self.addCleanup(widget.close)
        board=chess.Board();board.push_uci('e2e4');widget.set_board(board)
        self.assertEqual(widget._animated_pieces,())

    def test_animation_emits_multiple_interpolated_frames_without_blocking_qt(self):
        widget,after=self.new_board(chess.Board(),'e2e4');widget.show()
        values=[];widget._animation.valueChanged.connect(values.append)
        heartbeat=[];timer=QTimer();timer.setInterval(5);timer.timeout.connect(lambda:heartbeat.append(time.perf_counter()))
        loop=QEventLoop();QTimer.singleShot(220,loop.quit);timer.start();loop.exec();timer.stop()
        self.assertGreaterEqual(len(values),4)
        self.assertGreaterEqual(len(heartbeat),10)
        self.assertEqual(widget._animated_pieces,())
        self.assertEqual(widget.board.fen(),after.fen())

    def test_auto_orientation_keeps_identical_predictions_in_both_views(self):
        image=QImage(str(Path(__file__).parent/'fixtures/chesscom_moved_position.jpg'))
        white=recognize_position(image)
        auto=recognize_position_auto_orientation(image)
        self.assertEqual(auto.piece_placement,white.piece_placement)
        self.assertEqual(remap_recognition(white,'black').piece_placement,recognize_position(image,'black').piece_placement)

    def test_reused_onnx_session_sleeps_instead_of_spinning_between_frames(self):
        from ajedrez.paths import resource_root
        path=str((resource_root()/'models/chess-tiles-v2.onnx').resolve())
        session=_load_classifier(path)
        self.assertIs(session,_load_classifier(path))
        options=session.get_session_options()
        self.assertEqual(options.get_session_config_entry('session.intra_op.allow_spinning'),'0')
        self.assertEqual(options.get_session_config_entry('session.inter_op.allow_spinning'),'0')

    def test_busy_worker_never_queues_an_old_capture(self):
        window=MainWindow();self.addCleanup(window.close)
        window._live_monitoring=True;window._live_screen=object();window._live_region=object()
        window.recognition_worker=object()
        try:
            with patch('ajedrez.ui.main_window.capture_screen') as capture:
                for _ in range(30):window._live_capture_tick()
                capture.assert_not_called()
        finally:window.recognition_worker=None
        self.assertEqual(window.live_interval_spin.value(),125)
        self.assertEqual(window._live_timer.timerType(),Qt.TimerType.PreciseTimer)


if __name__=='__main__':unittest.main()
