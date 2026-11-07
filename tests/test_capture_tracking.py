"""Region relocation using real board images and simulated monitor geometries."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from PySide6.QtCore import QRect,QPoint,QEventLoop,QTimer
from PySide6.QtGui import QImage,QPainter,QColor
import chess
from ajedrez.capture.board_region import background_colors,grid_score,locate_board,LocatedRegion
from ajedrez.capture.window_anchor import WindowGeometry,WindowAnchor,to_physical,to_logical,physical_bounds
from ajedrez.capture.screen_capture import ScreenFrame,crop_region
from ajedrez.ui.main_window import MainWindow
from ajedrez.recognition import recognize_position
from ajedrez.live_board import LiveBoardTracker
from tests.qt_test_support import test_application
from tests.image_helpers import board_from_real_tiles
from tests.test_live_board import observation


class Screen:
    def __init__(self,geometry,ratio=1.,name="simulado"):
        self.bounds=geometry;self.ratio=ratio;self.label=name
    def geometry(self):return self.bounds
    def devicePixelRatio(self):return self.ratio
    def name(self):return self.label


def canvas(board,region,size=(1200,900)):
    image=QImage(*size,QImage.Format.Format_RGB32);image.fill(QColor("#303030"))
    painter=QPainter(image);painter.drawImage(region,board);painter.end()
    return image


class CaptureTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();deadline=QTimer()
        poll.setInterval(5);deadline.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None);deadline.timeout.connect(loop.quit)
        poll.start();deadline.start(8000);loop.exec();poll.stop();deadline.stop()
        self.assertTrue(predicate(),"La captura/análisis no se recuperó")

    def test_fixed_region_reproduces_the_moved_window_failure(self):
        board=QImage(str(Path(__file__).parent/'fixtures/chesscom_moved_position.jpg'))
        first=QRect(60,100,400,400);moved=QRect(570,300,320,320)
        frame=ScreenFrame("prueba",QRect(0,0,1200,900),canvas(board,moved),1.)
        colors=background_colors(board)
        self.assertLess(grid_score(crop_region(frame,first),colors),.80)
        self.assertGreater(grid_score(crop_region(frame,moved),colors),.80)

    def test_real_images_relocate_at_multiple_sizes_and_keep_onnx_result(self):
        for name in ('chesscom_moved_position.jpg','chesscom_review_capture_506x507.png'):
            board=QImage(str(Path(__file__).parent/'fixtures'/name))
            colors=background_colors(board);expected=recognize_position(board).piece_placement
            times=[]
            for size in (256,480,720):
                image=canvas(board,QRect(137,93,size,size),size=(1200,920))
                started=time.perf_counter();region=locate_board(image,colors);times.append((time.perf_counter()-started)*1000)
                with self.subTest(name=name,size=size):
                    self.assertIsNotNone(region)
                    self.assertLessEqual(abs(region.x()-137),5);self.assertLessEqual(abs(region.y()-93),5)
                    self.assertLessEqual(abs(region.width()-size),8)
                    self.assertEqual(recognize_position(image.copy(region)).piece_placement,expected)
            print(f"REGION_BENCH {name}: relocation_ms={times}")

    def test_excludes_the_analyzer_board_and_refuses_multiple_boards(self):
        board=QImage(str(Path(__file__).parent/'fixtures/chesscom_moved_position.jpg'));colors=background_colors(board)
        first=QRect(70,90,320,320);second=QRect(650,350,400,400)
        image=canvas(board,first);p=QPainter(image);p.drawImage(second,board);p.end()
        self.assertIsNone(locate_board(image,colors))
        found=locate_board(image,colors,(first.adjusted(-10,-10,10,10),))
        self.assertIsNotNone(found);self.assertLessEqual(abs(found.x()-second.x()),5)
        self.assertIsNone(locate_board(image,colors,(image.rect(),)))

    def test_blank_occluded_and_non_square_images_do_not_pass_geometry(self):
        board=QImage(str(Path(__file__).parent/'fixtures/chesscom_moved_position.jpg'));colors=background_colors(board)
        image=QImage(400,400,QImage.Format.Format_RGB32);image.fill(QColor("#303030"))
        self.assertEqual(grid_score(image,colors),0.)
        self.assertIsNone(locate_board(image,colors))
        self.assertEqual(grid_score(board.scaled(600,300),colors),0.)
        self.assertIsNone(background_colors(image))

    def test_dpi_and_negative_monitor_coordinates_roundtrip(self):
        for origin in ((0,0),(-1920,0),(1920,-900)):
            for ratio in (1.,1.25,1.5,2.):
                screen=Screen(QRect(*origin,960,600),ratio)
                rect=QRect(origin[0]+120,origin[1]+80,320,320)
                physical=to_physical(rect,screen)
                self.assertEqual(to_logical(physical,screen),rect)
                self.assertTrue(physical_bounds(screen).contains(physical))

    def test_window_move_and_dpi_prediction_does_not_depend_on_maximizing(self):
        initial=WindowGeometry(QRect(100,50,1200,900),96,1)
        anchor=WindowAnchor(1,initial,QRect(220,150,400,400))
        self.assertEqual(anchor.predicted(WindowGeometry(QRect(-1800,70,1200,900),96,1)),QRect(-1680,170,400,400))
        self.assertEqual(anchor.predicted(WindowGeometry(QRect(1920,0,1800,1350),144,1)),QRect(2100,150,600,600))

    def test_relocation_retains_history_and_real_stockfish_updates_after_move(self):
        before=chess.Board();after=before.copy();after.push_san("e4")
        first=Screen(QRect(0,0,1200,900),name="primero")
        second=Screen(QRect(-1200,0,1200,900),name="segundo")
        old_region=QRect(90,100,400,400);new_local=QRect(240,170,320,320)
        board_image=board_from_real_tiles(before)
        state=WindowGeometry(first.geometry(),96,99)
        new_state=WindowGeometry(second.geometry(),96,99)
        image=canvas(board_from_real_tiles(after),new_local)
        w=MainWindow()
        try:
            w._live_monitoring=True;w._live_generation=920
            w._live_screen=first;w._live_region=old_region
            w._live_region_colors=background_colors(board_image)
            w._live_anchor=WindowAnchor(1,state,old_region);w._live_window_geometry=state
            w._live_tracker=LiveBoardTracker(before);tracker=w._live_tracker
            w.time_spin.setValue(.1)
            def capture(screen):
                return ScreenFrame(screen.name(),screen.geometry(),image,1.)
            with patch("ajedrez.capture.window_anchor.WindowAnchor.geometry",return_value=new_state), \
                 patch("ajedrez.ui.main_window.QGuiApplication.screens",return_value=[first,second]), \
                 patch("ajedrez.ui.main_window.capture_screen",side_effect=capture), \
                 patch("ajedrez.ui.main_window.left_mouse_button_down",return_value=False):
                w._live_capture_tick()
                self.assertEqual(w._live_tracker.board.fen(),before.fen())
                self.wait(lambda:w._live_region_worker is None and w._live_region_valid)
                self.assertIs(w._live_screen,second)
                self.assertEqual(w._live_generation,920);self.assertIs(w._live_tracker,tracker)
                for _ in range(3):
                    w._live_capture_tick();self.wait(lambda:w.recognition_worker is None)
                self.assertEqual(w._live_tracker.board.fen(),after.fen())
                self.assertEqual(w._live_tracker.board.move_stack,after.move_stack)
                self.wait(lambda:w._live_search_finalized and len(w.board_widget.analysis_moves)==5)
                self.assertEqual(w._live_search_ticket.fen,after.fen())
        finally:w.close();self.app.processEvents()

    def test_geometry_change_discards_an_old_recognition_without_resetting_game(self):
        board=chess.Board();after=board.copy();after.push_san("e4")
        state=WindowGeometry(QRect(0,0,1200,900),96,99)
        screen=Screen(state.client)
        w=MainWindow()
        try:
            w._live_monitoring=True;w._live_generation=9;w._live_screen=screen;w._live_region=QRect(30,40,400,400)
            w._live_tracker=LiveBoardTracker(board);w._live_anchor=WindowAnchor(1,state,w._live_region)
            w._live_window_geometry=state;w._live_recognition_geometry_revision=0
            moved=WindowGeometry(QRect(20,30,1000,800),96,99)
            with patch("ajedrez.capture.window_anchor.WindowAnchor.geometry",return_value=moved), \
                 patch("ajedrez.ui.main_window.QGuiApplication.screens",return_value=[screen]), \
                 patch.object(w,"_request_live_analysis") as analyze:
                w._recognition_completed(observation(after),("live_poll",9))
                analyze.assert_not_called()
            self.assertEqual(w._live_tracker.board.fen(),board.fen());self.assertEqual(w._live_generation,9)
        finally:w.close()

    def test_invalid_crop_never_enters_onnx_and_search_is_throttled(self):
        board=board_from_real_tiles(chess.Board());screen=Screen(QRect(0,0,1200,900))
        blank=canvas(board,QRect(700,400,300,300))
        w=MainWindow()
        try:
            w._live_monitoring=True;w._live_screen=screen;w._live_region=QRect(0,0,400,400)
            w._live_region_colors=background_colors(board);w._live_tracker=LiveBoardTracker(chess.Board())
            with patch("ajedrez.ui.main_window.capture_screen",return_value=ScreenFrame("s",screen.geometry(),blank,1.)), \
                 patch("ajedrez.ui.main_window.left_mouse_button_down",return_value=False), \
                 patch.object(w,"_start_recognition") as recognize,patch.object(w,"_search_live_region") as search:
                w._live_capture_tick();recognize.assert_not_called();search.assert_called_once()
            w._live_region_retry_at=time.monotonic()+10
            with patch("ajedrez.ui.main_window.capture_screen") as grab:
                w._search_live_region(None);grab.assert_not_called()
            self.assertEqual(w._live_tracker.board.fen(),chess.STARTING_FEN)
        finally:w.close()

    def test_minimized_source_and_stale_relocation_keep_confirmed_board(self):
        screen=Screen(QRect(0,0,1200,900));state=WindowGeometry(screen.geometry(),96,99)
        w=MainWindow()
        try:
            w._live_monitoring=True;w._live_screen=screen;w._live_region=QRect(0,0,400,400)
            w._live_tracker=LiveBoardTracker(chess.Board());w._live_anchor=WindowAnchor(1,state,w._live_region)
            w._live_window_geometry=state
            with patch("ajedrez.capture.window_anchor.WindowAnchor.geometry",return_value=None), \
                 patch("ajedrez.ui.main_window.capture_screen") as capture:
                w._live_capture_tick();capture.assert_not_called()
            w._live_region_found(LocatedRegion(0,QRect(100,100,400,400),1.),(-1,-1,None),[screen],None)
            self.assertEqual(w._live_region,QRect(0,0,400,400));self.assertEqual(w._live_tracker.board.fen(),chess.STARTING_FEN)
        finally:w.close()

    def test_selection_binds_source_before_restore_and_keeps_initial_retry_active(self):
        screen=Screen(QRect(0,0,1200,900));image=board_from_real_tiles(chess.Board())
        frame=ScreenFrame("s",screen.geometry(),canvas(image,QRect(40,60,400,400)),1.)
        w=MainWindow()
        try:
            w._capture_screen=screen;w._live_setup_pending=True
            with patch("ajedrez.ui.main_window.WindowAnchor.bind",return_value=None) as bind, \
                 patch.object(w,"_restore_after_capture"),patch.object(w,"_start_recognition") as recognize:
                w._capture_selected(frame,image,QRect(40,60,400,400))
                bind.assert_called_once();recognize.assert_called_once_with(image,"auto","live_setup")
            self.assertTrue(w._live_monitoring);self.assertTrue(w._live_timer.isActive())
            self.assertIsNotNone(w._live_region_colors)
        finally:w.close()

    def test_steady_region_has_one_capture_per_tick_and_no_relocation_search(self):
        image=board_from_real_tiles(chess.Board());screen=Screen(QRect(0,0,800,800))
        w=MainWindow()
        try:
            w._live_monitoring=True;w._live_screen=screen;w._live_region=screen.geometry()
            w._live_region_colors=background_colors(image)
            with patch("ajedrez.ui.main_window.left_mouse_button_down",return_value=False), \
                 patch("ajedrez.ui.main_window.capture_screen",return_value=ScreenFrame("s",screen.geometry(),image,1.)) as capture, \
                 patch.object(w,"_start_recognition") as recognize,patch.object(w,"_search_live_region") as search:
                w._live_capture_tick();w._live_capture_tick();w._live_signature_confirmed=True;w._live_capture_tick()
                self.assertEqual(capture.call_count,3);self.assertEqual(recognize.call_count,2);search.assert_not_called()
        finally:w.close()

    def test_paused_or_previous_session_relocation_cannot_change_region(self):
        screen=Screen(QRect(0,0,1200,900));w=MainWindow();source=object()
        try:
            w._live_monitoring=True;w._live_region_worker=source;w._live_region=QRect(40,60,400,400)
            original=QRect(w._live_region);match=LocatedRegion(0,QRect(200,200,320,320),1.)
            token=(w._live_generation,w._live_geometry_revision,None)
            w._live_paused=True;w._live_region_found(match,token,[screen],source)
            self.assertEqual(w._live_region,original)
            w._live_paused=False;w._live_generation+=1;w._live_region_found(match,token,[screen],source)
            self.assertEqual(w._live_region,original)
        finally:w._live_region_worker=None;w.close()

    def test_relocation_grid_and_recognition_preserve_black_orientation(self):
        board=chess.Board();board.push_san("e4");board.push_san("c5")
        image=board_from_real_tiles(board,"black");colors=background_colors(image)
        screenshot=canvas(image,QRect(170,230,480,480))
        rect=locate_board(screenshot,colors)
        self.assertIsNotNone(rect)
        self.assertEqual(recognize_position(screenshot.copy(rect),"black").piece_placement,board.board_fen())

    def test_binding_prefers_actual_hit_test_over_a_full_desktop_overlay(self):
        from types import SimpleNamespace
        screen=Screen(QRect(0,0,1200,900));region=QRect(70,120,400,400)
        state=WindowGeometry(QRect(30,40,680,820),96,os.getpid()+100)
        api=SimpleNamespace(WindowFromPoint=lambda point:7,GetAncestor=lambda handle,mode:7)
        with patch("ajedrez.capture.window_anchor.WindowAnchor._api",return_value=api), \
             patch("ajedrez.capture.window_anchor.WindowAnchor.geometry",return_value=state):
            anchor=WindowAnchor.bind(region,screen)
        self.assertEqual(anchor.handle,7)
        self.assertEqual(anchor.region,region)
