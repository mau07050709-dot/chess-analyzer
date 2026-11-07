"""Live-only pin overlay, independent from red X and engine state."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest,threading,time
from unittest.mock import patch
import chess
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtGui import QColor,QImage,QPainter
from tests.qt_test_support import test_application
from tests.test_pins import absolute,relative
from ajedrez.pins import find_pins,PinResult
from ajedrez.ui.pin_worker import PinWorker
from ajedrez.ui.board_widget import BoardWidget
from ajedrez.ui.main_window import MainWindow


class PinUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();timer=QTimer();timer.setInterval(2)
        timer.timeout.connect(lambda:loop.quit() if predicate() else None)
        deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
        timer.start();deadline.start(5000)
        if not predicate():loop.exec()
        timer.stop();deadline.stop();self.assertTrue(predicate())

    def window(self):
        window=MainWindow();self.addCleanup(window.close)
        window._request_live_analysis=lambda board:None
        return window

    def test_off_in_manual_mode_and_active_for_historyless_live_board(self):
        window=self.window();board=absolute()
        window.board_widget.set_board(board);window._request_pin_badges(board,True)
        self.assertIsNone(window._pin_worker)
        window._live_monitoring=True;window._apply_live_position(board,'test',turn_known=False)
        self.wait(lambda:chess.E2 in window.board_widget.pinned_squares)
        self.assertEqual(window.fen_edit.text(),board.fen())
        self.assertEqual(window.board_widget.board.fen(),board.fen())

    def test_same_position_is_not_recalculated_and_manual_selector_does_not_affect_pins(self):
        window=self.window();board=absolute();window._live_monitoring=True
        window._apply_live_position(board,'test');self.wait(lambda:chess.E2 in window.board_widget.pinned_squares)
        with patch.object(window._pin_worker,'request_position') as request:
            for turn in (None,True,False):
                window._analysis_turn=turn;window._request_pin_badges(board,True)
            request.assert_not_called()
        self.assertEqual(window.board_widget.board.fen(),board.fen())

    def test_new_position_new_game_and_stopping_reject_old_results(self):
        window=self.window();board=absolute();window._live_monitoring=True
        window._apply_live_position(board,'test');self.wait(lambda:chess.E2 in window.board_widget.pinned_squares)
        token=window._pin_token;result=find_pins(board)
        board.remove_piece_at(chess.E8);window._apply_live_position(board,'test')
        window._pin_badges_ready(token,token[2],result)
        self.assertEqual(window.board_widget.pinned_squares,frozenset())
        window._live_generation+=1
        current=window._pin_token;window._pin_badges_ready(current,current[2],result)
        self.assertEqual(window.board_widget.pinned_squares,frozenset())
        window._stop_live_reading();self.assertIsNone(window._pin_token)
        window._pin_badges_ready(token,token[2],result)
        self.assertEqual(window.board_widget.pinned_squares,frozenset())

    def test_pin_and_x_are_both_drawn_without_overlap_in_both_orientations_and_sizes(self):
        widget=BoardWidget();self.addCleanup(widget.close);widget.set_board(absolute())
        widget.set_pinned_squares([chess.E2]);widget.set_vulnerable_squares([chess.E2])
        for view in ('white','black'):
            widget.set_orientation(view)
            for side in (400,640):
                widget.resize(side,side);cell=side/8
                pin=widget._pin_badge_rect(chess.E2,0,0,cell)
                red=widget._vulnerability_badge_rect(chess.E2,0,0,cell)
                self.assertFalse(pin.intersects(red))
                self.assertAlmostEqual(pin.center().y(),red.center().y())
                self.assertEqual(widget.pinned_squares,frozenset([chess.E2]));widget.grab()

    def test_white_vector_pin_and_dark_circle_are_visible(self):
        widget=BoardWidget();self.addCleanup(widget.close);widget.set_board(absolute())
        widget.set_pinned_squares([chess.E2,chess.E1]);self.assertNotIn(chess.E1,widget.pinned_squares)
        image=QImage(480,480,QImage.Format.Format_RGB32);image.fill(QColor('#eee9df'))
        painter=QPainter(image);widget._draw_pin_badges(painter,0,0,60);painter.end()
        rect=widget._pin_badge_rect(chess.E2,0,0,60)
        colors={image.pixelColor(x,y).name() for y in range(int(rect.top()),int(rect.bottom()))
                for x in range(int(rect.left()),int(rect.right()))}
        self.assertIn('#ffffff',colors);self.assertIn('#363a3d',colors)
        self.assertFalse(widget.set_pinned_squares([chess.E2],expected_fen=chess.STARTING_FEN))

    def test_indicator_follows_existing_animation_then_is_at_destination(self):
        board=absolute('R');widget=BoardWidget();self.addCleanup(widget.close)
        widget.set_board(board);widget.set_pinned_squares([chess.E2])
        board.push_uci('e2e3');widget.set_board(board)
        self.assertEqual(widget.pinned_squares,frozenset())
        widget.set_pinned_squares([chess.E3]);widget._animation.setCurrentTime(70)
        rect=widget._pin_badge_rect(chess.E3,0,0,60)
        a,b=widget._display_coords(chess.E2),widget._display_coords(chess.E3)
        self.assertAlmostEqual(rect.center().y(),((a[1]+b[1])/2+.18)*60)
        widget._animation.setCurrentTime(140)
        self.assertAlmostEqual(widget._pin_badge_rect(chess.E3,0,0,60).center().y(),(b[1]+.18)*60)
        self.assertEqual(widget.board.fen(),board.fen())

    def test_absolute_pinned_defender_and_vulnerable_piece_keep_separate_meanings(self):
        from tests.test_vulnerability import position
        board=position({'e1':'K','h8':'k','e2':'P','d3':'N','e8':'r','f4':'n'})
        window=self.window();window._live_monitoring=True;window._apply_live_position(board,'test')
        self.wait(lambda:chess.E2 in window.board_widget.pinned_squares and chess.D3 in window.board_widget.vulnerable_squares)

    def test_pin_worker_discards_old_work_and_caches_the_latest_layout(self):
        worker=PinWorker();ready=[];started=threading.Event();calls=[]
        def calc(board,**kwargs):
            calls.append(board.board_fen())
            if len(calls)==1:
                started.set()
                while not kwargs['cancelled']():time.sleep(.001)
                return PinResult((),False,0,0)
            return find_pins(board,**kwargs)
        worker.completed.connect(lambda token,key,result:ready.append((token,result)))
        try:
            with patch('ajedrez.ui.pin_worker.find_pins',side_effect=calc):
                worker.start();worker.request_position(absolute(),True,1);self.assertTrue(started.wait(2))
                for i in range(2,20):worker.request_position(relative(),True,i)
                self.wait(lambda:bool(ready) and ready[-1][0]==19)
                self.assertTrue(all(token!=1 for token,_ in ready))
                count=len(calls);worker.request_position(relative(),False,20)
                self.wait(lambda:ready[-1][0]==20);self.assertEqual(len(calls),count)
        finally:worker.stop();self.assertTrue(worker.wait(2000))

    def test_pins_x_and_five_real_stockfish_arrows_coexist(self):
        from tests.test_vulnerability import position
        board=position({'e1':'K','h8':'k','e2':'P','d3':'N','e8':'r','f4':'n'})
        window=MainWindow();self.addCleanup(window.close);window._live_monitoring=True
        window.time_spin.setValue(.15);window._apply_live_position(board,'test')
        self.wait(lambda:chess.E2 in window.board_widget.pinned_squares
                  and chess.D3 in window.board_widget.vulnerable_squares and len(window.board_widget.analysis_moves)==5)
        self.assertTrue(all(move in board.legal_moves for move in window.board_widget.analysis_moves))
        self.assertEqual(window.board_widget.board.fen(),board.fen())

    def test_pgn_navigation_never_keeps_live_pin_marks(self):
        import tempfile
        from pathlib import Path
        from ajedrez.pgn import load_pgn_games
        window=self.window();window._live_monitoring=True
        window._apply_live_position(absolute(),'test');self.wait(lambda:bool(window.board_widget.pinned_squares))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'game.pgn';path.write_text('1. e4 e5 *')
            game=load_pgn_games(path)[0];window._set_pgn_game(game);window._navigate_pgn(2)
        self.assertEqual(window.board_widget.pinned_squares,frozenset())
        self.assertFalse(window._live_monitoring)
        self.assertEqual(window.board_widget.board.fen(),game.board_at(2).fen())


if __name__=='__main__':unittest.main()
