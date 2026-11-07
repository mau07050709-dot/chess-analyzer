"""Reproduce intermittent live-state failures and stress the latest-position flow."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import threading
import queue
from types import SimpleNamespace
from unittest.mock import patch
import chess
from pathlib import Path
from PySide6.QtGui import QImage
from PySide6.QtCore import QEventLoop, QTimer, QPointF, QRect
from PySide6.QtGui import QColor
from ajedrez.engine import AnalysisLine, AnalysisResult
from ajedrez.live_board import LiveBoardTracker
from ajedrez.ui.main_window import MainWindow, LiveAnalysisWorker
from ajedrez.ui.board_widget import BoardWidget
from tests.qt_test_support import test_application
from tests.test_live_board import observation
from tests.image_helpers import board_from_real_tiles
from ajedrez.recognition import recognize_position, recognize_position_auto_orientation

RUY="e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8 Nf1 Bf8 Ng3 g6 a4 c5 d5 c4 Be3 Nc5 Qd2 Qc7 Bg5 Bg7 Bh6 Bh8".split()
OPERA="e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#".split()


def sequence_states(sans,mirror=False):
    board=chess.Board();moves=[]
    for san in sans:
        move=board.parse_san(san);moves.append(move);board.push(move)
    board=chess.Board().mirror() if mirror else chess.Board()
    states=[board.copy(stack=True)]
    for move in moves:
        if mirror:move=chess.Move(chess.square_mirror(move.from_square),chess.square_mirror(move.to_square),promotion=move.promotion)
        board.push(move);states.append(board.copy(stack=True))
    return states


def sample_lines(board, depth=8, count=5):
    return tuple(AnalysisLine(i,move.uci(),board.san(move),20-i,None,depth,.01,(board.san(move),))
                 for i,move in enumerate(list(board.legal_moves)[:count],1))


class LiveStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=test_application()

    def wait_for(self,predicate,timeout=5000):
        loop=QEventLoop();timer=QTimer();timer.setInterval(5)
        timer.timeout.connect(lambda: loop.quit() if predicate() else None)
        deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
        timer.start();deadline.start(timeout);loop.exec();timer.stop();deadline.stop()
        self.assertTrue(predicate(),"Timeout esperando el estado más reciente")

    def test_three_stable_rook_half_frames_then_castle_keep_confirmed_history(self):
        for turn in ("w","b"):
            for target in ("g","c"):
                with self.subTest(turn=turn,target=target):
                    before=chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 7 12")
                    rank="1" if turn=="w" else "8"
                    rook_from="h" if target=="g" else "a"
                    rook_to="f" if target=="g" else "d"
                    half=before.copy();half.push_uci(f"{rook_from}{rank}{rook_to}{rank}")
                    tracker=LiveBoardTracker(before)
                    for _ in range(3): tracker.observe(observation(half))
                    final=before.copy();move=chess.Move.from_uci(f"e{rank}{target}{rank}");final.push(move)
                    tracker.observe(observation(final));update=tracker.observe(observation(final))
                    self.assertEqual(update.board.fen(),final.fen())
                    self.assertTrue(tracker.turn_known)

    def test_single_missing_piece_never_reanchors_the_confirmed_state(self):
        board=chess.Board();tracker=LiveBoardTracker(board)
        partial=board.copy();partial.remove_piece_at(chess.H2)
        for _ in range(4): tracker.observe(observation(partial))
        self.assertEqual(tracker.board.fen(),board.fen())
        self.assertTrue(tracker.turn_known)
        self.assertEqual(tracker.observe(observation(board)).status,"unchanged")

    def test_new_depth_prefix_does_not_erase_a_complete_current_batch(self):
        window=MainWindow();board=chess.Board();source=object()
        window._live_monitoring=True;window.live_analysis_worker=source
        window._live_analysis_request_id=7
        window._bind_live_search(board.fen(),7,source)
        try:
            full=sample_lines(board,8)
            window._live_analysis_progress(board.fen(),full,7,source)
            self.assertEqual(len(window.board_widget.analysis_moves),5)
            window._live_analysis_progress(board.fen(),sample_lines(board,9,1),7,source)
            self.assertEqual(len(window.board_widget.analysis_moves),5)
            newer=sample_lines(board,9)
            window._live_analysis_completed(board.fen(),AnalysisResult(board.fen(),"test",newer),7,source)
            window._live_analysis_progress(board.fen(),sample_lines(board,8,1),7,source)
            self.assertEqual(window.board_widget.analysis_moves,tuple(chess.Move.from_uci(l.best_move_uci) for l in newer))
        finally:
            window.live_analysis_worker=None;window._live_monitoring=False;window.close()

    def test_arrow_is_a_center_to_center_line_without_curve(self):
        class Painter:
            def setPen(self,value):pass
            def setBrush(self,value):pass
            def drawLine(self,start,end):self.line=(start,end)
            def drawPath(self,path):self.curve=path
            def drawPolygon(self,polygon):pass
        for index in range(5):
            painter=Painter();start=QPointF(30,150);end=QPointF(90,30)
            BoardWidget._draw_analysis_arrow(painter,start,end,60,QColor("red"),index)
            self.assertFalse(hasattr(painter,"curve"))
            self.assertEqual(painter.line,(start,end))

    def test_cancelled_old_engine_error_does_not_strand_new_position(self):
        started=threading.Event();created=[]
        class Session:
            def __init__(self,*args,**kwargs):created.append(self)
            def analyze(self,fen,*,cancel_event,**kwargs):
                if fen==chess.STARTING_FEN:
                    started.set();cancel_event.wait(2)
                    raise RuntimeError("Fallo UCI simulado después de cancelar la posición vieja")
                return AnalysisResult(fen,"test",sample_lines(chess.Board(fen)))
            def close(self):pass
        with patch("ajedrez.ui.main_window.UCIEngineSession",Session):
            worker=LiveAnalysisWorker("test",5);received=[]
            worker.completed.connect(lambda fen,result,token:received.append(fen))
            worker.start();worker.request_position(chess.STARTING_FEN,.2,None)
            self.assertTrue(started.wait(2))
            latest=chess.Board();latest.push_san("e4")
            worker.request_position(latest.fen(),.2,None)
            try:
                self.wait_for(lambda:latest.fen() in received,1500)
            finally:
                worker.stop();self.assertTrue(worker.wait(3000))

    def test_recent_confirmed_history_recovers_after_a_wrong_legal_placement(self):
        before=chess.Board();tracker=LiveBoardTracker(before)
        wrong=before.copy()
        wrong.set_piece_at(chess.F1,chess.Piece(chess.KNIGHT,chess.WHITE))
        wrong.set_piece_at(chess.F8,chess.Piece(chess.KNIGHT,chess.BLACK))
        self.assertTrue(wrong.is_valid())
        tracker.observe(observation(wrong));tracker.observe(observation(wrong))
        final=before.copy();final.push_san("e4")
        tracker.observe(observation(final));update=tracker.observe(observation(final))
        self.assertEqual(update.board.fen(),final.fen())
        self.assertTrue(tracker.turn_known)
        self.assertIn("reciente",update.message)

    def test_castling_partial_after_a_missed_opponent_move_is_not_accepted(self):
        before=chess.Board("r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R b KQkq - 0 10")
        tracker=LiveBoardTracker(before)
        opponent=before.copy();opponent.push_san("a6")
        half=opponent.copy();half.push_uci("h1f1")
        for _ in range(4):tracker.observe(observation(half))
        self.assertEqual(tracker.board.fen(),before.fen())
        final=opponent.copy();final.push_uci("e1g1")
        tracker.observe(observation(final));update=tracker.observe(observation(final))
        self.assertEqual(update.board.fen(),final.fen())
        self.assertTrue(tracker.turn_known)

    def test_three_sequences_repeat_with_skipped_frames_and_doubt_in_both_views(self):
        for repetition in range(3):
            for sans,mirror in ((RUY,False),(OPERA,False),(OPERA,True)):
                states=sequence_states(sans,mirror)
                for orientation in ("white","black"):
                    tracker=LiveBoardTracker(states[0]);index=0;step=0
                    while index<len(states)-1:
                        jump=(1,1,2,1,3,1,1)[(step+repetition)%7]
                        index=min(index+jump,len(states)-1);step+=1
                        with self.subTest(repetition=repetition,orientation=orientation,mirror=mirror,ply=index,sequence=" ".join(sans[:index])):
                            old=tracker.board.fen()
                            tracker.observe(observation(states[index],.2))
                            self.assertEqual(tracker.board.fen(),old)
                            tracker.observe(observation(states[index]),visual_stable=False)
                            update=tracker.observe(observation(states[index]),visual_stable=True)
                            if update.status=="waiting":
                                self.assertIn("enroque",update.message)
                                self.assertEqual(tracker.board.fen(),old)
                                continue
                            self.assertIn(update.status,("updated","resynced"))
                            self.assertEqual(update.board.board_fen(),states[index].board_fen())
                            self.assertEqual(update.board.turn,states[index].turn)
                            self.assertEqual(update.board.castling_rights,states[index].castling_rights)
                            self.assertTrue(tracker.turn_known)
                    self.assertEqual(tracker.board.board_fen(),states[-1].board_fen())
                    self.assertEqual(tracker.board.turn,states[-1].turn)
                    self.assertEqual(tracker.board.castling_rights,states[-1].castling_rights)

    def test_current_engine_failure_retries_without_a_new_move(self):
        count=[]
        class Session:
            def __init__(self,*args,**kwargs):count.append(self)
            def new_game(self):pass
            def analyze(self,fen,**kwargs):
                if len(count)==1:raise RuntimeError("Fallo transitorio simulado del motor actual")
                return AnalysisResult(fen,"test",sample_lines(chess.Board(fen)))
            def close(self):pass
        with patch("ajedrez.ui.main_window.UCIEngineSession",Session),patch("ajedrez.ui.main_window.find_engine",return_value="test"):
            window=MainWindow()
            try:
                window._live_monitoring=True;window._live_tracker=LiveBoardTracker(chess.Board())
                window._apply_live_position(chess.Board(),"prueba de recuperación")
                self.wait_for(lambda:window._live_search_finalized and bool(window.board_widget.analysis_moves))
                self.assertEqual(len(count),2)
                self.assertEqual(window.fen_edit.text(),chess.STARTING_FEN)
            finally:window.close();self.app.processEvents()

    def test_repeated_cancellations_keep_only_the_latest_queued_search(self):
        states=sequence_states(RUY);latest=states[30].fen()
        class Session:
            def __init__(self,*args,**kwargs):pass
            def analyze(self,fen,*,cancel_event,**kwargs):
                if fen!=latest:cancel_event.wait(2)
                return AnalysisResult(fen,"test",sample_lines(chess.Board(fen)))
            def close(self):pass
        for repetition in range(5):
            with self.subTest(repetition=repetition),patch("ajedrez.ui.main_window.UCIEngineSession",Session):
                worker=LiveAnalysisWorker("test",5);received=[]
                worker.completed.connect(lambda fen,result,token:received.append((fen,token)))
                worker.start();last_id=0
                try:
                    for board in states[:31]:
                        request_id=worker.request_position(board.fen(),.2,None)
                        self.assertGreater(request_id,last_id);last_id=request_id
                        self.assertLessEqual(worker._requests.qsize(),1)
                    self.wait_for(lambda:any(fen==latest for fen,token in received))
                    self.assertEqual(received,[(latest,last_id)])
                finally:worker.stop();self.assertTrue(worker.wait(3000))

    def test_real_onnx_stockfish_ui_follow_40_plies_in_both_orientations(self):
        from ajedrez.engine import find_engine
        try:find_engine()
        except FileNotFoundError as exc:self.skipTest(str(exc))
        states=sequence_states(RUY)
        for orientation in ("white","black"):
            window=MainWindow();rendered=[]
            original=window._render_lines
            def render(lines):
                ticket=window._live_search_ticket
                self.assertIsNotNone(ticket)
                self.assertEqual(ticket.fen,window.board_widget.board.fen())
                original(lines)
                self.assertTrue(all(move in window.board_widget.board.legal_moves for move in window.board_widget.analysis_moves))
                rendered.append((ticket.position_id,ticket.fen))
            try:
                window._render_lines=render
                window._live_monitoring=True;window._live_generation=33
                window._live_orientation=orientation;window._live_frame_is_stable=True
                window._live_tracker=LiveBoardTracker(states[0]);window.time_spin.setValue(.1)
                window._apply_live_position(states[0],"comienzo de estrés")
                self.wait_for(lambda:bool(rendered))
                index=0;step=0
                while index<40:
                    target=min(40,index+(1,1,2,1,3)[step%5]);step+=1
                    with self.subTest(orientation=orientation,ply=target,sequence=" ".join(RUY[:target])):
                        if target%4==0:
                            oldfen=window.fen_edit.text();oldid=window._live_analysis_request_id
                            window._recognition_completed(observation(states[target],.2),("live_poll",33))
                            self.assertEqual(window.fen_edit.text(),oldfen)
                            self.assertEqual(window._live_analysis_request_id,oldid)
                        result=recognize_position(board_from_real_tiles(states[target],orientation),orientation)
                        self.assertEqual(result.piece_placement,states[target].board_fen())
                        self.assertEqual(result.uncertain_squares,())
                        window._recognition_completed(result,("live_poll",33))
                        window._recognition_completed(result,("live_poll",33))
                        self.assertEqual(window.board_widget.board.board_fen(),states[target].board_fen())
                        self.assertEqual(window.board_widget.board.turn,states[target].turn)
                        self.app.processEvents()
                    index=target
                self.wait_for(lambda:window._live_search_finalized and window._live_search_ticket.fen==window.fen_edit.text())
                self.assertTrue(window.board_widget.analysis_moves)
                self.assertEqual(window.board_widget.board.turn,states[-1].turn)
            finally:window.close();self.app.processEvents()

    def test_user_report_images_are_not_proof_of_a_different_turn(self):
        fixtures=Path(__file__).parent/"fixtures"
        cases=(("phase10_2_report_1.png",None,"h3","3r3k/3q2pp/1r1p4/p1p1p3/1p2P3/1P1Q2PP/P2R1PB1/6K1"),
               ("phase10_2_report_2.png",(31,19,569,568),"g2","r2q1rk1/pp3ppp/1bn1pn2/3p1b2/1P1P4/P1N1P1P1/4NPBP/R1BQ1RK1"))
        for name,region,square,placement in cases:
            image=QImage(str(fixtures/name))
            if region:image=image.copy(*region)
            result=recognize_position_auto_orientation(image)
            self.assertEqual(result.orientation,"white")
            self.assertIn(square,result.uncertain_squares)
            expected=chess.Board(f"{placement} w - - 0 1")
            detected=chess.Board(f"{result.piece_placement} w - - 0 1")
            mismatches={chess.square_name(sq) for sq in chess.SQUARES if expected.piece_at(sq)!=detected.piece_at(sq)}
            self.assertEqual(mismatches,{"h3"} if region is None else set())
            self.assertTrue(mismatches.issubset(set(result.uncertain_squares)))
            confirmed=chess.Board();tracker=LiveBoardTracker(confirmed)
            tracker.observe(result)
            self.assertEqual(tracker.board.fen(),confirmed.fen())

    def test_payload_fen_and_old_completion_cannot_change_current_results(self):
        window=MainWindow();source=object();board=chess.Board()
        window._live_monitoring=True;window.live_analysis_worker=source
        window._bind_live_search(board.fen(),7,source)
        try:
            result=AnalysisResult(board.fen(),"test",sample_lines(board))
            window._live_analysis_completed(board.fen(),result,7,source)
            saved=(window.eval_label.text(),window.board_widget.analysis_moves)
            other=board.copy();other.push_san("e4")
            wrong=AnalysisResult(other.fen(),"test",sample_lines(board))
            window._live_analysis_completed(board.fen(),wrong,7,source)
            window._live_analysis_completed(other.fen(),wrong,6,source)
            self.assertEqual((window.eval_label.text(),window.board_widget.analysis_moves),saved)
        finally:window.live_analysis_worker=None;window._live_monitoring=False;window.close()

    def test_returning_to_the_same_fen_does_not_accept_an_old_search(self):
        window=MainWindow();source=object();first=chess.Board();second=first.copy();second.push_san("e4")
        window._live_monitoring=True;window.live_analysis_worker=source
        try:
            for revision,board in enumerate((first,second,first),1):
                window._live_position_revision=revision
                window.board_widget.set_board(board);window.fen_edit.setText(board.fen())
                window._bind_live_search(board.fen(),revision,source)
            current=AnalysisResult(first.fen(),"test",sample_lines(first,11))
            window._live_analysis_completed(first.fen(),current,3,source)
            saved=(window.eval_label.text(),window.board_widget.analysis_moves)
            window._live_analysis_completed(first.fen(),AnalysisResult(first.fen(),"old",sample_lines(first,2)),1,source)
            window._live_analysis_progress(first.fen(),sample_lines(first,12,1),3,source)
            self.assertEqual((window.eval_label.text(),window.board_widget.analysis_moves),saved)
        finally:window.live_analysis_worker=None;window._live_monitoring=False;window.close()

    def test_out_of_order_multipv_is_buffered_without_mixing_batches(self):
        from ajedrez.engine import UCIEngineSession
        session=object.__new__(UCIEngineSession)
        session._closed=False;session.engine_path="unit-protocol";session.multipv=5
        session.write_lock=threading.Lock();session.process=SimpleNamespace(poll=lambda:None,stdin=None)
        session.messages=queue.Queue();session._send=lambda command:None
        moves=("e2e4","d2d4","g1f3","b1c3","c2c4")
        for index in (1,3,2,5,4):
            session.messages.put(f"info depth 9 multipv {index} score cp 20 pv {moves[index-1]}")
        session.messages.put("info depth 8 multipv 1 score cp -999 pv g1h3")
        session.messages.put("bestmove e2e4")
        snapshots=[]
        result=session.analyze(chess.STARTING_FEN,depth=9,progress_callback=snapshots.append)
        self.assertEqual(tuple(line.best_move_uci for line in result.lines),moves)
        for batch in snapshots:
            self.assertEqual(tuple(line.index for line in batch),tuple(range(1,len(batch)+1)))
            self.assertEqual({line.depth for line in batch},{9})

    def test_capture_never_reads_the_overlapping_analyzer_window(self):
        window=MainWindow();window.move(0,0);window.show();self.app.processEvents()
        window._live_monitoring=True;window._live_screen=object();window._live_region=QRect(20,20,100,100)
        try:
            with patch("ajedrez.ui.main_window.capture_screen") as capture:
                window._live_capture_tick()
            capture.assert_not_called()
            self.assertIn("superpone",window.live_status.text())
        finally:window._stop_live_reading();window.close();self.app.processEvents()

    def test_frame_finishing_during_a_drag_cannot_change_the_position(self):
        window=MainWindow();board=chess.Board();moved=board.copy();moved.push_san("e4")
        window._live_monitoring=True;window._live_generation=5;window._live_tracker=LiveBoardTracker(board)
        try:
            with patch("ajedrez.ui.main_window.left_mouse_button_down",return_value=True):
                window._recognition_completed(observation(moved),("live_poll",5))
            self.assertEqual(window.board_widget.board.fen(),board.fen())
            self.assertEqual(window._live_tracker.board.fen(),board.fen())
        finally:window._stop_live_reading();window.close()

    def test_real_rook_move_pending_castling_can_recover_three_further_plies(self):
        board=chess.Board("r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 10")
        tracker=LiveBoardTracker(board)
        board.push_san("Rf1")
        for _ in range(5):tracker.observe(observation(board))
        self.assertEqual(tracker.board.turn,chess.WHITE)
        for san in ("a6","a3","b6"):board.push_san(san)
        tracker.observe(observation(board));update=tracker.observe(observation(board))
        self.assertEqual(update.board.fen(),board.fen())
        self.assertTrue(tracker.turn_known)
