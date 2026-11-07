"""Real screenshot inference plus controlled transitions, no remote board use."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
from PySide6.QtCore import Qt,QRect,QEventLoop,QTimer
from PySide6.QtGui import QImage,QPainter,QColor
from ajedrez.ffa_recognition import recognize_ffa,FFAObservation
from ajedrez.ffa_live import FFATracker
from ajedrez import ffa_engine as engine
from ajedrez.four_player import PLAYERS,initial_pieces,playable,display_cell
from ajedrez.capture.board_region import rgb
from ajedrez.ui.four_player_tab import FourPlayerTab
from tests.qt_test_support import test_application


def reference():
    return QImage(str(Path(__file__).parent/'fixtures/ffa_reference.png')).copy(57,8,868,868)


def observed(position,bottom='red'):
    return FFAObservation(engine.pieces_of(position),(),bottom,0)


def real_tile_board(pieces,bottom='red',side=868):
    """Synthetic new position using foreground pixels of the REAL reference.

    This does not replace validation against a second real-game capture.
    """
    source=reference();templates={}
    for cell,piece in initial_pieces().items():
        if (piece.player,piece.kind) in templates:continue
        c,r=cell;tile=source.copy(c*62,r*62,62,62).convertToFormat(QImage.Format.Format_RGBA8888)
        values=rgb(tile).astype(float)
        bg=np.median(np.concatenate((values[:3].reshape(-1,3),values[-3:].reshape(-1,3))),axis=0)
        foreground=np.linalg.norm(values-bg,axis=2)>30
        data=np.ascontiguousarray(np.dstack((values.astype('uint8'),foreground.astype('uint8')*255)))
        templates[piece.player,piece.kind]=QImage(data.data,62,62,data.strides[0],QImage.Format.Format_RGBA8888).copy()
    image=QImage(868,868,QImage.Format.Format_RGB32);image.fill(QColor('#302e2b'))
    painter=QPainter(image)
    for row in range(14):
        for col in range(14):
            if not playable(col,row):continue
            c,r=display_cell(col,row,bottom)
            painter.fillRect(c*62,r*62,62,62,QColor('#aaa' if (c+r)%2==0 else '#ddd'))
            piece=pieces.get((col,row))
            if piece:painter.drawImage(c*62,r*62,templates[piece.player,piece.kind])
    painter.end()
    return image.scaled(side,side,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)


class FFALiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();end=QTimer();end.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None)
        end.timeout.connect(loop.quit);poll.start(2);end.start(10000)
        if not predicate():loop.exec()
        poll.stop();end.stop();self.assertTrue(predicate())

    def test_real_reference_all_160_cells_four_colours(self):
        result=recognize_ffa(reference())
        self.assertEqual(result.pieces,initial_pieces());self.assertFalse(result.uncertain)
        self.assertFalse(result.transient)

    def test_four_initial_orientations_and_sizes(self):
        for bottom in PLAYERS:
            for side in (448,868):
                with self.subTest(bottom=bottom,side=side):
                    result=recognize_ffa(real_tile_board(initial_pieces(),bottom,side),'red')
                    self.assertFalse(result.uncertain)
                    self.assertEqual(result.bottom,bottom)
                    self.assertEqual(result.pieces,initial_pieces())

    def test_onnx_session_is_reused_for_160_cell_batch(self):
        from ajedrez.recognition import _load_classifier
        recognize_ffa(reference());before=_load_classifier.cache_info()
        recognize_ffa(reference());after=_load_classifier.cache_info()
        self.assertEqual(before.misses,after.misses);self.assertGreater(after.hits,before.hits)

    def test_new_positions_from_real_piece_tiles(self):
        position=engine.new_position()
        for _ in range(4):
            result=engine.analyze(position,100)
            position=engine.play_move(position,result.move)
            observation=recognize_ffa(real_tile_board(engine.pieces_of(position)))
            self.assertFalse(observation.uncertain)
            self.assertEqual(observation.pieces,engine.pieces_of(position))

    def test_stability_drag_partial_board_and_known_metadata(self):
        position=engine.new_position();tracker=FFATracker(position)
        next_position=engine.play_move(position,'h2h3');r=observed(next_position)
        before=position.to_fen4()
        self.assertEqual(tracker.observe(r,dragging=True)[0],'retry')
        self.assertEqual(tracker.position.to_fen4(),before)
        self.assertEqual(tracker.observe(replace(r,uncertain=((7,12),)))[0],'retry')
        self.assertEqual(tracker.observe(r)[0],'waiting')
        status,new=tracker.observe(r)
        self.assertEqual(status,'updated');self.assertEqual(new,next_position)
        lifted=dict(r.pieces);lifted.pop((7,11))
        partial=replace(r,pieces=lifted)
        tracker.observe(partial);tracker.observe(partial)
        self.assertEqual(tracker.position,next_position)

    def test_two_missed_plies_are_recovered_with_turn_and_rights(self):
        initial=engine.new_position();position=engine.play_move(initial,'h2h3')
        b,gen,*_=engine.modules();position=engine.play_move(position,b.move_str(gen.gen_legal(position)[0]))
        tracker=FFATracker(initial);r=observed(position)
        tracker.observe(r);status,new=tracker.observe(r)
        self.assertEqual(status,'updated');self.assertEqual(new,position)
        self.assertTrue(tracker.turn_known)

    def test_unknown_midgame_does_not_invent_turn_or_castling(self):
        position=engine.play_move(engine.new_position(),'h2h3')
        tracker=FFATracker();r=observed(position)
        tracker.observe(r);status,new=tracker.observe(r)
        self.assertEqual(status,'unlinked');self.assertFalse(tracker.turn_known)
        self.assertFalse(any(new.ck));self.assertFalse(any(new.cq))
        self.assertEqual(new.ep,[None]*4)
        b,gen,*_=engine.modules();actual=engine.play_move(position,b.move_str(gen.gen_legal(position)[0]))
        r=observed(actual);tracker.observe(r);status,new=tracker.observe(r)
        self.assertEqual(status,'updated');self.assertTrue(tracker.turn_known)
        self.assertEqual(new.turn,actual.turn)

    def test_transient_red_pair_is_not_a_completed_move(self):
        image=reference();p=QPainter(image)
        for c,r in ((7,10),(7,9)):p.fillRect(c*62,r*62,62,62,QColor('#b45c49'))
        p.end();result=recognize_ffa(image)
        self.assertTrue(result.transient)
        tracker=FFATracker(engine.new_position())
        self.assertEqual(tracker.observe(result)[0],'retry')
        self.assertEqual(tracker.position,engine.new_position())

    def test_four_colors_castling_is_one_complete_transition(self):
        b,gen,*_=engine.modules()
        for seat in range(4):
            position=engine.new_position();position.turn=seat
            for sq in b.SQUARES:
                p=position.sq[sq]
                if p and not (b.PC_TYPE[p]==b.KING or (b.PC_COLOR[p]==seat and b.PC_TYPE[p]==b.ROOK)):
                    position.sq[sq]=0
            position.find_kings();position.recompute_key()
            castles=[m for m in gen.gen_legal(position) if b.mv_flag(m) in (b.F_CASTLE_SHORT,b.F_CASTLE_LONG)]
            self.assertEqual(len(castles),2)
            for move in castles:
                target=engine.play_move(position,b.move_str(move))
                tracker=FFATracker(position);result=observed(target)
                tracker.observe(result);status,new=tracker.observe(result)
                self.assertEqual(status,'updated');self.assertEqual(new,target)

    def test_bad_crop_is_rejected_without_a_candidate(self):
        for image in (QImage(),reference().scaled(200,200),reference().copy(0,0,500,800)):
            with self.assertRaises(ValueError):recognize_ffa(image)

    def test_live_gui_real_recognition_engine_pause_and_stale_discard(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        self.assertEqual(tab.read_button.text(),'Leer tablero en pantalla')
        tab.reading=True;tab.tracker=FFATracker(tab.position)
        with patch('ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False):
            for _ in range(2):
                tab.start_recognition(reference());self.wait(lambda:tab.recognition_worker is None)
            self.wait(lambda:tab.worker is None)
            self.assertIsNotNone(tab.board.recommendation)
            position=engine.play_move(tab.position,'h2h3')
            image=real_tile_board(engine.pieces_of(position))
            for _ in range(2):
                tab.start_recognition(image);self.wait(lambda:tab.recognition_worker is None)
            self.wait(lambda:tab.worker is None)
            self.assertEqual(tab.position,position)
            self.assertEqual(tab.board.recommendation.player,'blue')
            token=tab.reading_generation;tab.toggle_reading()
            self.assertTrue(tab.paused)
            tab.recognition_completed(observed(engine.new_position()),token,b'old')
            self.assertEqual(tab.position,position)
            tab.toggle_reading();self.assertFalse(tab.paused);tab.capture_timer.stop()
            tab.stop_reading();self.assertFalse(tab.reading)
            self.assertIsNone(tab.board.recommendation)

if __name__=='__main__':unittest.main()
