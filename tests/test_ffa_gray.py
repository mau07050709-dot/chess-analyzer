"""User's real gray-army captures, annotated square by square in visual order.

Annotations are independent of inference. d means an inert gray piece whose
original army cannot be recovered from a standalone image.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QEventLoop,QTimer,Qt
from PySide6.QtGui import QImage,QPainter,QColor
from ajedrez import ffa_engine as engine
from ajedrez.ffa_live import FFATracker
from ajedrez.ffa_recognition import recognize_ffa,canonical_cell,_rescue_shapes
from ajedrez.ffa_visual_cues import position_from_pieces
from ajedrez.four_player import FourPiece,PLAYERS,playable,display_cell
from ajedrez.ui.four_player_tab import FourPlayerTab
from PySide6.QtCore import QRect
from tests.qt_test_support import test_application
from tests.test_ffa_live import observed

FIX=Path(__file__).parent/'fixtures'
ROWS={
'first': '''
-- -- -- dR dN -- gK -- dB -- dR -- -- --
-- -- -- dP dP dP -- dP dP dP dP -- -- --
-- -- -- -- -- -- -- -- rQ -- -- -- -- --
yR yP -- -- dQ -- dP -- -- -- -- -- rP rR
-- yP -- -- yN -- -- dN -- -- -- -- rP --
yB -- yP -- -- -- -- -- -- -- rN -- rP --
yQ yP -- -- -- -- -- -- -- -- rP -- -- rK
yK -- -- yP -- -- -- -- -- -- -- -- rP --
yB yP -- -- -- -- bP -- -- -- -- rN rP rB
-- -- -- yP -- -- -- -- -- -- -- -- rP --
yR yP -- -- -- -- -- -- -- -- -- -- rP rR
-- -- -- -- bP bP -- -- bN -- -- -- -- --
-- -- -- -- -- -- bN bP bP bP bP -- -- --
-- -- -- bQ -- -- -- bK bB -- bR -- -- --''',
'two': '''
-- -- -- dR dN gK -- -- dB -- dR -- -- --
-- -- -- dP dP dP -- dP dP dP dP -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- --
dR dP -- -- dQ -- dP -- -- -- -- -- rP rR
-- dP -- -- dN -- -- dN -- -- -- -- rP --
dB -- dP -- -- -- -- -- -- -- rN -- rP --
dQ dP -- -- -- -- -- -- -- -- rP -- -- rK
-- yK -- rQ -- -- -- -- -- -- -- -- rP --
dB dP -- -- -- -- bP -- -- -- -- rN rP rB
-- -- -- dP -- -- -- -- -- -- -- -- rP --
dR dP -- -- -- -- -- -- -- -- -- -- rP rR
-- -- -- -- bP bP -- -- bN -- -- -- -- --
-- -- -- -- -- -- bN bP bP bP bP -- -- --
-- -- -- bQ -- -- -- bK bB -- bR -- -- --''',
'three': '''
-- -- -- dR -- dQ -- -- -- -- dR -- -- --
-- -- -- dP dK -- -- dP dP -- dP -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- --
dR dP -- -- dQ -- dP -- -- -- -- -- dP --
-- dP -- -- dN -- -- dN -- -- bQ -- dP rK
dB -- dP -- -- -- -- -- -- -- -- -- dP --
-- -- -- -- -- -- -- -- -- -- -- -- -- dR
-- -- -- -- -- -- -- -- -- -- dP -- dB --
dB -- -- -- -- -- -- -- -- -- -- dP dP --
-- -- yK dP -- -- -- -- -- -- -- -- -- --
dR dP -- -- -- -- bQ -- -- -- -- -- dP dR
-- -- -- -- bP bP -- -- bN bP -- -- -- --
-- -- -- -- -- -- bN bP bP -- bP -- -- --
-- -- -- -- -- -- -- bK -- -- bR -- -- --'''}

def image(name):
    return QImage(str(FIX/f'ffa_gray_{name}.png')).copy({'first':34,'two':47,'three':58}[name],5,868,868)

def expected(name):
    colors={'d':'dead','r':'red','b':'blue','y':'yellow','g':'green'}
    result={}
    for r,line in enumerate(ROWS[name].strip().splitlines()):
        for c,code in enumerate(line.split()):
            if code!='--':result[canonical_cell(c,r,'blue')]=FourPiece(colors[code[0]],code[1])
    return result

class FFAGrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();timer=QTimer();timeout=QTimer();timeout.setSingleShot(True)
        timer.timeout.connect(lambda:loop.quit() if predicate() else None);timeout.timeout.connect(loop.quit)
        timer.start(2);timeout.start(10000)
        if not predicate():loop.exec()
        timer.stop();timeout.stop();self.assertTrue(predicate())

    def test_three_real_positions_every_piece_and_empty_square(self):
        for name,count in [('first',58),('two',57),('three',45)]:
            for size in (448,620,868):
                with self.subTest(name=name,size=size):
                    result=recognize_ffa(image(name).scaled(size,size,Qt.IgnoreAspectRatio,Qt.SmoothTransformation))
                    self.assertEqual(result.bottom,'blue');self.assertEqual(len(result.pieces),count)
                    self.assertEqual(result.pieces,expected(name))
                    self.assertFalse(result.uncertain);self.assertFalse(result.transient)

    def test_real_gray_capture_rotates_to_all_four_views(self):
        source=image('three')
        for bottom in PLAYERS:
            # Platforms rotate square locations, not the piece artwork.
            view=QImage(868,868,QImage.Format.Format_RGB32);view.fill(QColor('#302e2b'))
            painter=QPainter(view)
            for r in range(14):
                for c in range(14):
                    if playable(c,r):
                        destination=display_cell(*canonical_cell(c,r,'blue'),bottom)
                        painter.drawImage(destination[0]*62,destination[1]*62,source.copy(c*62,r*62,62,62))
            painter.end()
            result=recognize_ffa(view)
            self.assertEqual(result.pieces,expected('three'))
            self.assertFalse(result.uncertain)
            self.assertEqual(result.bottom,bottom)

    def test_doubtful_king_needs_two_independent_confident_views(self):
        class Session:
            def get_inputs(self):return [type('Input',(),{'name':'tiles'})()]
            def run(self,*_):return [self.probabilities]
        session=Session();scores=np.array([[.2,.67,.05,.03,.02,.02,.01]])
        pixels=np.zeros((1,32,32,3),np.uint8)
        for values in ((.9,.8),(.9,.6)):
            session.probabilities=np.zeros((2,13),np.float32)
            session.probabilities[:,1]=values
            result=_rescue_shapes(session,pixels,scores)
            self.assertEqual(result[0,1]>=.70,min(values)>=.70)
        session.probabilities[1]=0;session.probabilities[1,2]=.99
        self.assertTrue(np.array_equal(_rescue_shapes(session,pixels,scores),scores))

    def test_gray_pieces_are_obstacles_not_discarded_or_active_armies(self):
        position=position_from_pieces(expected('three'))
        self.assertEqual(engine.pieces_of(position),expected('three'))
        self.assertEqual(position.alive,[True,True,True,False])
        restored=engine.parse_position(position.to_fen4())
        self.assertEqual(engine.pieces_of(restored),expected('three'))
        b,gen,*_=engine.modules()
        for seat in (0,2):
            position.turn=seat;position.recompute_key()
            for move in gen.gen_legal(position):
                self.assertEqual(b.PC_TYPE[position.sq[b.mv_from(move)]],b.KING)

    def test_color_only_retirement_preserves_other_history_and_points(self):
        position=engine.new_position();position.points=[8,11,0,3]
        position.ep=[None,(40,41),None,None]
        tracker=FFATracker(position);pieces=engine.pieces_of(position)
        for cell,p in list(pieces.items()):
            if p.player=='green' and p.kind!='K':pieces[cell]=FourPiece('dead',p.kind)
        observation=replace(observed(position),pieces=pieces)
        tracker.observe(observation);status,new=tracker.observe(observation)
        self.assertEqual(status,'updated');self.assertTrue(tracker.turn_known)
        self.assertEqual(new.points,position.points);self.assertEqual(new.ep,position.ep)
        self.assertEqual(new.turn,position.turn);self.assertEqual(new.alive,[True]*4)
        self.assertEqual(new.ck[:3],position.ck[:3]);self.assertEqual(new.cq[:3],position.cq[:3])
        self.assertFalse(new.ck[3]);self.assertFalse(new.cq[3])

    def test_gray_king_removes_only_that_seat_without_point_award(self):
        position=engine.new_position();position.turn=3;position.recompute_key()
        pieces=engine.pieces_of(position)
        for cell,p in list(pieces.items()):
            if p.player=='green':pieces[cell]=FourPiece('dead',p.kind)
        tracker=FFATracker(position);o=replace(observed(position),pieces=pieces)
        tracker.observe(o);status,new=tracker.observe(o)
        self.assertEqual(status,'updated');self.assertFalse(new.alive[3])
        self.assertEqual(new.turn,0);self.assertEqual(new.points,[0]*4)

    def test_multiple_gray_armies_and_missed_moves_do_not_freeze_tracking(self):
        tracker=FFATracker()
        for name in ('first','two','three'):
            observation=recognize_ffa(image(name));position=None
            for _ in range(5):
                _,candidate=tracker.observe(observation)
                if candidate is not None:position=candidate;break
            self.assertIsNotNone(position,name)
            self.assertEqual(engine.pieces_of(position),expected(name))
            self.assertFalse(any(position.ck));self.assertFalse(any(position.cq))

    def test_legal_move_after_retirement_preserves_known_metadata(self):
        position=position_from_pieces(expected('first'));position.turn=1;position.points=[1,8,2,0]
        position.recompute_key();tracker=FFATracker(position)
        result=engine.analyze(position,100);target=engine.play_move(position,result.move)
        observation=observed(target);tracker.observe(observation);status,new=tracker.observe(observation)
        self.assertEqual(status,'updated');self.assertEqual(new,target)
        self.assertTrue(tracker.turn_known);self.assertTrue(tracker.history_known)

    def test_capture_of_gray_king_is_not_a_missing_live_king(self):
        pieces={(7,13):FourPiece('red','K'),(0,7):FourPiece('blue','K'),
                (6,0):FourPiece('yellow','K'),(12,6):FourPiece('dead','K'),
                (12,8):FourPiece('blue','R')}
        position=position_from_pieces(pieces);position.turn=1;position.recompute_key()
        target=engine.play_move(position,'m6m8');tracker=FFATracker(position)
        observation=observed(target);tracker.observe(observation);status,new=tracker.observe(observation)
        self.assertEqual(status,'updated');self.assertEqual(new,target)
        self.assertEqual(new.points,[0]*4)

    def test_live_region_defaults_to_my_color_and_keeps_explicit_blue(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        tab._saved_window_state=tab.windowState()
        with patch('ajedrez.ui.four_player_tab.WindowAnchor.bind',return_value=None),patch.object(tab,'start_recognition'):
            tab.region_selected(image('first'),QRect(0,0,868,868));tab.capture_timer.stop()
            self.assertEqual(tab.player.currentData(),'mine')
            tab.player.setCurrentIndex(tab.player.findData('blue'))
            tab.region_selected(image('first'),QRect(0,0,868,868));tab.capture_timer.stop()
            self.assertEqual(tab.player.currentData(),'blue')

    def test_drag_and_doubt_do_not_apply_gray_state_or_launch_analysis(self):
        initial=engine.new_position();tracker=FFATracker(initial)
        observation=recognize_ffa(image('three'))
        for value in (replace(observation,uncertain=((0,3),)),replace(observation,transient=True)):
            for _ in range(3):self.assertEqual(tracker.observe(value)[0],'retry')
            self.assertEqual(tracker.position,initial)
        for _ in range(3):self.assertEqual(tracker.observe(observation,dragging=True)[0],'retry')
        self.assertEqual(tracker.position,initial)

    def test_real_engine_blue_with_gray_obstacles_returns_legal_blue_move(self):
        b,gen,*_=engine.modules()
        for name in ('first','two','three'):
            position=position_from_pieces(expected(name));position.turn=1;position.recompute_key()
            result=engine.analyze(position,100)
            self.assertEqual(result.player,'blue')
            self.assertEqual(expected(name)[result.origin].player,'blue')
            self.assertIn(result.move,{b.move_str(m) for m in gen.gen_legal(position)})

    def test_live_selected_blue_analyzes_gray_frames_without_waiting_for_turn(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        tab.player.setCurrentIndex(tab.player.findData('blue'));tab.milliseconds.setValue(100)
        tab.reading=True;tab.tracker=FFATracker()
        with patch('ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False):
            for name in ('first','two','three'):
                o=recognize_ffa(image(name));token=tab.reading_generation
                for n in range(5):
                    tab.recognition_completed(o,token,f'{name}{n}'.encode())
                    if tab.board.pieces==expected(name):break
                self.wait(lambda:tab.worker is None and tab.board.recommendation is not None)
                self.assertEqual(tab.board.pieces,expected(name))
                self.assertEqual(tab.board.recommendation.player,'blue')
                self.assertEqual(tab.player.currentData(),'blue')
                self.assertFalse(any(tab.position.ck))
                self.assertNotIn('esperando el reloj',tab.read_status.text())

    def test_selected_color_does_not_change_turn_or_history_and_discards_old_result(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        tab.milliseconds.setValue(100)
        original=tab.position.to_fen4();tab.perspective.setCurrentIndex(1)
        for player in PLAYERS:
            old_token=tab.generation
            tab.player.setCurrentIndex(tab.player.findData(player));tab.start_analysis()
            self.wait(lambda:tab.worker is None)
            current=tab.board.recommendation
            self.assertEqual(current.player,player);self.assertEqual(tab.position.to_fen4(),original)
            self.assertEqual(tab.board.bottom,player)
            tab.show_result(replace(current,player=PLAYERS[(PLAYERS.index(player)+1)%4]),old_token)
            self.assertEqual(tab.board.recommendation,current)

    def test_my_color_maps_each_bottom_without_changing_real_turn(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        tab.milliseconds.setValue(100);tab.player.setCurrentIndex(tab.player.findData('mine'))
        original=tab.position.to_fen4()
        for seat,player in enumerate(PLAYERS):
            tab.perspective.setCurrentIndex(seat);tab.start_analysis();self.wait(lambda:tab.worker is None)
            self.assertEqual(tab.board.recommendation.player,player)
            self.assertEqual(tab.position.to_fen4(),original)

    def test_eliminated_turn_hint_never_returns_none_as_updated_position(self):
        o=replace(recognize_ffa(image('three')),turn_hint='green',turn_source='clock')
        tracker=FFATracker();tracker.observe(o);status,position=tracker.observe(o)
        self.assertEqual(status,'unlinked');self.assertIsNotNone(position)
        self.assertFalse(tracker.turn_known)

if __name__=='__main__':unittest.main()
