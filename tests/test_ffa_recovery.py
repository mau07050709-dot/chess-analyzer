"""Regressions from the two reported screenshots; no browser automation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import unittest
from PySide6.QtCore import Qt,QRectF,QTimer,QEventLoop
from PySide6.QtGui import QImage,QPainter,QColor,QFont
from ajedrez.ffa_recognition import recognize_ffa
from ajedrez.ffa_visual_cues import clocks,ClockTurn
from ajedrez.ffa_live import FFATracker
from ajedrez import ffa_engine as engine
from ajedrez.four_player import PLAYERS,initial_pieces
from ajedrez.ui.four_player_tab import FourPlayerTab
from tests.qt_test_support import test_application
from tests.test_ffa_live import real_tile_board,observed

FIX=Path(__file__).parent/'fixtures'
MOVES=('h2h4','b7d7','h13h11','m8l8')

def reported_board():return QImage(str(FIX/'ffa_failure_four_moves.png')).copy(94,29,868,868)

def expected_board():
    board=engine.new_position()
    for move in MOVES:board=engine.play_move(board,move)
    return board

def clock_tick(image,player,text):
    """Controlled digit changes within clock boxes measured in the REAL image."""
    result=image.copy();sample=next(s for s in clocks(image) if s.player==player)
    x,y,w,h=sample.box;scale=image.width()/280
    rect=QRectF(x*scale,y*scale,w*scale,h*scale)
    colors={'red':'#a83140','blue':'#326990','yellow':'#957611','green':'#487b55'}
    painter=QPainter(result);painter.fillRect(rect,QColor(colors[player]))
    # Deterministic seven-segment pixels: Qt's offscreen backend on this host
    # renders system-font digits as identical missing-font boxes.
    segments={'0':'abcedf','1':'bc','2':'abged','3':'abgcd','4':'fgbc',
              '5':'afgcd','6':'afgecd','7':'abc','8':'abcdefg','9':'abfgcd'}
    height=rect.height()*.72;width=height*.5;thick=max(3,rect.height()*.09)
    x=rect.center().x()-len(text)*(width+6)/2;y=rect.center().y()-height/2
    for char in text:
        if char==':':
            for yy in (y+height*.3,y+height*.65):painter.fillRect(QRectF(x+width/2,yy,thick,thick),QColor('white'))
        else:
            bars={'a':(0,0,width,thick),'g':(0,height/2,width,thick),'d':(0,height-thick,width,thick),
                  'f':(0,0,thick,height/2),'b':(width-thick,0,thick,height/2),
                  'e':(0,height/2,thick,height/2),'c':(width-thick,height/2,thick,height/2)}
            for key in segments[char]:
                dx,dy,ww,hh=bars[key];painter.fillRect(QRectF(x+dx,y+dy,ww,hh),QColor('white'))
        x+=width+6
    painter.end()
    return result


class FFARecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();end=QTimer();end.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None);end.timeout.connect(loop.quit)
        poll.start(2);end.start(10000)
        if not predicate():loop.exec()
        poll.stop();end.stop();self.assertTrue(predicate())

    def test_reported_complete_image_has_yellow_below_and_64_correct_pieces(self):
        result=recognize_ffa(reported_board(),'red')
        self.assertEqual(result.bottom,'yellow')
        self.assertEqual(result.pieces,engine.pieces_of(expected_board()))
        self.assertFalse(result.uncertain);self.assertFalse(result.transient)
        self.assertEqual({m.player for m in result.last_moves},set(PLAYERS))
        self.assertIsNone(result.turn_hint) # Four retained pairs do NOT identify the latest one.
        self.assertEqual({s.player for s in result.clock_samples},set(PLAYERS))

    def test_reported_truncated_image_keeps_real_missing_pixel_doubts(self):
        # The supplied screenshot ends 15 px before the bottom of the board.
        image=QImage(str(FIX/'ffa_failure_first_move.png')).copy(114,68,868,868)
        result=recognize_ffa(image)
        self.assertEqual(result.bottom,'yellow');self.assertFalse(result.transient)
        self.assertEqual(result.turn_hint,'blue')
        self.assertTrue(result.uncertain)
        self.assertEqual(FFATracker().observe(result)[0],'retry')

    def test_midgame_home_armies_resolve_each_orientation_without_turn_inference(self):
        pieces=engine.pieces_of(expected_board())
        for bottom in PLAYERS:
            result=recognize_ffa(real_tile_board(pieces,bottom))
            self.assertEqual(result.bottom,bottom);self.assertEqual(result.pieces,pieces)
            self.assertIsNone(result.turn_hint)

    def test_three_missed_moves_recover_actual_reported_position_and_metadata(self):
        tracker=FFATracker(engine.play_move(engine.new_position(),MOVES[0]))
        result=recognize_ffa(reported_board())
        tracker.observe(result);status,position=tracker.observe(result)
        self.assertEqual(status,'updated');self.assertEqual(position,expected_board())
        self.assertTrue(tracker.turn_known);self.assertTrue(tracker.history_known)

    def test_running_clock_requires_two_digit_changes_not_a_single_increment(self):
        for player in PLAYERS:
            detector=ClockTurn()
            views=[recognize_ffa(clock_tick(reported_board(),player,text)) for text in ('1:09','1:08','1:07')]
            self.assertIsNone(detector.observe(views[0],0))
            self.assertIsNone(detector.observe(views[0],.4))
            self.assertIsNone(detector.observe(views[1],1))
            self.assertEqual(detector.observe(views[2],2),player)
            detector.observe(replace(views[2],pieces=initial_pieces()),3)
            self.assertIsNone(detector.observe(views[2],4))

    def test_no_clock_or_ambiguous_multiple_marks_never_assign_default_red(self):
        result=recognize_ffa(reported_board());tracker=FFATracker()
        tracker.observe(result);status,position=tracker.observe(result)
        self.assertEqual(status,'unlinked');self.assertFalse(tracker.turn_known)
        self.assertFalse(any(position.ck));self.assertFalse(any(position.cq))
        detector=ClockTurn()
        self.assertIsNone(detector.observe(result,0));self.assertIsNone(detector.observe(result,1))
        for player in PLAYERS:
            t=FFATracker();r=replace(result,turn_hint=player,turn_source='clock')
            t.observe(r);_,position=t.observe(r)
            self.assertTrue(t.turn_known);self.assertEqual(position.turn,PLAYERS.index(player))
            self.assertFalse(any(position.ck))

    def test_initial_layout_does_not_prove_starting_colour(self):
        for player in PLAYERS:
            tracker=FFATracker();r=replace(observed(engine.new_position()),turn_hint=player,turn_source='clock')
            tracker.observe(r);_,position=tracker.observe(r)
            self.assertEqual(position.turn,PLAYERS.index(player))
        tracker=FFATracker();r=observed(engine.new_position());tracker.observe(r);tracker.observe(r)
        self.assertFalse(tracker.turn_known)

    def test_premove_of_wrong_player_does_not_bypass_confirmed_turn(self):
        initial=engine.new_position();initial.turn=1;initial.recompute_key()
        projected=engine.play_move(engine.new_position(),'h2h4')
        tracker=FFATracker(initial)
        observation=replace(observed(projected),turn_hint='blue',turn_source='last_move')
        tracker.observe(observation);status,_=tracker.observe(observation)
        self.assertEqual(status,'retry');self.assertEqual(tracker.position,initial)

    def test_two_clocks_changing_together_are_ambiguous(self):
        detector=ClockTurn();base=recognize_ffa(reported_board())
        detector.observe(base,0)
        both=clock_tick(clock_tick(reported_board(),'red','1:08'),'blue','1:08')
        self.assertIsNone(detector.observe(recognize_ffa(both),1))
        self.assertEqual(detector.hits,0)

    def test_actual_gui_clock_bootstrap_analysis_arrows_and_stale_guard(self):
        tab=FourPlayerTab();self.addCleanup(tab.close);self.addCleanup(tab.shutdown)
        tab.reading=True;tab.tracker=FFATracker()
        for stamp,text in ((0,'1:09'),(.4,'1:09'),(1,'1:08'),(2,'1:07')):
            observation=recognize_ffa(clock_tick(reported_board(),'blue',text))
            with patch('ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False),patch(
                    'ajedrez.ffa_visual_cues.time.monotonic',return_value=stamp):
                tab.recognition_completed(observation,tab.reading_generation,str(stamp).encode())
        self.wait(lambda:tab.worker is None)
        self.assertEqual(tab.position.turn,1);self.assertEqual(tab.board.orientation if hasattr(tab.board,'orientation') else tab.board.bottom,'yellow')
        self.assertEqual(tab.board.recommendation.player,'blue')
        self.assertFalse(any(tab.position.ck))
        token=tab.reading_generation;tab.stop_reading()
        tab.recognition_completed(observation,token,b'stale')
        self.assertIsNone(tab.board.recommendation)

if __name__=='__main__':unittest.main()
