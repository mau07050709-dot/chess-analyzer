"""Real video screenshot and controlled degradation; no downloaded models."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import unittest,json,time
from unittest.mock import patch
import chess
from PySide6.QtCore import Qt,QBuffer,QByteArray,QIODevice,QEventLoop,QTimer
from PySide6.QtGui import QImage,QPainter,QColor
from tests.qt_test_support import test_application
from tests.image_helpers import reverse_cells
from tests.test_last_move_highlight import painted_highlight
from ajedrez.recognition import recognize_position
from ajedrez.last_move_highlight import detect_highlights,detect_turn_highlights,resolve_last_move
from ajedrez.capture.board_region import locate_board,background_colors
from ajedrez.ui.main_window import MainWindow

FIX=Path(__file__).parent/'fixtures'
REAL='r6k/pp4p1/1n3n1p/2bqp3/6b1/3BBP2/PPP1Q1PP/R4RK1'
NORMAL='rnbqkb1r/pp1ppppp/2p2n2/8/2P5/2N5/PP1PPPPP/R1BQKBNR'

def real_crop():return QImage(str(FIX/'video_stream_reference.png')).copy(619,380,437,437)
def compressed(image,side,quality):
 image=image.scaled(side,side,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
 data=QByteArray();buffer=QBuffer(data);buffer.open(QIODevice.OpenModeFlag.WriteOnly)
 image.save(buffer,'JPG',quality);buffer.close();return QImage.fromData(data,'JPG')

class VideoRecognitionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=test_application()
 def wait(self,predicate):
  loop=QEventLoop();poll=QTimer();end=QTimer();end.setSingleShot(True);poll.setInterval(3)
  poll.timeout.connect(lambda:loop.quit() if predicate() else None);end.timeout.connect(loop.quit)
  poll.start();end.start(10000)
  if not predicate():loop.exec()
  poll.stop();end.stop();self.assertTrue(predicate(),'No terminó el flujo real ONNX/Stockfish')
 def feed(self,w,image):
  with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False):
   for _ in range(2):
    w._live_frame_is_stable=True
    w._start_recognition(image,'white','live_poll' if w._live_tracker else 'live_setup')
    self.wait(lambda:w.recognition_worker is None)
 def test_real_video_pieces_are_correct_before_any_photometric_retry(self):
  r=recognize_position(real_crop(),rescue_degraded=False)
  self.assertEqual(r.piece_placement,REAL);self.assertEqual(r.uncertain_squares,())
 def test_three_marks_need_geometry_color_and_original_piece_evidence(self):
  image=real_crop();r=recognize_position(image)
  self.assertIsNone(detect_highlights(image))
  pair=detect_turn_highlights(image,r.piece_placement,'white')
  evidence=resolve_last_move(chess.Board(REAL+' b - - 0 1'),pair,'white')
  self.assertEqual((evidence.origin,evidence.destination,evidence.next_turn),(chess.C8,chess.G4,chess.WHITE))
  flipped=reverse_cells(image)
  self.assertEqual(set(detect_turn_highlights(flipped,REAL,'black').squares('black')),{chess.C8,chess.G4})
 def test_ambiguous_or_occupied_extra_highlights_do_not_prove_a_turn(self):
  board=chess.Board(REAL+' w - - 0 1')
  for squares,color in (((chess.C8,chess.G4,chess.F2),(245,246,130)),
                         ((chess.C8,),None),((chess.C8,chess.G4,chess.F2,chess.F3),None)):
   image=painted_highlight(board,squares,color=color)
   self.assertIsNone(detect_turn_highlights(image,REAL,'white'))
 def test_normal_frames_keep_original_predictions_without_extra_inference(self):
  for name in ('chesscom_moved_position.jpg','chesscom_review_capture_506x507.png'):
   image=QImage(str(FIX/name));raw=recognize_position(image,rescue_degraded=False)
   with patch('ajedrez.recognition._rescue_degraded') as retry:
    current=recognize_position(image);retry.assert_not_called()
   self.assertEqual(raw.predictions,current.predictions);self.assertEqual(current.piece_placement,NORMAL)
 def test_small_boards_are_localized_in_full_hd_and_correctly_segmented(self):
  image=QImage(str(FIX/'chesscom_moved_position.jpg'))
  for side in (128,160,192,224):
   small=image.scaled(side,side,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
   canvas=QImage(1920,1080,QImage.Format.Format_RGB32);canvas.fill(QColor('#303030'))
   p=QPainter(canvas);p.drawImage(400,200,small);p.end()
   rect=locate_board(canvas,background_colors(small))
   self.assertIsNotNone(rect);self.assertLessEqual(abs(rect.x()-400),2);self.assertLessEqual(abs(rect.y()-200),2)
   r=recognize_position(canvas.copy(rect));self.assertEqual(r.piece_placement,NORMAL);self.assertFalse(r.uncertain_squares)
 def test_compression_retry_preserves_labels_and_uses_real_confidences(self):
  image=QImage(str(FIX/'chesscom_moved_position.jpg'))
  for side in (192,256):
   degraded=compressed(image,side,20)
   raw=recognize_position(degraded,rescue_degraded=False);fixed=recognize_position(degraded)
   self.assertTrue(raw.uncertain_squares);self.assertFalse(fixed.uncertain_squares)
   self.assertEqual(fixed.piece_placement,NORMAL);self.assertEqual(raw.piece_placement,fixed.piece_placement)
   self.assertEqual(raw.confidence_threshold,fixed.confidence_threshold)
   self.assertGreater(fixed.timings.inference_ms,0)
 def test_information_lost_by_severe_compression_is_not_silently_accepted(self):
  image=compressed(QImage(str(FIX/'chesscom_moved_position.jpg')),128,20)
  r=recognize_position(image);self.assertTrue(r.uncertain_squares)
 def test_alternate_real_piece_set_and_board_colors_remain_recognizable(self):
  # Independent real image supplied earlier; not the CNN's classic-piece fixture.
  expected=json.loads((FIX/'vulnerable_references.json').read_text(encoding='utf-8'))[0]
  r=recognize_position(QImage(str(FIX/expected['image'])))
  self.assertEqual(r.piece_placement,expected['placement']);self.assertFalse(r.uncertain_squares)
 def test_all_existing_alternate_art_references_with_visually_checked_empty_h2(self):
  entries=json.loads((FIX/'vulnerable_references.json').read_text(encoding='utf-8'))
  entries.append(json.loads((FIX/'pin_reference.json').read_text(encoding='utf-8')))
  for entry in entries:
   expected=chess.Board(entry['placement']+' w - - 0 1')
   # Those two old tactical diagrams' metadata includes an extra pawn. Direct
   # inspection of both images shows EMPTY h2. Do not alter tactical fixtures.
   if entry['image'] in ('vulnerable_reference_2.png','vulnerable_reference_3.png'):
    expected.remove_piece_at(chess.H2)
   r=recognize_position(QImage(str(FIX/entry['image'])))
   self.assertEqual(r.piece_placement,expected.board_fen(),entry['image'])
   self.assertFalse(r.uncertain_squares,entry['image'])
 def test_mild_blur_preserves_a_normal_board(self):
  image=QImage(str(FIX/'chesscom_moved_position.jpg'))
  blurred=image.scaled(320,320,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
  blurred=blurred.scaled(640,640,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
  r=recognize_position(blurred)
  self.assertEqual(r.piece_placement,NORMAL);self.assertFalse(r.uncertain_squares)
 def test_combined_video_theme_size_and_jpeg(self):
  for side,quality in ((256,45),(224,65)):
   image=compressed(real_crop(),side,quality)
   r=recognize_position(image);self.assertEqual(r.piece_placement,REAL);self.assertFalse(r.uncertain_squares)
 def test_real_video_previously_waiting_for_turn_now_gets_five_real_engine_lines(self):
  w=MainWindow();self.addCleanup(w.close);w._live_monitoring=True;w._live_setup_pending=True
  w.time_spin.setValue(.15)
  # Reproduce the old detector path, preserving all other real components.
  with patch('ajedrez.ui.main_window.detect_turn_highlights',side_effect=lambda image,*_:detect_highlights(image)):
   self.feed(w,real_crop())
   self.assertFalse(w._live_tracker.turn_known);self.assertIsNone(w.live_analysis_worker)
  self.feed(w,real_crop())
  self.assertTrue(w._live_tracker.turn_known);self.assertEqual(w._live_tracker.board.turn,chess.WHITE)
  self.wait(lambda:w._live_search_finalized and len(w.board_widget.analysis_moves)==5)
  self.assertEqual(w._live_tracker.board.board_fen(),REAL)
  self.assertTrue(all(m in w._live_tracker.board.legal_moves for m in w.board_widget.analysis_moves))
  self.assertEqual(w._live_tracker.board.castling_rights,0);self.assertIsNone(w._live_tracker.board.ep_square)
 def test_manual_analysis_turn_is_not_overridden_by_the_video_marks(self):
  w=MainWindow();self.addCleanup(w.close);w._live_monitoring=True;w._live_setup_pending=True
  w._analysis_turn=chess.BLACK;w.time_spin.setValue(.15)
  self.feed(w,real_crop())
  self.assertFalse(w._live_tracker.turn_known);self.assertEqual(w._analysis_turn,chess.BLACK)
  self.wait(lambda:w._live_search_finalized and bool(w.board_widget.analysis_moves))
  self.assertFalse(w._effective_analysis_board().turn)

if __name__=='__main__':unittest.main()
