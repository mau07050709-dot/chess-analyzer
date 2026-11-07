"""Desktop layout restoration and real wood-video regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PySide6.QtGui import QImage
from ajedrez.ui.main_window import MainWindow
from ajedrez.recognition import recognize_position
from tests.qt_test_support import test_application

FIX=Path(__file__).parent/'fixtures'

class DesktopRestoreTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=test_application()
 def window(self):
  w=MainWindow();self.addCleanup(w.close);return w
 def test_default_layout_uses_old_board_space_without_artwork_reservation(self):
  w=self.window();w.show();self.app.processEvents()
  self.assertFalse(w._responsive_layout)
  self.assertFalse(hasattr(w,'header_scroll'))
  self.assertFalse(hasattr(w,'analysis_scroll'))
  self.assertFalse(w.board_panel.reserve_artwork)
  self.assertEqual(w.board_widget.minimumWidth(),400)
  for width,height in ((1100,740),(1500,900)):
   w.resize(width,height);self.app.processEvents()
   panel=w.board_panel
   self.assertEqual(w.board_widget.width(),min(panel.height(),panel.width()-44))
   self.assertEqual(w.board_widget.height(),w.eval_bar.height())
 def test_real_wood_frames_keep_all_labels_and_resolve_empty_cell_doubts(self):
  cases=(('video_wood_reference.png',(20,94,434,434),
          'r5k1/4qppp/3bp3/1Q1pN3/1P1Pn3/P4N2/2r2PPP/2B2RK1'),
         ('video_wood_second.png',(226,228,391,391),
          'r5k1/2q2ppp/3bp3/3pN3/1P1Pn3/P2Q1N2/2r2PPP/2B2RK1'))
  for name,crop,expected in cases:
   with self.subTest(name=name):
    image=QImage(str(FIX/name)).copy(*crop)
    raw=recognize_position(image,'black',rescue_degraded=False)
    result=recognize_position(image,'black')
    self.assertEqual(raw.piece_placement,expected)
    self.assertEqual(result.piece_placement,expected)
    self.assertEqual(result.uncertain_squares,())
    self.assertGreaterEqual(result.minimum_confidence,result.confidence_threshold)
 def test_semantic_stability_requires_consecutive_confident_same_scope_frames(self):
  w=self.window();w._live_monitoring=True
  result=SimpleNamespace(uncertain_squares=(),orientation='black',piece_placement='same')
  source=SimpleNamespace(last_move_highlight=None)
  seen=[]
  with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False),patch.object(
      w,'_recognition_completed',side_effect=lambda *a,**k:seen.append(w._live_frame_is_stable)):
   w._live_frame_is_stable=False
   context=('live_poll',w._live_generation)
   w._deliver_recognition(result,context,source)
   w._deliver_recognition(result,context,source)
   self.assertEqual(seen,[False,True])
   result.uncertain_squares=('d8',)
   w._deliver_recognition(result,context,source)
   result.uncertain_squares=()
   w._deliver_recognition(result,context,source)
   self.assertEqual(seen[-2:],[False,False])
   w._live_geometry_revision+=1
   w._deliver_recognition(result,context,source)
   self.assertFalse(seen[-1])
   source.premove_evidence=object()
   w._deliver_recognition(result,context,source)
   self.assertFalse(seen[-1])
   del source.premove_evidence
   with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=True):
    w._deliver_recognition(result,context,source)
    self.assertFalse(seen[-1])
  w._live_monitoring=False

if __name__=='__main__':unittest.main()
