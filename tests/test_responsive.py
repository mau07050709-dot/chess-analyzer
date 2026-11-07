"""Logical Qt geometry matrix; no changes to physical monitor settings."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import chess
from PySide6.QtCore import QRect,QPoint,Qt
from tests.qt_test_support import test_application
from ajedrez.ui.main_window import MainWindow

class ResponsiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.app=test_application()
 def test_resolution_and_dpi_matrix_keeps_square_board_and_scroll_access(self):
  w=MainWindow(responsive_layout=True);self.addCleanup(w.close);w.show()
  for width,height in ((1920,1080),(1600,900),(1366,768),(1280,720),(1024,768),(2560,1440)):
   for scale in (1,1.25,1.5,1.75):
    with self.subTest(resolution=(width,height),scale=scale):
     w.resize(int(width/scale)-16,int((height-64)/scale)-32);self.app.processEvents()
     b=w.board_widget;panel=w.board_panel
     self.assertEqual(b.width(),b.height());self.assertTrue(panel.rect().contains(b.geometry()))
     self.assertEqual(w.eval_bar.height(),b.height());self.assertEqual(w.eval_bar.y(),b.y())
     self.assertTrue(w.centralWidget().rect().contains(w.main_splitter.geometry()))
     for art in (panel.ino,panel.right_decorations):
      self.assertTrue(art.isVisible());self.assertTrue(panel.rect().contains(art.geometry()))
      self.assertFalse(art.geometry().intersects(b.geometry()))
     for control,scroll in ((w.capture_button,w.header_scroll),(w.live_review_button,w.header_scroll),
                            (w.threshold_button,w.header_scroll),(w.pgn_end_button,w.header_scroll),
                            (w.auto_turn_button,w.analysis_scroll),(w.analyze_button,w.analysis_scroll),
                            (w.variants_view,w.analysis_scroll)):
      point=control.mapTo(scroll.widget(),control.rect().center())
      scroll.ensureVisible(point.x(),point.y(),8,8);self.app.processEvents()
      center=control.mapTo(scroll.viewport(),control.rect().center())
      self.assertTrue(scroll.viewport().rect().contains(center),(type(control).__name__,getattr(control,"text",lambda:"variants")(),center,scroll.viewport().size()))
 def test_resize_preserves_logical_state_overlays_and_capture_region(self):
  w=MainWindow(responsive_layout=True);self.addCleanup(w.close);w.show()
  b=w.board_widget;b.set_best_moves(['e2e4','d2d4','g1f3','c2c4','g2g3'])
  b.set_premove_squares([chess.E2,chess.E4]);b.set_pinned_squares([chess.E2]);b.set_vulnerable_squares([chess.D2])
  w._live_region=QRect(45,70,640,640);before=b.board.fen()
  for view in ('white','black'):
   b.set_orientation(view)
   for size in ((1920,1000),(1366,690),(720,540),(1100,740)):
    w.resize(*size);self.app.processEvents();side=b.width();cell=side/8
    for square in chess.SQUARES:
     x,y=b._display_coords(square);center=b._center(square,0,0,cell)
     self.assertAlmostEqual(center.x(),(x+.5)*cell);self.assertAlmostEqual(center.y(),(y+.5)*cell)
     self.assertTrue(b.rect().contains(center.toPoint()))
    self.assertEqual(len(b.analysis_moves),5);self.assertEqual(b.board.fen(),before)
    self.assertEqual(set(b.premove_squares),{chess.E2,chess.E4})
    self.assertEqual(w._live_region,QRect(45,70,640,640))
 def test_small_screen_initial_fit_uses_available_logical_geometry(self):
  from unittest.mock import patch
  w=MainWindow(responsive_layout=True);self.addCleanup(w.close)
  class Screen:
   def availableGeometry(self):return QRect(-1093,0,1093,570)
  with patch.object(w,'screen',return_value=Screen()):w._fit_available_screen()
  self.assertLessEqual(w.width(),1093);self.assertLessEqual(w.height(),570)
  self.assertGreaterEqual(w.x(),-1093);self.assertLessEqual(w.x()+w.width(),0)

if __name__=='__main__':unittest.main()
