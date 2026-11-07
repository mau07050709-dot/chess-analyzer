"""Artwork transparency and reserved-layout-space checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from ajedrez.paths import resource_root
from ajedrez.ui.main_window import MainWindow
from tests.qt_test_support import test_application


class DecorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def test_cutouts_have_real_alpha_and_white_clothes_stay_opaque(self):
        for name in ('ino','orihime','winry'):
            with self.subTest(name=name):
                image=QImage(str(resource_root()/'assets'/'decorations'/f'{name}.png'))
                self.assertFalse(image.isNull());self.assertTrue(image.hasAlphaChannel())
                rgba=image.convertToFormat(QImage.Format.Format_RGBA8888)
                pixels=np.frombuffer(rgba.constBits(),dtype=np.uint8).reshape(rgba.height(),rgba.width(),4)
                self.assertGreater(float(np.mean(pixels[:,:,3]==0)),.1)
                self.assertGreater(float(np.mean(pixels[:,:,3]>240)),.1)
                if name=='orihime':
                    white=(pixels[:,:,:3].min(axis=2)>220)&(pixels[:,:,3]>240)
                    self.assertGreater(float(np.mean(white)),.03)

    def test_decorations_never_overlap_board_bar_or_analysis_panel(self):
        w=MainWindow();self.addCleanup(w.close);w.resize(1920,1080);w.show();self.app.processEvents()
        panel=w.board_panel;right=panel.right_decorations
        self.assertTrue(panel.ino.isVisible());self.assertTrue(right.isVisible())
        original_board=w.board_widget.board.fen();original_orientation=w.board_widget.orientation
        for decoration in (panel.ino,right):
            self.assertFalse(decoration.geometry().intersects(w.board_widget.geometry()))
            self.assertFalse(decoration.geometry().intersects(w.eval_bar.geometry()))
            self.assertTrue(panel.rect().contains(decoration.geometry()))
        self.assertLess(panel.ino.geometry().right(),w.eval_bar.x())
        self.assertGreater(right.x(),w.board_widget.geometry().right())
        self.assertLess(right.winry.geometry().bottom(),right.orihime.y())
        for art in (panel.ino,right.winry,right.orihime):
            self.assertTrue(art.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))
            self.assertFalse(art.scaled.isNull());self.assertLessEqual(art.scaled.width(),art.width())
            self.assertLessEqual(art.scaled.height(),art.height())
            cache=art.original.cacheKey();art.update();self.app.processEvents()
            self.assertEqual(art.original.cacheKey(),cache)
        self.assertEqual(w.board_widget.board.fen(),original_board)
        self.assertEqual(w.board_widget.orientation,original_orientation)

    def test_narrow_board_area_scales_art_without_overlapping_board_or_bar(self):
        from ajedrez.ui.board_widget import BoardWidget
        from ajedrez.ui.analysis_controls import BoardEvaluationPanel,EvaluationBar
        board=BoardWidget();bar=EvaluationBar();panel=BoardEvaluationPanel(board,bar)
        self.addCleanup(panel.close)
        panel.resize(500,450);panel.show();self.app.processEvents()
        self.assertTrue(panel.ino.isVisible());self.assertTrue(panel.right_decorations.isVisible())
        for art in (panel.ino,panel.right_decorations):
            self.assertFalse(art.geometry().intersects(board.geometry()))
            self.assertFalse(art.geometry().intersects(bar.geometry()))
        self.assertEqual(board.height(),board.width())
        self.assertEqual(bar.height(),board.height())
        self.assertEqual(bar.y(),board.y())
        self.assertEqual(board.x()-bar.geometry().right(),11)
