"""Screen-crop logic and capture UI checks that do not require a real monitor."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from tests.qt_test_support import test_application
from PySide6.QtWidgets import QApplication

from ajedrez.capture.screen_capture import ScreenFrame, capture_screen, clamp_region, crop_region
from ajedrez.ui.capture_dialog import CapturePreviewDialog, RegionSelector
from ajedrez.ui.main_window import MainWindow


class ScreenCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_application()

    def make_frame(self, ratio=1.0):
        image = QImage(8, 8, QImage.Format.Format_ARGB32)
        image.fill(QColor("red"))
        for y in range(4, 8):
            for x in range(4, 8):
                image.setPixelColor(x, y, QColor("blue"))
        return ScreenFrame("Monitor virtual", QRect(100, 50, 4, 4), image, ratio)

    def test_crop_uses_screen_global_origin_and_returns_copy(self):
        frame = self.make_frame()
        image = crop_region(frame, QRect(101, 51, 2, 2), minimum_size=1)
        self.assertEqual((image.width(), image.height()), (2, 2))
        self.assertEqual(image.pixelColor(0, 0), QColor("red"))
        self.assertEqual(image.pixelColor(1, 1), QColor("red"))

    def test_crop_scales_logical_coordinates_for_high_dpi(self):
        frame = self.make_frame(ratio=2.0)
        image = crop_region(frame, QRect(102, 52, 2, 2), minimum_size=1)
        self.assertEqual((image.width(), image.height()), (4, 4))
        self.assertEqual(image.pixelColor(0, 0), QColor("blue"))

    def test_region_is_clipped_and_tiny_selection_is_rejected(self):
        clipped = clamp_region(QRect(99, 49, 3, 3), QRect(100, 50, 8, 8), minimum_size=1)
        self.assertEqual(clipped, QRect(100, 50, 2, 2))
        with self.assertRaisesRegex(ValueError, "al menos"):
            clamp_region(QRect(10, 10, 10, 10), QRect(0, 0, 100, 100))

    def test_windows_capture_uses_physical_pixels_and_keeps_qt_origin(self):
        screen = Mock()
        screen.geometry.return_value = QRect(-1200, 50, 800, 600)
        screen.devicePixelRatio.return_value = 1.5
        screen.name.return_value = "Secundario"
        image = QImage(1200, 900, QImage.Format.Format_RGB32)
        with patch("ajedrez.capture.screen_capture.sys.platform", "win32"), \
             patch("ajedrez.capture.screen_capture.QGuiApplication.platformName", return_value="windows"), \
             patch("ajedrez.capture.screen_capture.grab_desktop_pixels", return_value=image) as native:
            frame = capture_screen(screen)
        native.assert_called_once_with(QRect(-1200, 50, 1200, 900))
        screen.grabWindow.assert_not_called()
        self.assertEqual(frame.geometry, screen.geometry())
        self.assertEqual(frame.device_pixel_ratio, 1.5)
        self.assertEqual(frame.image.size(), image.size())

    def test_windows_capture_uses_qt_if_native_capture_fails(self):
        screen = Mock()
        screen.geometry.return_value = QRect(0, 0, 160, 120)
        screen.devicePixelRatio.return_value = 1.0
        screen.grabWindow.return_value = QPixmap.fromImage(QImage(160, 120, QImage.Format.Format_RGB32))
        with patch("ajedrez.capture.screen_capture.sys.platform", "win32"), \
             patch("ajedrez.capture.screen_capture.QGuiApplication.platformName", return_value="windows"), \
             patch("ajedrez.capture.screen_capture.grab_desktop_pixels", side_effect=OSError("GDI error")):
            frame = capture_screen(screen)
        screen.grabWindow.assert_called_once_with(0)
        self.assertFalse(frame.image.isNull())

    def test_screen_capture_wraps_a_local_qt_snapshot(self):
        class FakeScreen:
            def geometry(self):
                return QRect(10, 20, 8, 8)

            def grabWindow(self, _window):
                return QPixmap.fromImage(self.image)

            def name(self):
                return "Monitor de prueba"

            image = QImage(8, 8, QImage.Format.Format_ARGB32)

        frame = capture_screen(FakeScreen())
        self.assertEqual(frame.screen_name, "Monitor de prueba")
        self.assertEqual(frame.geometry, QRect(10, 20, 8, 8))
        self.assertFalse(frame.image.isNull())

    def test_escape_cancels_region_selector_without_an_image(self):
        frame = self.make_frame()
        selector = RegionSelector(frame)
        cancelled = []
        selector.cancelled.connect(lambda: cancelled.append(True))
        selector.show()
        QTest.keyClick(selector, Qt.Key.Key_Escape)
        self.assertEqual(cancelled, [True])

    def test_dragging_selector_emits_a_cropped_region_with_screen_coordinates(self):
        image = QImage(100, 100, QImage.Format.Format_ARGB32)
        image.fill(QColor("green"))
        frame = ScreenFrame("Monitor virtual", QRect(-20, 30, 100, 100), image, 1.0)
        selector = RegionSelector(frame)
        result = []
        selector.selected.connect(lambda crop, rect: result.append((crop, rect)))
        selector.show()
        QTest.mousePress(selector, Qt.MouseButton.LeftButton, pos=QPoint(10, 15))
        QTest.mouseMove(selector, QPoint(80, 90), 20)
        QTest.mouseRelease(selector, Qt.MouseButton.LeftButton, pos=QPoint(80, 90))
        self.assertEqual(len(result), 1)
        crop, region = result[0]
        self.assertEqual(region.topLeft(), QPoint(-10, 45))
        self.assertGreaterEqual(crop.width(), 70)
        self.assertGreaterEqual(crop.height(), 75)

    def test_preview_and_main_window_capture_controls_exist(self):
        preview = CapturePreviewDialog(self.make_frame().image, "Monitor virtual", QRect(0, 0, 64, 64))
        self.assertFalse(preview.image_label.pixmap().isNull())
        window = MainWindow()
        self.assertGreaterEqual(window.screen_combo.count(), 1)
        self.assertTrue(window.capture_button.isEnabled())
        preview.close()
        window.close()
        preview.deleteLater()
        window.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
