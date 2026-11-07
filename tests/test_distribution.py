"""Dark palette, icon and resource resolution without changing chess logic."""
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PySide6.QtGui import QPalette,QIcon,QColor
from PySide6.QtWidgets import QPushButton,QLineEdit
from ajedrez.paths import resource_root,user_data_dir
from ajedrez.ui.theme import install_theme
from tests.qt_test_support import test_application


def contrast(a,b):
    def luminance(color):
        values=[v/255 for v in QColor(color).getRgb()[:3]]
        values=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in values]
        return sum(v*w for v,w in zip(values,(.2126,.7152,.0722)))
    low,high=sorted((luminance(a),luminance(b)))
    return (high+.05)/(low+.05)


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def test_dark_palette_and_readable_control_states(self):
        install_theme(self.app)
        self.assertEqual(self.app.palette().color(QPalette.ColorRole.Window).name(),'#202124')
        for foreground,background in (('#F2F2F2','#202124'),('#F2F2F2','#343638'),
                ('#FFFFFF','#424548'),('#FFFFFF','#326A4C'),('#AEB4B7','#2C2E30'),
                ('#AEB4B7','#343638'),('#FFFFFF','#50565A'),('#FFFFFF','#363A3D'),
                ('#D0D5D2','#2E3D34'),('#FFAAA0','#202124')):
            self.assertGreaterEqual(contrast(foreground,background),4.5,(foreground,background))
        for widget in (QPushButton('Prueba'),QLineEdit('123')):
            widget.ensurePolished()
            self.assertGreater(contrast(widget.palette().color(QPalette.ColorRole.Text),
                                        widget.palette().color(QPalette.ColorRole.Base)),4.5)
            widget.setEnabled(False);widget.ensurePolished()
            self.assertGreater(contrast(widget.palette().color(QPalette.ColorGroup.Disabled,QPalette.ColorRole.Text),
                                        widget.palette().color(QPalette.ColorGroup.Disabled,QPalette.ColorRole.Base)),4.5)
            widget.close()

    def test_icon_contains_windows_sizes_and_loads_in_qt(self):
        path=resource_root()/'assets'/'ajedrez.ico';data=path.read_bytes()
        reserved,kind,count=struct.unpack_from('<HHH',data)
        self.assertEqual((reserved,kind),(0,1))
        sizes={data[6+i*16] or 256 for i in range(count)}
        self.assertTrue({16,32,48,256}<=sizes)
        icon=QIcon(str(path));self.assertFalse(icon.isNull())
        self.assertFalse(icon.pixmap(32,32).isNull())

    def test_frozen_paths_do_not_depend_on_working_directory(self):
        with patch('sys.frozen',True,create=True),patch('sys._MEIPASS',r'C:\Bundle\_internal',create=True):
            self.assertEqual(resource_root(),Path(r'C:\Bundle\_internal'))

    def test_writable_data_is_separate_from_bundle(self):
        with tempfile.TemporaryDirectory(dir=resource_root()) as temporary,patch.dict('os.environ',{'LOCALAPPDATA':temporary}):
            target=user_data_dir()
            self.assertEqual(target,Path(temporary)/'ChessAnalyzerPublic')
            (target/'probe.txt').write_text('ok',encoding='utf-8')
            self.assertEqual((target/'probe.txt').read_text(encoding='utf-8'),'ok')
