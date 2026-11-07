"""Own QApplication for the test process and destroy widgets before Qt exits.

Unlike the desktop entry point, unittest does not run app.exec()/aboutToQuit.
Relying on Python module finalization leaves native widgets alive after Qt's
thread-local/platform resources, which can abort an otherwise successful run.
"""
import atexit
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import delete, isValid

_application = None
_registered = False


def test_application():
    global _application, _registered
    _application = QApplication.instance() or QApplication([])
    if not _registered:
        atexit.register(_shutdown)
        _registered = True
    return _application


def _shutdown():
    global _application
    app = _application
    if app is None or not isValid(app):
        return
    for widget in app.topLevelWidgets():
        widget.close()  # MainWindow stops and joins its analysis workers here.
        widget.deleteLater()
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    from ajedrez.ui.board_widget import piece_renderer
    piece_renderer.cache_clear()
    delete(app)
    _application = None
