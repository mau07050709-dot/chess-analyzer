"""Start the PySide6 desktop application."""
from __future__ import annotations

import sys
import logging
import os
from logging.handlers import RotatingFileHandler

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import install_theme
from ..paths import resource_root, user_data_dir
from PySide6.QtGui import QIcon


def main() -> int:
    debug=os.environ.get("AJEDREZ_DEBUG","").lower() in ("1","true","yes")
    handlers=[logging.StreamHandler()] if sys.stderr is not None else []
    directory=user_data_dir()/"logs"
    try:
        directory.mkdir(exist_ok=True)
        handlers.append(RotatingFileHandler(directory/"sincronizacion.log",
                                            maxBytes=2_000_000,backupCount=2,encoding="utf-8"))
    except OSError as exc:
        if sys.stderr is not None:
            print(f"No se pudo abrir el registro: {exc}",file=sys.stderr)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s PID=%(process)d %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Analizador de ajedrez")
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ProyectoAjedrez.Analizador.Desktop")
    app.setWindowIcon(QIcon(str(resource_root()/"assets"/"ajedrez.ico")))
    app.setStyle("Fusion")
    install_theme(app)
    window = MainWindow()
    if window._responsive_layout:
        window._fit_available_screen()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
