"""Windowed distribution entry point with a visible error on startup failure."""
import sys
import logging
from ajedrez.ui.app import main

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        from ajedrez.paths import user_data_dir
        from PySide6.QtWidgets import QApplication, QMessageBox
        details=traceback.format_exc()
        (user_data_dir()/'startup-error.log').write_text(details,encoding='utf-8')
        app=QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None,'No se pudo iniciar el analizador',
            'Ocurrió un error al iniciar. Consulta startup-error.log en:\n'+str(user_data_dir()))
        sys.exit(1)
