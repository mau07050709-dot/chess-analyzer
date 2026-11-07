"""Shared high-contrast application theme for all windows and dialogs."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor, QPalette, QFont


APP_STYLESHEET = """
QWidget { color: #F2F2F2; }
QTabWidget::pane { border: 1px solid #484B4D; background: #202124; }
QTabBar::tab { background: #2C2E30; color: #F2F2F2; padding: 7px 14px; border: 1px solid #484B4D; }
QTabBar::tab:selected { background: #326A4C; color: #FFFFFF; }
QTabBar::tab:hover { background: #424548; }
QMainWindow, QMenu, QMessageBox, QFileDialog { background: #202124; }
QWidget#root, QDialog { background: #202124; color: #F2F2F2; }
QLabel { color: #F2F2F2; background: transparent; }
QLabel#title { font-size: 25px; font-weight: 700; }
QLabel#subtitle, QLabel#muted { color: #AEB4B7; }
QLabel#sectionTitle { font-size: 14px; font-weight: 650; color: #D0D5D2; }
QLabel#status { color: #9DD7AE; font-weight: 650; }
QLabel#bestMove { font-size: 20px; font-weight: 700; color: #9DD7AE; }
QLabel#evaluationValue { font-size: 16px; font-weight: 700; color: #F2F2F2; }

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
QTableWidget, QProgressBar {
    color: #F2F2F2;
    background: #343638;
    selection-color: #FFFFFF;
    selection-background-color: #326A4C;
    border: 1px solid #62686C;
    border-radius: 5px;
    padding: 6px;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid #326A4C;
    padding: 5px;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QComboBox:disabled {
    color: #AEB4B7; background: #2C2E30; border-color: #484B4D;
}
QComboBox QAbstractItemView {
    color: #F2F2F2; background: #343638; selection-color: #FFFFFF;
    selection-background-color: #326A4C; border: 1px solid #62686C;
}
QTableWidget { gridline-color: #484B4D; alternate-background-color: #303234; }
QTableWidget::item:selected { color: #FFFFFF; background: #326A4C; }
QHeaderView::section { color: #F2F2F2; background: #424548; border: 1px solid #484B4D; padding: 5px; }
QProgressBar { color: #F2F2F2; background: #343638; border: 1px solid #62686C; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #326A4C; border-radius: 3px; }
QListWidget { color: #F2F2F2; background: #343638; border: 1px solid #62686C; }
QListWidget::item:selected { color: #FFFFFF; background: #326A4C; }
QCheckBox { color: #F2F2F2; spacing: 7px; }
QCheckBox:disabled { color: #AEB4B7; }

QPushButton {
    color: #F2F2F2;
    background: #424548;
    border: 1px solid #62686C;
    border-radius: 6px;
    padding: 8px 12px;
}
QPushButton:hover { color: #FFFFFF; background: #50565A; border-color: #89979A; }
QPushButton:pressed { color: #FFFFFF; background: #363A3D; border-color: #9DA8AB; }
QPushButton:focus { border: 2px solid #326A4C; }
QPushButton:disabled { color: #AEB4B7; background: #2C2E30; border-color: #484B4D; }
QPushButton#turnButton { padding: 6px 9px; }
QPushButton#turnButton:checked { color: #FFFFFF; background: #326A4C; border: 1px solid #9DD7AE; }
QPushButton#primaryButton {
    color: #FFFFFF; background: #326A4C; border: 1px solid #4D8B65; font-weight: 650;
}
QPushButton#primaryButton:hover { color: #FFFFFF; background: #3B7B58; }
QPushButton#primaryButton:pressed { color: #FFFFFF; background: #28543D; }
QPushButton#primaryButton:focus { border: 2px solid #A5D8B5; }
QPushButton#primaryButton:disabled { color: #D0D5D2; background: #2E3D34; border-color: #62686C; }
QFrame#divider { color: #484B4D; }
QToolTip { color: #F2F2F2; background: #343638; border: 1px solid #62686C; padding: 5px; }
QMenu { color: #F2F2F2; border: 1px solid #484B4D; }
QMenu::item:selected { background: #326A4C; }
QGroupBox { border: 1px solid #484B4D; border-radius: 6px; margin-top: 12px; padding-top: 10px; }
QScrollBar:vertical { background: #2C2E30; width: 14px; }
QScrollBar::handle:vertical { background: #62686C; min-height: 24px; border-radius: 5px; }
QScrollBar:horizontal { background: #2C2E30; height: 14px; }
QScrollBar::handle:horizontal { background: #62686C; min-width: 24px; border-radius: 5px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #89979A; background: #343638; border-radius: 3px; }
QCheckBox::indicator:checked { background: #73B589; border: 2px solid #CDEBD7; }
QCheckBox::indicator:disabled { background: #424548; border-color: #62686C; }
QPlainTextEdit { font-family: "Cascadia Mono", "Consolas", monospace; }
"""


def install_theme(app: QApplication | None = None) -> None:
    """Apply the shared stylesheet app-wide, including independently opened dialogs."""
    app = app or QApplication.instance()
    if app is not None and app.styleSheet() != APP_STYLESHEET:
        palette = QPalette()
        for role, color in ((QPalette.ColorRole.Window, '#202124'),
                            (QPalette.ColorRole.WindowText, '#F2F2F2'),
                            (QPalette.ColorRole.Base, '#343638'),
                            (QPalette.ColorRole.AlternateBase, '#2C2E30'),
                            (QPalette.ColorRole.Text, '#F2F2F2'),
                            (QPalette.ColorRole.Button, '#424548'),
                            (QPalette.ColorRole.ButtonText, '#FFFFFF'),
                            (QPalette.ColorRole.Highlight, '#326A4C'),
                            (QPalette.ColorRole.HighlightedText, '#FFFFFF'),
                            (QPalette.ColorRole.PlaceholderText, '#AEB4B7'),
                            (QPalette.ColorRole.ToolTipBase, '#343638'),
                            (QPalette.ColorRole.ToolTipText, '#F2F2F2')):
            palette.setColor(role, QColor(color))
        for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
            palette.setColor(QPalette.ColorGroup.Disabled, role, QColor('#AEB4B7'))
        app.setPalette(palette)
        app.setFont(QFont('Segoe UI', 10))
        app.setStyleSheet(APP_STYLESHEET)
