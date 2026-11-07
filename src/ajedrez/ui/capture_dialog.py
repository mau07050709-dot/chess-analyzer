"""Full-screen region selector and captured-image preview dialog."""
from __future__ import annotations

import chess
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal, QSize
from PySide6.QtGui import QColor, QImage, QIntValidator, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from ..capture.screen_capture import ScreenFrame, clamp_region, crop_region
from .board_widget import BoardWidget
from .theme import install_theme


class RegionSelector(QWidget):
    """Show a captured monitor as an overlay and let the user drag a crop rectangle."""

    selected = Signal(QImage, QRect)
    cancelled = Signal()
    selection_error = Signal(str)

    def __init__(self, frame: ScreenFrame) -> None:
        super().__init__(None)
        self.frame = frame
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._selection = QRect()
        self._error_message = ""
        self._finished = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setGeometry(frame.geometry)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle("Selecciona el tablero · Escape para cancelar")

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(self.rect(), self.frame.image)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 78))
        painter.setPen(QColor("white"))
        hint = "Arrastra para seleccionar el tablero · Escape para cancelar"
        painter.drawText(24, 38, self._error_message or hint)
        if self._selection.isValid() and not self._selection.isEmpty():
            painter.drawImage(self._selection, self.frame.image, self._image_rect(self._selection))
            painter.setPen(QPen(QColor("#38d89a"), 3, Qt.PenStyle.SolidLine))
            painter.drawRect(self._selection)
        elif self._drag_start is not None:
            painter.setPen(QPen(QColor("#38d89a"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self._drag_start, self._drag_current or self._drag_start).normalized())
        painter.end()

    def _image_rect(self, logical_rect: QRect) -> QRect:
        origin = self.frame.geometry.topLeft()
        local = logical_rect.translated(-origin)
        ratio = self.frame.device_pixel_ratio
        return QRectF(
            local.x() * ratio,
            local.y() * ratio,
            local.width() * ratio,
            local.height() * ratio,
        ).toAlignedRect().intersected(self.frame.image.rect())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            self._selection = QRect()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self._selection = QRect(self._drag_start, self._drag_current).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        self._selection = QRect(self._drag_start, event.position().toPoint()).normalized()
        self._drag_start = None
        self._drag_current = None
        global_region = self._selection.translated(self.frame.geometry.topLeft())
        try:
            validated = clamp_region(global_region, self.frame.geometry)
            image = crop_region(self.frame, validated)
        except (RuntimeError, ValueError) as exc:
            self._error_message = str(exc)
            self.selection_error.emit(str(exc))
            self._selection = QRect()
            self.update()
            return
        self._finished = True
        self.selected.emit(image, validated)
        self.hide()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._finished:
            self._cancel()
        event.accept()

    def _cancel(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.cancelled.emit()
        self.hide()


class CapturePreviewDialog(QDialog):
    """Preview one crop inside the application and offer a repeat-capture action."""

    recapture_requested = Signal()
    recognition_requested = Signal(QImage, str)

    def __init__(self, image: QImage, screen_name: str, region: QRect, parent=None) -> None:
        super().__init__(parent)
        install_theme()
        self.image = image.copy()
        self.setWindowTitle("Vista previa de captura")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        self.description = QLabel(
            f"{screen_name} · imagen {self.image.width()} × {self.image.height()} px · "
            f"región de pantalla {region.width()} × {region.height()}"
        )
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setStyleSheet("background: #20252a; border-radius: 6px")
        layout.addWidget(self.description)
        layout.addWidget(self.image_label, 1)
        orientation_row = QHBoxLayout()
        orientation_row.addWidget(QLabel("Vista del tablero"))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("Blancas abajo", "white")
        self.orientation_combo.addItem("Negras abajo", "black")
        orientation_row.addWidget(self.orientation_combo)
        orientation_row.addStretch(1)
        layout.addLayout(orientation_row)
        buttons = QHBoxLayout()
        self.recognize_button = QPushButton("Reconocer posición")
        self.recognize_button.clicked.connect(
            lambda: self.recognition_requested.emit(self.image.copy(), self.orientation_combo.currentData())
        )
        self.repeat_button = QPushButton("Capturar otra región")
        self.repeat_button.clicked.connect(self._request_recapture)
        self.close_button = QPushButton("Cerrar")
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.repeat_button)
        buttons.addWidget(self.recognize_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self._refresh_image()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_image()

    def _refresh_image(self) -> None:
        if self.image.isNull() or not hasattr(self, "image_label"):
            return
        pixmap = QPixmap.fromImage(self.image).scaled(
            QSize(max(1, self.image_label.width()), max(1, self.image_label.height())),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    def _request_recapture(self) -> None:
        self.close()
        self.recapture_requested.emit()


class RecognitionReviewDialog(QDialog):
    """Review a recognized placement and require explicit FEN metadata."""

    def __init__(self, result, parent=None, initial_board: chess.Board | None = None) -> None:
        super().__init__(parent)
        install_theme()
        from ..recognition import board_from_placement

        self.result = result
        self.setWindowTitle("Revisar posición reconocida")
        self.resize(820, 650)
        layout = QVBoxLayout(self)
        summary = QLabel(
            f"Confianza media: {result.mean_confidence:.0%}; mínima por casilla: {result.minimum_confidence:.0%}. "
            + ("Revisa las casillas dudosas." if result.uncertain_squares else "Revisa el tablero antes de continuar.")
        )
        layout.addWidget(summary)
        content = QHBoxLayout()
        self.board = BoardWidget()
        self.board.setMinimumSize(420, 420)
        self.board.set_board(board_from_placement(result.piece_placement))
        self.board.set_orientation(result.orientation)
        self.board.editable = True
        self.board.square_clicked.connect(self._edit_square)
        content.addWidget(self.board, 1)
        metadata = QFormLayout()
        self.turn = QComboBox()
        self.turn.addItem("Seleccionar…", None)
        self.turn.addItem("Blancas", "w")
        self.turn.addItem("Negras", "b")
        self.castling = QComboBox()
        self.castling.addItem("Seleccionar…", None)
        self.castling.addItem("Sin derechos (-)", "-")
        for value in ("K", "Q", "KQ", "k", "q", "kq", "Kk", "Kq", "Qk", "Qq", "KQk", "KQq", "Kkq", "Qkq", "KQkq"):
            self.castling.addItem(value, value)
        self.en_passant = QComboBox()
        self.en_passant.addItem("Seleccionar…", None)
        self.en_passant.addItem("Ninguna (-)", "-")
        for rank in "36":
            for file in "abcdefgh":
                square = f"{file}{rank}"
                self.en_passant.addItem(square, square)
        self.halfmove = QLineEdit(); self.halfmove.setPlaceholderText("Indicar (puede ser 0)"); self.halfmove.setValidator(QIntValidator(0, 999999, self))
        self.fullmove = QLineEdit(); self.fullmove.setPlaceholderText("Indicar (mínimo 1)"); self.fullmove.setValidator(QIntValidator(1, 999999, self))
        if initial_board is not None:
            for combo, value in ((self.turn, "w" if initial_board.turn else "b"),
                                 (self.castling, initial_board.castling_xfen()),
                                 (self.en_passant, initial_board.ep_square and chess.square_name(initial_board.ep_square) or "-")):
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.halfmove.setText(str(initial_board.halfmove_clock))
            self.fullmove.setText(str(initial_board.fullmove_number))
        metadata.addRow("Turno", self.turn)
        metadata.addRow("Enroque", self.castling)
        metadata.addRow("Captura al paso", self.en_passant)
        metadata.addRow("Contador medio movimiento", self.halfmove)
        metadata.addRow("Número de jugada", self.fullmove)
        content.addLayout(metadata)
        layout.addLayout(content, 1)
        self.confirm_check = QCheckBox("Revisé las casillas dudosas y la posición completa")
        self.confirm_check.setChecked(False)
        layout.addWidget(self.confirm_check)
        self.error_label = QLabel("Los campos de turno, enroque y contadores no se ven en la imagen; complétalos para generar una FEN.")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QHBoxLayout()
        self.analyze_button = QPushButton("Usar FEN y analizar")
        self.analyze_button.clicked.connect(self._accept_fen)
        cancel = QPushButton("Cancelar")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1); buttons.addWidget(cancel); buttons.addWidget(self.analyze_button)
        layout.addLayout(buttons)
        self._refresh_uncertainties()

    def _refresh_uncertainties(self) -> None:
        uncertain = self.result.uncertain_squares
        self.error_label.setText(
            ("Casillas con confianza baja: " + ", ".join(uncertain) + ". " if uncertain else "")
            + "Los campos de turno, enroque y contadores no se ven en la imagen; complétalos para generar una FEN."
        )

    def _edit_square(self, square: int) -> None:
        choices = ["Vacía", "Rey blanco", "Dama blanca", "Torre blanca", "Alfil blanco", "Caballo blanco", "Peón blanco",
                   "Rey negro", "Dama negra", "Torre negra", "Alfil negro", "Caballo negro", "Peón negro"]
        piece, ok = QInputDialog.getItem(self, f"Corregir {chess.square_name(square)}", "Pieza en la casilla:", choices, 0, False)
        if not ok:
            return
        symbols = [None, "K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
        board = self.board.board.copy(stack=False)
        symbol = symbols[choices.index(piece)]
        board.set_piece_at(square, chess.Piece.from_symbol(symbol) if symbol else None)
        self.board.set_board(board)
        self.confirm_check.setChecked(True)
        # Preserve prediction confidences; the user explicitly validates corrections here.
        self.result = type(self.result)(board.fen().split()[0], self.result.orientation,
                                        self.result.predictions, self.result.confidence_threshold,
                                        getattr(self.result, "timings", None))
        self._refresh_uncertainties()

    def _accept_fen(self) -> None:
        from ..recognition import compose_full_fen
        if not self.confirm_check.isChecked():
            self.error_label.setText("Confirma que revisaste las casillas dudosas antes de continuar.")
            return
        if self.turn.currentData() is None:
            self.error_label.setText("Selecciona quién tiene el turno antes de continuar.")
            return
        if self.castling.currentData() is None:
            self.error_label.setText("Indica los derechos de enroque o selecciona «Sin derechos (-)».")
            return
        if self.en_passant.currentData() is None:
            self.error_label.setText("Indica la casilla al paso o selecciona «Ninguna (-)».")
            return
        if not self.halfmove.text():
            self.error_label.setText("Indica el contador de medio movimiento (puede ser 0).")
            return
        if not self.fullmove.text():
            self.error_label.setText("Indica el número de jugada, empezando en 1.")
            return
        try:
            self.fen = compose_full_fen(self.board.board.fen().split()[0], self.turn.currentData(),
                self.castling.currentData(), self.en_passant.currentData(), self.halfmove.text(), self.fullmove.text())
        except (ValueError, TypeError) as exc:
            self.error_label.setText(f"No se puede usar esta posición: {exc}")
            return
        self.accept()
