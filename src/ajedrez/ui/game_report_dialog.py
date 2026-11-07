"""Navigable table for a completed or partial PGN analysis."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..game_analysis import (
    CLASSIFICATION_META, CRITICAL_CATEGORIES, ClassificationThresholds,
    GameAnalysisReport, MATE_SCORE_CP,
)


class ClassificationSettingsDialog(QDialog):
    """Let the user adjust approximate centipawn loss bands."""

    def __init__(self, thresholds: ClassificationThresholds, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Umbrales de clasificación")
        layout = QVBoxLayout(self)
        note = QLabel("Las categorías son aproximadas; Stockfish y la posición pueden cambiar la evaluación.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.acceptable = self._spin(thresholds.acceptable_max)
        self.inaccuracy = self._spin(thresholds.inaccuracy_max)
        self.error = self._spin(thresholds.error_max)
        form.addRow("Buena o aceptable hasta (peones)", self.acceptable)
        form.addRow("Imprecisión hasta (peones)", self.inaccuracy)
        form.addRow("Error hasta (peones)", self.error)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_values)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(centipawns: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 20.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setValue(centipawns / 100)
        return spin

    def _accept_values(self) -> None:
        values = (self.acceptable.value(), self.inaccuracy.value(), self.error.value())
        if not values[0] < values[1] < values[2]:
            QMessageBox.warning(self, "Umbrales no válidos", "Usa valores crecientes: aceptable < imprecisión < error.")
            return
        self.thresholds = ClassificationThresholds(*(round(value * 100) for value in values))
        self.accept()


class GameReportDialog(QDialog):
    move_selected = Signal(int)
    badge_visibility_changed = Signal(bool)
    closed = Signal()

    def __init__(self, report: GameAnalysisReport, parent=None) -> None:
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("Informe de análisis de partida")
        self.resize(900, 520)
        layout = QVBoxLayout(self)
        completed = sum(move.classification != "No evaluada" for move in report.moves)
        state = " · Cancelado; se muestra el progreso completado" if report.cancelled else ""
        average = report.elapsed_seconds / report.analyzed_positions if report.analyzed_positions else 0.0
        counts = {}
        for move in report.moves:
            if move.category_id != "unrated":
                counts[(move.player, move.category_id)] = counts.get((move.player, move.category_id), 0) + 1
        summary_counts = " · ".join(
            f"{player}: " + ", ".join(
                f"{CLASSIFICATION_META[key][0]} {counts[(player, key)]}"
                for key in CLASSIFICATION_META if counts.get((player, key))
            )
            for player in ("Blancas", "Negras")
            if any(counts.get((player, key)) for key in CLASSIFICATION_META)
        )
        self.summary = QLabel(
            f"{completed} de {len(report.moves)} jugadas evaluadas; "
            f"{report.processed_positions} de {report.total_positions} posiciones procesadas. "
            f"Tiempo: {report.elapsed_seconds:.1f} s; "
            f"{report.analyzed_positions} búsquedas nuevas, {report.cache_hits} reutilizadas "
            f"({average:.2f} s por búsqueda nueva). "
            f"Umbrales: {report.thresholds.acceptable_max / 100:.2f} / "
            f"{report.thresholds.inaccuracy_max / 100:.2f} / "
            f"{report.thresholds.error_max / 100:.2f} peones."
            f"{(' Conteo: ' + summary_counts + '.') if summary_counts else ''}{state}"
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(len(report.moves), 6)
        self.table.setHorizontalHeaderLabels(("N.º", "Jugador", "Jugada", "Clasificación", "Mejor alternativa", "Pérdida"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        colors = {
            "brilliant": "#7CDDB5", "great": "#80BEFF", "best": "#B6BFFF",
            "excellent": "#93DBAD", "good": "#B3D5BD", "book": "#9AC9EE",
            "inaccuracy": "#F4D278", "mistake": "#FFBC91",
            "blunder": "#FFAAA0", "miss": "#E8ADE0", "unrated": "#AEB4B7",
        }
        for row, move in enumerate(report.moves):
            move_no = f"{move.move_number}{'.' if move.player == 'Blancas' else '...'}"
            loss = self._format_loss(move.loss_cp, move.lost_forced_mate)
            name, symbol = CLASSIFICATION_META.get(move.category_id, ("No evaluada", "—"))
            values = (move_no, move.player, move.move_san, f"{symbol}  {name}",
                      move.best_move_san or "No disponible", loss)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, move.ply)
                item.setData(Qt.ItemDataRole.UserRole + 1, move.category_id)
                if column == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column in (2, 3):
                    item.setForeground(QColor(colors.get(move.category_id, colors["unrated"])))
                if column == 3:
                    item.setToolTip(move.explanation)
                    item.setFont(self.font())
                self.table.setItem(row, column, item)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        controls = QHBoxLayout()
        self.previous_critical_button = QPushButton("Anterior crítica")
        self.next_critical_button = QPushButton("Siguiente crítica")
        self.previous_critical_button.clicked.connect(lambda: self._select_critical(-1))
        self.next_critical_button.clicked.connect(lambda: self._select_critical(1))
        controls.addWidget(self.previous_critical_button)
        controls.addWidget(self.next_critical_button)
        self.badge_checkbox = QCheckBox("Mostrar distintivo en el tablero")
        self.badge_checkbox.setChecked(True)
        self.badge_checkbox.toggled.connect(self.badge_visibility_changed)
        controls.addStretch(1)
        controls.addWidget(self.badge_checkbox)
        layout.addLayout(controls)
        self.explanation = QLabel("Selecciona una jugada para ver el motivo y la variante principal.")
        self.explanation.setWordWrap(True)
        self.explanation.setMinimumHeight(46)
        layout.addWidget(self.explanation)
        self.critical_rows = [index for index, move in enumerate(report.moves)
                              if move.category_id in CRITICAL_CATEGORIES]
        self.previous_critical_button.setEnabled(bool(self.critical_rows))
        self.next_critical_button.setEnabled(bool(self.critical_rows))
        if report.moves:
            self.table.selectRow(0)

    @staticmethod
    def _format_loss(loss_cp: int | None, lost_forced_mate: bool) -> str:
        if loss_cp is None:
            return "—"
        if lost_forced_mate:
            return "Pierde mate forzado"
        if loss_cp >= MATE_SCORE_CP:
            return "Cambio de mate"
        return f"{loss_cp / 100:.2f} peones"

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if row >= 0 and self.report.moves:
            move = self.report.moves[row]
            self.move_selected.emit(move.ply)
            name, symbol = CLASSIFICATION_META.get(move.category_id, ("No evaluada", "—"))
            loss = self._format_loss(move.loss_cp, move.lost_forced_mate)
            if move.best_move_san:
                comparison = ("Coincide con la primera opción del motor."
                              if move.move_san == move.best_move_san
                              else f"Difiere de la primera opción, {move.best_move_san}.")
                actual_vs_best = f"Jugada realizada: {move.move_san}. {comparison}"
            else:
                actual_vs_best = (f"Jugada realizada: {move.move_san}; "
                                  "alternativa del motor no disponible.")
            board_result = self.report.position_results.get(move.before_fen)
            pv = " ".join(board_result.lines[0].pv_san[:12]) if board_result and board_result.lines else ""
            pv_text = f" Variante principal: {pv}." if pv else ""
            self.explanation.setText(
                f"{symbol} {name} · pérdida: {loss}. {actual_vs_best} {move.explanation}{pv_text}"
            )

    def _select_critical(self, direction: int) -> None:
        if not self.critical_rows:
            return
        current = self.table.currentRow()
        if direction > 0:
            target = next((row for row in self.critical_rows if row > current), self.critical_rows[0])
        else:
            target = next((row for row in reversed(self.critical_rows) if row < current), self.critical_rows[-1])
        self.table.selectRow(target)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.closed.emit()
        super().closeEvent(event)
