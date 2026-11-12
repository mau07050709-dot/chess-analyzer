"""Passive presentation of the existing live pipeline; no capture or engine of its own."""
from __future__ import annotations
import sys
import time
import chess
import logging
import ctypes
from ctypes import wintypes
from PySide6.QtCore import Qt, QTimer, QPointF, QRect, QSettings, QSignalBlocker
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QSpinBox, QComboBox
from ..board_coordinates import square_to_grid
from .board_widget import BoardWidget
from .analysis_controls import EvaluationBar, POSITION_PROFILES

COLORS = ("#e34234", "#2878d0", "#229447", "#e88117", "#974bd0")

def square_center(square, orientation, width, height):
    x, y = square_to_grid(square, orientation)
    return QPointF((x+.5)*width/8, (y+.5)*height/8)

def exclude_from_capture(hwnd):
    """Fail closed: older Windows makes excluded windows black, which is unsafe."""
    if sys.platform != "win32" or sys.getwindowsversion().build < 19041:
        return False
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.SetWindowDisplayAffinity.argtypes = (wintypes.HWND, wintypes.DWORD)
    user.SetWindowDisplayAffinity.restype = wintypes.BOOL
    user.GetWindowDisplayAffinity.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user.GetWindowDisplayAffinity.restype = wintypes.BOOL
    affinity = wintypes.DWORD()
    return bool(user.SetWindowDisplayAffinity(hwnd, 0x11)
                and user.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)) and affinity.value == 0x11)

def external_bar_rect(board, screen, side):
    width = min(34, max(21, round(board.height() * .038)))
    gap = max(4, round(board.height() * .012))
    x = board.x() - gap - width if side == 'left' else board.x() + board.width() + gap
    bar = QRect(x, board.y(), width, board.height())
    return bar if screen.contains(bar) else None


class ArrowOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint)
        self.setWindowTitle("Análisis Overlay · flechas")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.moves = ()
        self.orientation = "white"
        self.opacity = .86
        self.thickness = 1.
        self.count = 5
        self.capture_safe = None
        self.board_area = QRect()
        self.bar = EvaluationBar(self)
        self.bar.hide()

    def prepare(self):
        if self.capture_safe is None:
            self.capture_safe = exclude_from_capture(int(self.winId()))
        return self.capture_safe

    def clear(self):
        self.moves = ()
        self.bar.hide()
        self.hide()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self.opacity)
        moves = self.moves[:self.count]
        order = sorted(range(len(moves)), key=lambda i: -(
            (moves[i].from_square%8-moves[i].to_square%8)**2
            +(moves[i].from_square//8-moves[i].to_square//8)**2))
        # Coordinates remain logical Qt screen units. Qt maps them to monitor DPI.
        area = self.board_area if not self.board_area.isEmpty() else self.rect()
        cell = min(area.width(), area.height())/8
        for i in order:
            move = moves[i]
            offset = QPointF(area.topLeft())
            a = offset + square_center(move.from_square, self.orientation, area.width(), area.height())
            b = offset + square_center(move.to_square, self.orientation, area.width(), area.height())
            # Reuse the exact classic painter, including its rank/color priority.
            p.save()
            p.translate(a)
            p.scale(self.thickness, self.thickness)
            end = QPointF((b.x()-a.x())/self.thickness, (b.y()-a.y())/self.thickness)
            BoardWidget._draw_analysis_arrow(p, QPointF(), end, cell, QColor(COLORS[i]), i)
            p.restore()
        p.end()

class OverlayPrivacyFilter(logging.Filter):
    """Do not persist external positions in diagnostics while Overlay is selected."""
    def __init__(self, tab):
        super().__init__(); self.tab = tab
    def filter(self, record):
        return self.tab.host.mode_tabs.currentWidget() is not self.tab

class OverlayTab(QWidget):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.overlay = ArrowOverlay(host)
        self._last = None
        self._shutdown = False
        self.aligned = None
        self._verified_scope = None
        self._verified_until = 0.
        self._internal_score = None
        self.preferences = QSettings('ProyectoAjedrez', 'AnalizadorOverlay')
        layout = QVBoxLayout(self)
        title = QLabel("Análisis Overlay")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QLabel("Selecciona un tablero visible. Comparte la lectura y Stockfish del analizador clásico."))
        self.monitors = QComboBox()
        for i in range(host.screen_combo.count()):
            self.monitors.addItem(host.screen_combo.itemText(i), host.screen_combo.itemData(i))
        layout.addWidget(self.monitors)
        row = QHBoxLayout()
        self.select = QPushButton("Seleccionar tablero")
        self.start = QPushButton("Iniciar análisis Overlay")
        self.pause = QPushButton("Pausar")
        self.stop = QPushButton("Detener")
        for b in (self.select, self.start, self.pause, self.stop): row.addWidget(b)
        layout.addLayout(row)
        self.enabled = QCheckBox("Activar Overlay")
        self.enabled.setChecked(True)
        self.temporary = QCheckBox("Ocultar temporalmente las flechas")
        layout.addWidget(self.enabled)
        layout.addWidget(self.temporary)
        self.opacity = QSpinBox()
        self.opacity.setRange(10, 100); self.opacity.setValue(86); self.opacity.setSuffix(" %")
        self.thickness = QSpinBox()
        self.thickness.setRange(50, 200); self.thickness.setValue(100); self.thickness.setSuffix(" %")
        self.variants = QComboBox()
        self.variants.addItem("Cinco variantes", 5); self.variants.addItem("Primera variante", 1)
        for label, control in (("Opacidad", self.opacity), ("Grosor", self.thickness), ("Variantes visibles", self.variants)):
            row = QHBoxLayout(); row.addWidget(QLabel(label)); row.addWidget(control); layout.addLayout(row)
        self.profile = QComboBox()
        for label, key, _ in POSITION_PROFILES:
            self.profile.addItem(label, key)
        self.custom_depth = QSpinBox()
        self.custom_depth.setRange(host.depth_spin.minimum(), host.depth_spin.maximum())
        from PySide6.QtWidgets import QDoubleSpinBox
        self.seconds = QDoubleSpinBox()
        self.seconds.setDecimals(host.time_spin.decimals())
        self.seconds.setRange(host.time_spin.minimum(), host.time_spin.maximum())
        self.seconds.setSingleStep(host.time_spin.singleStep())
        self.seconds.setSuffix(' s')
        row = QHBoxLayout()
        row.addWidget(QLabel("Límite por posición"))
        for control in (self.profile, self.custom_depth, self.seconds): row.addWidget(control)
        layout.addLayout(row)
        self.profile.currentIndexChanged.connect(host.depth_profile.setCurrentIndex)
        self.custom_depth.valueChanged.connect(host.depth_spin.setValue)
        self.seconds.valueChanged.connect(host.time_spin.setValue)
        host.depth_profile.currentIndexChanged.connect(self.sync_limits)
        host.depth_spin.valueChanged.connect(self.sync_limits)
        host.time_spin.valueChanged.connect(self.sync_limits)
        self.sync_limits()
        layout.addWidget(QLabel("Los cambios se aplican a la próxima búsqueda o al pulsar Iniciar análisis Overlay."))
        self.bar_mode = QComboBox()
        for label, key in (("En el programa","program"),("Sobre el juego","external"),
                           ("En ambos","both"),("Oculta","hidden")):
            self.bar_mode.addItem(label,key)
        self.bar_side = QComboBox()
        self.bar_side.addItem("Izquierda","left");self.bar_side.addItem("Derecha","right")
        row = QHBoxLayout()
        row.addWidget(QLabel("Barra de evaluación"))
        for widget, key, default in ((self.bar_mode,'bar_mode','program'),(self.bar_side,'bar_side','left')):
            index = widget.findData(self.preferences.value(key,default))
            widget.setCurrentIndex(max(0,index))
            widget.currentIndexChanged.connect(self.bar_settings_changed)
            row.addWidget(widget)
        layout.addLayout(row)
        self.program_bar = EvaluationBar(self)
        self.program_bar.setFixedHeight(164)
        row = QHBoxLayout()
        row.addWidget(self.program_bar)
        status = QVBoxLayout()
        row.addLayout(status, 1)
        layout.addLayout(row)
        self.state = QLabel()
        self.engine = QLabel()
        self.evaluation = QLabel()
        for label in (self.state, self.engine, self.evaluation):
            label.setWordWrap(True); status.addWidget(label)
        status.addStretch()
        layout.addStretch()
        self.select.clicked.connect(self.select_region)
        self.start.clicked.connect(self.begin)
        self.pause.clicked.connect(self.toggle_pause)
        self.stop.clicked.connect(self.end)
        self.enabled.toggled.connect(self.sync)
        self.temporary.toggled.connect(self.sync)
        self.opacity.valueChanged.connect(self.sync)
        self.thickness.valueChanged.connect(self.sync)
        self.variants.currentIndexChanged.connect(self.sync)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.sync)
        self._privacy_filter = OverlayPrivacyFilter(self)
        self._log_handlers = tuple(logging.getLogger().handlers)
        for handler in self._log_handlers: handler.addFilter(self._privacy_filter)

    def sync_limits(self, *_):
        h = self.host
        for target, source in ((self.profile,h.depth_profile),(self.custom_depth,h.depth_spin),(self.seconds,h.time_spin)):
            with QSignalBlocker(target):
                if target is self.profile: target.setCurrentIndex(source.currentIndex())
                else: target.setValue(source.value())
        self.custom_depth.setEnabled(self.profile.currentData() == 'custom')
        self.seconds.setEnabled(self.profile.currentData() == 'time')

    def bar_settings_changed(self, *_):
        self.preferences.setValue('bar_mode',self.bar_mode.currentData())
        self.preferences.setValue('bar_side',self.bar_side.currentData())
        self.sync()

    def select_region(self):
        self.invalidate()
        self.host.screen_combo.setCurrentIndex(self.monitors.currentIndex())
        self.host._begin_board_reading()

    def begin(self):
        if not self.host._live_monitoring:
            self.select_region()
        elif self.host._live_paused:
            self.host._toggle_live_reading()
        self.enabled.setChecked(True)
        h = self.host
        if (h._live_monitoring and not h._live_paused and h._live_tracker is not None
                and (h._live_tracker.turn_known or h._analysis_turn is not None)):
            h._request_live_analysis(h._live_tracker.board)
        self.sync()

    def toggle_pause(self):
        self.invalidate()
        self.host._toggle_live_reading()
        self.sync()

    def end(self):
        self.invalidate()
        self.host._stop_live_reading()
        self.sync()

    def activate(self, active):
        if active:
            self.timer.start(); self.sync()
        else:
            self.timer.stop(); self.invalidate()

    def observe_geometry(self, source, context):
        h = self.host
        if (not context or not context[0].startswith('live_') or context[1] != h._live_generation
                or h._live_region is None): return
        rect = getattr(source, 'overlay_grid_rect', None)
        size = getattr(source, 'overlay_image_size', None)
        if rect is None or size is None: return
        r = h._live_region
        sx, sy = r.width()/size.width(), r.height()/size.height()
        self.aligned = (h._live_geometry_revision, QRect(r), QRect(
            round(r.x()+rect.x()*sx), round(r.y()+rect.y()*sy),
            round(rect.width()*sx), round(rect.height()*sy)))

    def invalidate(self):
        self._verified_scope = None
        self._verified_until = 0.
        self._last = None
        self.overlay.clear()

    def observe_position(self, result, context):
        h = self.host
        tracker = h._live_tracker
        if (not self.timer.isActive() or not context or context[0] != 'live_poll'
                or context[1] != h._live_generation or tracker is None
                or h._live_recognition_geometry_revision not in (None,h._live_geometry_revision)):
            return
        if result.orientation != h._live_orientation:
            self.invalidate()
            return
        if result.piece_placement == tracker.board.board_fen():
            return
        if not result.uncertain_squares:
            self.invalidate()
            return
        # A doubtful label is not evidence of a move, but a confident change
        # elsewhere is. Do not let incomplete frames extend the grace period.
        try:
            observed = chess.BaseBoard(result.piece_placement)
        except ValueError:
            self.invalidate()
            return
        uncertain = {chess.parse_square(square) for square in result.uncertain_squares}
        if any(observed.piece_at(square) != tracker.board.piece_at(square)
               for square in chess.SQUARES if square not in uncertain):
            self.invalidate()

    def sync(self, *_):
        if self._shutdown:
            return
        h = self.host
        active = h.mode_tabs.currentWidget() is self
        self.engine.setText(h.engine_status.text())
        self.evaluation.setText(h.eval_label.text()+" · "+h.best_move_label.text())
        self.state.setText(h.live_status.text())
        self.pause.setText("Reanudar" if h._live_paused else "Pausar")
        self.pause.setEnabled(h._live_monitoring)
        self.stop.setEnabled(h._live_monitoring)
        mode = self.bar_mode.currentData()
        self.program_bar.setVisible(mode in ('program','both'))
        self.bar_side.setEnabled(mode in ('external','both'))
        tracker = h._live_tracker
        region = h._live_region
        eligible = (active and h._live_monitoring and not h._live_paused and not h._closed
                    and h._live_region_valid and not h._exploring_live
                    and tracker is not None and (tracker.turn_known or h._analysis_turn is not None)
                    and region is not None and h._live_screen is not None
                    and h._live_screen.geometry().contains(region)
                    and h.board_widget.board.fen() == tracker.board.fen())
        moves = ()
        score = None
        if eligible:
            board = h._effective_analysis_board()
            scope = (h._live_generation, h._live_geometry_revision, board.fen(),
                     h._live_orientation, QRect(region))
            now = time.monotonic()
            # A drag suspends capture. Keep the verified position and give
            # the first post-release frame time to confirm it or replace it.
            holding_verified_position = h._live_drag_active and scope == self._verified_scope
            if h._live_signature_confirmed or holding_verified_position:
                self._verified_scope = scope
                self._verified_until = now + .55
            elif scope != self._verified_scope or now > self._verified_until:
                self._verified_scope = None
            valid = (scope == self._verified_scope
                     and h._live_analysis_cache_key and h._live_analysis_cache_key[0] == scope[2])
            if valid:
                lines = h._live_displayed_lines[:5]
                if lines:
                    try:
                        batch = tuple(chess.Move.from_uci(line.best_move_uci) for line in lines)
                    except (ValueError, TypeError):
                        batch = ()
                    if batch and all(move in board.legal_moves for move in batch):
                        moves = batch
                        score = ('score',lines[0].score_cp,lines[0].mate)
                elif h._live_search_finalized and not board.legal_moves:
                    score = ('terminal',not board.turn) if board.is_checkmate() else ('score',0,None)
        else:
            self._verified_scope = None
        if score is None:
            self.evaluation.setText('Evaluación: esperando un resultado para la posición confirmada.')
        program_key = (h._live_orientation,score)
        if program_key != self._internal_score:
            self._set_bar(self.program_bar, *program_key)
            self._internal_score = program_key
        if not self.enabled.isChecked():
            self.overlay.clear();self._last = None
            return
        if self.temporary.isChecked():
            moves = ()
        external = None
        if (moves or score) and region is not None:
            if (self.aligned and self.aligned[0] == h._live_geometry_revision
                    and self.aligned[1] == region):
                region = self.aligned[2]
            screen = h._live_screen.geometry()
            if not screen.contains(region):
                self.overlay.clear();self._last = None
                return
            if score and mode in ('external','both'):
                external = external_bar_rect(region, screen, self.bar_side.currentData())
                if external is None:
                    self.state.setText(h.live_status.text()+" · Sin espacio para la barra en ese lado.")
        if not moves and external is None:
            self.overlay.clear();self._last = None
            return
        if not self.overlay.prepare():
            self.overlay.clear()
            self.state.setText("Overlay oculto: Windows no permite excluirlo de la captura con seguridad.")
            return
        bounds = region.united(external) if external is not None else region
        key = (QRect(region),QRect(bounds),external,h._live_orientation,
               moves,self.opacity.value(),self.thickness.value(),self.variants.currentData(),
               score if external is not None else None)
        if key != self._last or not self.overlay.isVisible():
            overlay = self.overlay
            overlay.orientation = h._live_orientation
            overlay.moves = moves
            overlay.opacity = self.opacity.value()/100
            overlay.thickness = self.thickness.value()/100
            overlay.count = self.variants.currentData()
            overlay.board_area = region.translated(-bounds.topLeft())
            overlay.setGeometry(bounds)
            if external is not None:
                overlay.bar.setFixedWidth(external.width())
                overlay.bar.setGeometry(external.translated(-bounds.topLeft()))
                self._set_bar(overlay.bar, h._live_orientation, score)
            overlay.bar.setVisible(external is not None)
            if not overlay.isVisible():
                overlay.show()
            overlay.update()
            self._last = key

    @staticmethod
    def _set_bar(bar, orientation, score):
        if bar.orientation != orientation:
            bar.set_orientation(orientation)
        if score is None:
            bar.setValue(50)
        elif score[0] == 'terminal':
            bar.set_terminal(score[1])
        else:
            bar.set_score(*score[1:])

    def shutdown(self):
        if self._shutdown: return
        self._shutdown = True
        for handler in self._log_handlers: handler.removeFilter(self._privacy_filter)
        self.timer.stop()
        self.overlay.clear()
        self.overlay.close()
        self.overlay.deleteLater()
