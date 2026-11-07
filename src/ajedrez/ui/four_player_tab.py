"""Four-player FFA board and asynchronous local Tetrarch analysis."""
from functools import lru_cache
import re
import chess
import chess.svg
import math
import threading
import hashlib
from dataclasses import replace
from PySide6.QtCore import QByteArray, QRectF, QPointF, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QGuiApplication
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QPlainTextEdit, QLineEdit, QSpinBox, QToolButton
from ..four_player import PLAYERS, PLAYER_NAMES, PLAYER_COLORS, display_cell, playable, initial_pieces
from .. import ffa_engine
from ..ffa_recognition import recognize_ffa
from ..ffa_live import FFATracker
from ..ffa_visual_cues import ClockTurn
from ..capture.screen_capture import capture_screen,crop_region
from ..capture.window_anchor import WindowAnchor,to_logical,physical_bounds
from ..capture.live_input import left_mouse_button_down
from .capture_dialog import RegionSelector


@lru_cache(maxsize=30)
def colored_piece(kind, player):
    # Reuse the same licensed python-chess vectors as the existing board.
    svg = chess.svg.piece(chess.Piece.from_symbol(kind))
    svg = re.sub(r'#(?:ffffff|fff)\b',PLAYER_COLORS[player],svg,flags=re.IGNORECASE)
    renderer = QSvgRenderer(QByteArray(svg.encode('utf-8')))
    renderer.setParent(QApplication.instance())
    return renderer


class FourPlayerBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pieces = initial_pieces()
        self.bottom = 'red'
        self.recommendation = None
        self.setMinimumSize(128,128)
        self.setAccessibleName('Tablero de cuatro jugadores, análisis FFA')
        QApplication.instance().aboutToQuit.connect(colored_piece.cache_clear)
        self.destroyed.connect(colored_piece.cache_clear)

    def set_bottom(self, player):
        if player not in PLAYERS:
            raise ValueError('Perspectiva desconocida.')
        self.bottom = player
        self.update()

    def cell_rect(self, column, row):
        column,row = display_cell(column,row,self.bottom)
        side = min(self.width(),self.height())
        cell = side/14
        return QRectF((self.width()-side)/2+column*cell,
                      (self.height()-side)/2+row*cell,cell,cell)

    def _paint_board(self,event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for row in range(14):
            for column in range(14):
                if not playable(column,row):
                    continue
                rect = self.cell_rect(column,row)
                painter.fillRect(rect,QColor('#eee9df' if (column+row)%2 else '#79917e'))
                piece = self.pieces.get((column,row))
                if piece:
                    inset = rect.width()*.04
                    colored_piece(piece.kind,piece.player).render(painter,rect.adjusted(inset,inset,-inset,-inset))
        painter.end()

    def paintEvent(self,event):
        self._paint_board(event)
        if self.recommendation is None:return
        painter=QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        a=self.cell_rect(*self.recommendation.origin).center()
        z=self.cell_rect(*self.recommendation.destination).center()
        width=max(3,min(self.width(),self.height())/100)
        painter.setPen(QPen(QColor('#e34234'),width,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap))
        painter.drawLine(a,z)
        angle=math.atan2(z.y()-a.y(),z.x()-a.x())
        size=width*2.5
        painter.setBrush(QColor('#e34234'));painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([z,QPointF(z.x()-size*math.cos(angle-.45),z.y()-size*math.sin(angle-.45)),
                                      QPointF(z.x()-size*math.cos(angle+.45),z.y()-size*math.sin(angle+.45))]))
        painter.end()


class FFAWorker(QThread):
    updated=Signal(object)
    failed=Signal(str)

    def __init__(self,position,milliseconds):
        super().__init__()
        self.position=position.copy()
        self.milliseconds=milliseconds
        self.cancel=threading.Event()

    def run(self):
        try:
            ffa_engine.analyze(self.position,self.milliseconds,self.cancel,self.updated.emit)
        except ffa_engine.AnalysisCancelled:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class FFARecognitionWorker(QThread):
    completed=Signal(object)
    failed=Signal(str)

    def __init__(self,image,bottom,board_rect=None,previous_pieces=None):
        super().__init__();self.image=image;self.bottom=bottom;self.board_rect=board_rect
        self.previous_pieces=previous_pieces

    def run(self):
        try:self.completed.emit(recognize_ffa(self.image,self.bottom,self.board_rect,self.previous_pieces))
        except Exception as exc:self.failed.emit(str(exc))


class FourPlayerTab(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.position=ffa_engine.new_position()
        self.worker=None
        self.generation=0
        self._closed=False
        self.reading=False;self.paused=False;self.reading_generation=0
        self.recognition_worker=None;self.selector=None;self.tracker=None
        self.capture_region=None;self.capture_monitor=None;self.anchor=None
        self._last_image=None;self._confirmed_image=None;self._auto_pending=False
        self._turn_known=True
        self._explicit_position=False;self._clock_turn=ClockTurn()
        self._ffa_board_rect=None
        self.capture_timer=QTimer(self);self.capture_timer.timeout.connect(self.capture_tick)
        layout = QVBoxLayout(self)
        title = QLabel('Cuatro jugadores · Todos contra todos')
        title.setObjectName('sectionTitle')
        layout.addWidget(title)
        capture_controls=QHBoxLayout()
        self.screen_combo=QComboBox()
        for index,screen in enumerate(QGuiApplication.screens()):
            self.screen_combo.addItem(f'Pantalla {index+1} · {screen.name()}',index)
        capture_controls.addWidget(self.screen_combo)
        self.read_button=QPushButton('Leer tablero en pantalla');self.read_button.setObjectName('primaryButton')
        capture_controls.addWidget(self.read_button)
        self.pause_button=QPushButton('Pausar lectura');self.pause_button.setEnabled(False)
        capture_controls.addWidget(self.pause_button)
        self.stop_read_button=QPushButton('Detener lectura');self.stop_read_button.setEnabled(False)
        capture_controls.addWidget(self.stop_read_button)
        self.interval=QSpinBox();self.interval.setRange(100,5000);self.interval.setValue(250);self.interval.setSuffix(' ms')
        capture_controls.addWidget(self.interval);layout.addLayout(capture_controls)
        self.read_status=QLabel('Lectura automática detenida. Selecciona toda la cruz, incluidas las esquinas vacías.')
        self.read_status.setObjectName('muted');self.read_status.setWordWrap(True);layout.addWidget(self.read_status)
        self.turn_details_button=QToolButton()
        self.turn_details_button.setText('Turno real · avanzado')
        self.turn_details_button.setCheckable(True)
        self.turn_details_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.turn_details_button.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(self.turn_details_button,0,Qt.AlignmentFlag.AlignLeft)
        self.turn_details=QWidget()
        turn_controls=QHBoxLayout(self.turn_details);turn_controls.setContentsMargins(0,0,0,0)
        turn_controls.addWidget(QLabel('Corregir turno real (no el color que analizas):'))
        self.observed_turn=QComboBox();self.observed_turn.addItem('Detectar automáticamente',None)
        for color in PLAYERS:self.observed_turn.addItem(PLAYER_NAMES[color],color)
        self.observed_turn.setToolTip('Corrección opcional del seguimiento. Para elegir tus recomendaciones usa Mi color / Evaluar a.')
        turn_controls.addWidget(self.observed_turn);turn_controls.addStretch()
        layout.addWidget(self.turn_details);self.turn_details.hide()
        self.turn_details_button.toggled.connect(self.turn_details.setVisible)
        self.turn_details_button.toggled.connect(lambda checked:self.turn_details_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow))
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Vista capturada:'))
        self.perspective = QComboBox()
        for player in PLAYERS:
            self.perspective.addItem(PLAYER_NAMES[player]+' abajo',player)
        controls.addWidget(self.perspective)
        controls.addWidget(QLabel('Mi color / Evaluar a:'))
        self.player=QComboBox()
        self.player.addItem('Turno actual',None)
        for player in PLAYERS:self.player.addItem(PLAYER_NAMES[player],player)
        self.player.addItem('Mi color (abajo)','mine')
        self.player.setToolTip('Elige Rojo, Azul, Amarillo o Verde: se muestra abajo y se analizan sus jugadas. '
                               'Mi color (abajo) utiliza la orientación detectada en la captura. '
                               'El turno real y el historial siguen detectándose por separado.')
        controls.addWidget(self.player)
        self.milliseconds=QSpinBox()
        self.milliseconds.setRange(100,5000);self.milliseconds.setValue(1000)
        self.milliseconds.setSuffix(' ms')
        controls.addWidget(self.milliseconds)
        self.analyze_button=QPushButton('Analizar FFA')
        self.analyze_button.setObjectName('primaryButton')
        controls.addWidget(self.analyze_button)
        self.stop_button=QPushButton('Detener');self.stop_button.setEnabled(False)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.board = FourPlayerBoard()
        layout.addWidget(self.board,1)
        self.status = QLabel('Tetrarch FFA · una variante principal real · sin barra de evaluación.')
        self.status.setWordWrap(True)
        self.status.setObjectName('muted')
        layout.addWidget(self.status)
        self.fen4=QPlainTextEdit();self.fen4.setMaximumHeight(75)
        self.fen4.setPlaceholderText('Pega una posición FEN4 de cuatro jugadores.')
        self.fen4.setPlainText(self.position.to_fen4())
        layout.addWidget(self.fen4)
        actions=QHBoxLayout()
        self.load_button=QPushButton('Cargar FEN4');actions.addWidget(self.load_button)
        self.reset_button=QPushButton('Posición inicial');actions.addWidget(self.reset_button)
        self.move_input=QLineEdit();self.move_input.setPlaceholderText('Jugada, por ejemplo h2h3')
        actions.addWidget(self.move_input)
        self.move_button=QPushButton('Aplicar jugada');actions.addWidget(self.move_button)
        layout.addLayout(actions)
        self.turn_label=QLabel();layout.addWidget(self.turn_label)
        note=QLabel('La puntuación FFA es una utilidad del jugador analizado, no ventaja blanca/negra. '
                    'Evaluar un color fijo supone que le toca mover en esta posición; no cambia el turno real ni el historial. '
                    'Mi color sigue el lado inferior. Los puntos históricos no siempre se pueden deducir de una imagen.')
        note.setWordWrap(True);note.setObjectName('muted');layout.addWidget(note)
        self.perspective.currentIndexChanged.connect(self.perspective_changed)
        self.player.currentIndexChanged.connect(self.analysis_player_changed)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop)
        self.load_button.clicked.connect(self.load_position)
        self.reset_button.clicked.connect(self.reset_position)
        self.move_button.clicked.connect(self.apply_move)
        self.read_button.clicked.connect(self.begin_reading)
        self.pause_button.clicked.connect(self.toggle_reading)
        self.stop_read_button.clicked.connect(self.stop_reading)
        self.interval.valueChanged.connect(lambda v:self.capture_timer.setInterval(v))
        self.observed_turn.currentIndexChanged.connect(self.observed_turn_selected)
        self.set_position(self.position)

    def invalidate(self,*_):
        self.generation+=1
        self.board.recommendation=None;self.board.update()
        if self.worker:self.worker.cancel.set()

    def set_position(self,position):
        self.invalidate()
        self.position=position
        self.board.pieces=ffa_engine.pieces_of(position);self.board.update()
        self.fen4.setPlainText(position.to_fen4())
        self.turn_label.setText('Turno: '+PLAYER_NAMES[PLAYERS[position.turn]])
        self.status.setText('Posición FFA cargada · Tetrarch listo para analizar.')

    def load_position(self):
        try:
            self.set_position(ffa_engine.parse_position(self.fen4.toPlainText()))
            self._turn_known=True
            self._explicit_position=True
            if self.reading:
                self.tracker=FFATracker(self.position);self._confirmed_image=None
                self.request_auto_analysis()
        except Exception as exc:self.status.setText('No se pudo cargar FEN4: '+str(exc))

    def apply_move(self):
        try:
            position=ffa_engine.play_move(self.position,self.move_input.text())
            if self.reading:self.stop_reading()
            self.set_position(position)
            self.move_input.clear()
        except Exception as exc:self.status.setText(str(exc))

    def start_analysis(self):
        if self.worker is not None:return
        if self.reading and not self._turn_known and self.player.currentData() is None:
            self.status.setText('Turno no confirmado; esperando una jugada o selecciona «Evaluar a».');return
        self.invalidate()
        snapshot=self.position.copy()
        player=self.analysis_color()
        if PLAYERS.index(player)!=snapshot.turn:
            snapshot.turn=PLAYERS.index(player)
            snapshot.ep=[None]*4
            snapshot.recompute_key()
        if not snapshot.alive[snapshot.turn]:
            self.status.setText('Ese jugador está eliminado en esta posición.');return
        token=self.generation
        self.worker=FFAWorker(snapshot,self.milliseconds.value())
        self.worker.updated.connect(lambda r,t=token:self.show_result(r,t))
        self.worker.failed.connect(lambda reason,t=token:self.show_error(reason,t))
        self.worker.finished.connect(self.finished)
        self.analyze_button.setEnabled(False);self.stop_button.setEnabled(True)
        self.status.setText('Tetrarch está analizando FFA…')
        self.worker.start()

    def reset_position(self):
        if self.reading:self.stop_reading()
        self._turn_known=True;self._explicit_position=True;self.set_position(ffa_engine.new_position())

    def show_result(self,result,token):
        if self._closed or token!=self.generation or result.player!=self.analysis_color():return
        self.board.recommendation=result;self.board.update()
        self.status.setText(f'{PLAYER_NAMES[result.player]} · Mejor jugada: {result.move} · '
                            f'Profundidad {result.depth} · {result.seconds:.3f} s · '
                            f'Valor FFA: {result.score} (perspectiva de {PLAYER_NAMES[result.player]})')

    def show_error(self,reason,token):
        if not self._closed and token==self.generation:self.status.setText('No se pudo analizar: '+reason)

    def stop(self):
        self._auto_pending=False
        self.invalidate();self.status.setText('Análisis FFA cancelado.')

    def finished(self):
        worker=self.worker;self.worker=None
        if worker:worker.deleteLater()
        self.analyze_button.setEnabled(True);self.stop_button.setEnabled(False)
        pending=self._auto_pending;self._auto_pending=False
        if pending and not self.paused and not self._closed:
            token=self.generation
            QTimer.singleShot(0,lambda:self.start_analysis() if token==self.generation and not self._closed
                              and not self.paused else None)

    def shutdown(self):
        self.stop_reading()
        self._closed=True;self.invalidate()
        if self.worker:self.worker.wait()
        if self.recognition_worker:self.recognition_worker.wait()
        if self.selector:self.selector.close()

    def analysis_player_changed(self,*_):
        self.sync_board_view()
        self.invalidate()
        self.request_auto_analysis()

    def analysis_color(self):
        selected=self.player.currentData()
        if selected=='mine':return self.perspective.currentData()
        return selected or PLAYERS[self.position.turn]

    def sync_board_view(self):
        # Captured orientation and internal display are different coordinates:
        # choosing my colour rotates only the replicated view, never the input
        # used to map the screenshot's squares into the canonical position.
        selected=self.player.currentData()
        self.board.set_bottom(selected if selected in PLAYERS else self.perspective.currentData())

    def observed_turn_selected(self,*_):
        player=self.observed_turn.currentData()
        if player is None or not self.reading or self.tracker is None or self.tracker.position is None:return
        position=self.tracker.confirm_turn(player)
        if position is None:return
        self._turn_known=True;self.set_position(position)
        self.read_status.setText('Turno indicado: '+PLAYER_NAMES[player]+' · el seguimiento continuará automáticamente.')
        self.observed_turn.blockSignals(True);self.observed_turn.setCurrentIndex(0);self.observed_turn.blockSignals(False)
        self.request_auto_analysis()

    def perspective_changed(self,*_):
        self.sync_board_view()
        if self.player.currentData()=='mine':
            self.invalidate()
            if self.reading:self.request_auto_analysis()
        if self.reading:
            self.reading_generation+=1;self.tracker.reset_pending();self._confirmed_image=None

    def apply_observed_orientation(self,bottom):
        previous_player=self.analysis_color()
        self.perspective.blockSignals(True)
        self.perspective.setCurrentIndex(PLAYERS.index(bottom))
        self.perspective.blockSignals(False)
        self.sync_board_view()
        if previous_player!=self.analysis_color() and self.player.currentData()=='mine':self.invalidate()

    def request_auto_analysis(self):
        if self._closed or self.paused:return
        if self.worker is not None:self._auto_pending=True
        else:self.start_analysis()

    def begin_reading(self):
        self.stop_reading()
        screens=QGuiApplication.screens();index=self.screen_combo.currentData()
        if index is None or not 0<=index<len(screens):
            self.read_status.setText('No se encuentra la pantalla seleccionada.');return
        self.capture_monitor=screens[index]
        self._saved_window_state=self.window().windowState()
        self.read_button.setEnabled(False)
        self.window().showMinimized()
        QTimer.singleShot(450,self.show_selector)

    def show_selector(self):
        if self._closed:return
        try:
            self.selector=RegionSelector(capture_screen(self.capture_monitor))
            self.selector.selected.connect(self.region_selected)
            self.selector.cancelled.connect(self.selection_cancelled)
            self.selector.selection_error.connect(self.read_status.setText)
            self.selector.show();self.selector.raise_();self.selector.activateWindow()
        except Exception as exc:
            self.read_status.setText('No se pudo capturar: '+str(exc));self.restore_window()

    def dispose_selector(self):
        selector=self.selector;self.selector=None
        if selector:selector.hide();selector.deleteLater()

    def restore_window(self):
        self.read_button.setEnabled(True)
        if self._closed:return
        self.window().setWindowState(self._saved_window_state)
        self.window().show();self.window().raise_()

    def selection_cancelled(self):
        self.dispose_selector();self.restore_window()
        self.read_status.setText('Selección cancelada; puedes volver a intentarlo.')

    def region_selected(self,image,region):
        self.dispose_selector()
        self._ffa_board_rect=None
        self.capture_region=region
        self.anchor=WindowAnchor.bind(region,self.capture_monitor)
        self.restore_window()
        self.reading=True;self.paused=False;self.reading_generation+=1
        # Starting with a FEN4 matching the image preserves user-provided metadata.
        # Otherwise bootstrap without assigning the first image a certain turn.
        self.tracker=FFATracker(self.position if self._explicit_position else None)
        # Live mode is for the user's army, even during somebody else's turn.
        # An explicit colour remains selected between captures and sessions.
        if self.player.currentData() is None:
            self.player.blockSignals(True);self.player.setCurrentIndex(self.player.findData('mine'));self.player.blockSignals(False)
        self._clock_turn.reset()
        self.pause_button.setEnabled(True);self.stop_read_button.setEnabled(True)
        self.capture_timer.start(self.interval.value())
        self.start_recognition(image)

    def toggle_reading(self):
        if not self.reading:return
        self.paused=not self.paused;self.reading_generation+=1
        self.tracker.reset_pending();self._confirmed_image=None
        self._clock_turn.reset()
        if self.paused:
            self.capture_timer.stop();self.invalidate()
        else:self.capture_timer.start(self.interval.value())
        self.pause_button.setText('Reanudar lectura' if self.paused else 'Pausar lectura')
        self.read_status.setText('Lectura pausada.' if self.paused else 'Reanudando lectura…')

    def stop_reading(self):
        self.capture_timer.stop();self.reading=False;self.paused=False
        self.reading_generation+=1;self._auto_pending=False
        self._confirmed_image=None;self._last_image=None
        self._clock_turn.reset()
        self.pause_button.setEnabled(False);self.stop_read_button.setEnabled(False)
        self.pause_button.setText('Pausar lectura')
        self.read_status.setText('Lectura automática detenida.')
        self.invalidate()

    def capture_tick(self):
        if not self.reading or self.paused or self.recognition_worker is not None:return
        if left_mouse_button_down():
            self.tracker.reset_pending();self._confirmed_image=None;return
        try:
            if self.anchor:
                state=self.anchor.current()
                if state is None:
                    self.read_status.setText('Ventana no disponible; se conserva la última posición.');return
                physical=self.anchor.predicted(state)
                for screen in QGuiApplication.screens():
                    if physical_bounds(screen).contains(physical.center()):self.capture_monitor=screen;break
                self.capture_region=to_logical(physical,self.capture_monitor)
            image=crop_region(capture_screen(self.capture_monitor),self.capture_region)
            self.start_recognition(image)
        except Exception as exc:
            self.read_status.setText('Captura no disponible; se reintentará. '+str(exc))

    def start_recognition(self,image):
        if self.recognition_worker is not None or not self.reading or self.paused:return
        if left_mouse_button_down():
            self.tracker.reset_pending();return
        raw=image.convertToFormat(image.Format.Format_RGBA8888)
        digest=hashlib.blake2b(raw.constBits(),digest_size=12).digest()
        if digest==self._confirmed_image:return
        token=self.reading_generation
        self._last_image=digest
        previous=ffa_engine.pieces_of(self.tracker.position) if self.tracker and self.tracker.position else None
        source=FFARecognitionWorker(image,self.perspective.currentData(),self._ffa_board_rect,previous)
        self.recognition_worker=source
        source.completed.connect(lambda observation:self.recognition_completed(observation,token,digest))
        source.failed.connect(lambda message:self.recognition_failed(message,token))
        source.finished.connect(lambda:self.recognition_finished(source))
        source.start()

    def recognition_failed(self,message,token):
        if token==self.reading_generation and self.reading:
            self.read_status.setText('Lectura dudosa; se reintentará. '+message)

    def recognition_finished(self,source):
        if source is self.recognition_worker:self.recognition_worker=None
        source.deleteLater()

    def recognition_completed(self,observation,token,digest):
        if self._closed or token!=self.reading_generation or not self.reading or self.paused:return
        self._ffa_board_rect=observation.board_rect
        clock_player=self._clock_turn.observe(observation)
        explicit=self.observed_turn.currentData()
        if explicit is not None:
            observation=replace(observation,turn_hint=explicit,turn_source='manual')
        elif clock_player is not None:
            observation=replace(observation,turn_hint=clock_player,turn_source='clock')
        status,position=self.tracker.observe(observation,dragging=left_mouse_button_down())
        messages={'retry':'Se reintentará la lectura sin cambiar la posición',
                  'waiting':'Confirmando posición estable',
                  'unchanged':'Tablero sincronizado',
                  'updated':'Posición y turno actualizados',
                  'turn_updated':'Turno recuperado sin esperar otra jugada',
                  'unlinked':'Piezas reconocidas; esperando el reloj o indica «Juega ahora»'}
        selected=self.player.currentData()
        if selected is not None:
            color=observation.bottom if selected=='mine' else selected
            messages['unlinked']='Piezas reconocidas · análisis para '+PLAYER_NAMES[color]+' · turno real sin confirmar'
        detail=''
        if observation.uncertain:
            detail=' · Casillas dudosas: '+', '.join(chr(97+c)+str(14-r) for c,r in observation.uncertain)
        elif observation.transient:detail=' · Marcas o transición todavía ambiguas'
        self.read_status.setText(f'{messages[status]}{detail} · reconocimiento {observation.milliseconds:.1f} ms')
        if status in ('updated','unlinked','turn_updated'):
            self._turn_known=self.tracker.turn_known
            self.set_position(position)
            self.apply_observed_orientation(observation.bottom)
            if not self._turn_known:self.turn_label.setText('Turno no confirmado · datos históricos incompletos')
            if explicit is not None and self._turn_known:
                self.observed_turn.blockSignals(True);self.observed_turn.setCurrentIndex(0);self.observed_turn.blockSignals(False)
            self._confirmed_image=digest
            self.request_auto_analysis()
        elif status=='unchanged':
            self._turn_known=self.tracker.turn_known
            self.apply_observed_orientation(observation.bottom)
            self._confirmed_image=digest
            if self.board.recommendation is None:self.request_auto_analysis()
