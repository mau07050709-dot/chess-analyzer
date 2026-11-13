"""Main PySide6 window and non-blocking Stockfish worker."""
from __future__ import annotations

import threading
import time
import queue
import hashlib
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import chess
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QRect, QPoint
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
    QInputDialog,
    QSpinBox,
    QSplitter,
    QScrollArea,
    QTabWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QToolButton,
    QMenu,
    QWidgetAction,
)

from ..engine import AnalysisLine, AnalysisResult, UCIEngineSession, analyze_position, find_engine
from ..capture.screen_capture import ScreenFrame, capture_screen, crop_region
from ..capture.live_input import left_mouse_button_down
from ..capture.board_region import background_colors, grid_score, fitted_grid
from ..capture.move_dots import without_move_dots
from ..capture.window_anchor import WindowAnchor, physical_bounds, to_physical, to_logical
from .capture_region_worker import RegionSearchWorker
from .vulnerability_worker import VulnerabilityWorker
from .pin_worker import PinWorker
from ..vulnerability import vulnerability_key
from ..premove import PremoveMonitor, detect_premove_cells, interpret_premove
from ..recognition import recognize_position, recognize_position_auto_orientation
from ..live_board import LiveBoardTracker, legal_turn_candidates
from ..last_move_highlight import detect_highlights, detect_turn_highlights, HighlightTurnRecovery
from ..live_session import LiveSessionWatch
from ..pgn import PGNGame, PGNImportError, load_pgn_games
from ..game_analysis import (
    DEFAULT_THRESHOLDS,
    ClassificationThresholds,
    GameAnalysisReport,
    analyze_game,
)
from .board_widget import BoardWidget
from .analysis_controls import EvaluationBar, BoardEvaluationPanel, analysis_position, POSITION_PROFILES, PROFILE_DEPTHS
from .capture_dialog import CapturePreviewDialog, RecognitionReviewDialog, RegionSelector
from .game_report_dialog import ClassificationSettingsDialog, GameReportDialog
from .theme import install_theme

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSearchTicket:
    position_id: tuple[int, int]
    fen: str
    request_id: int
    multipv: int
    source: object
    cache_key: tuple | None


class AnalysisWorker(QThread):
    """Run UCI search away from the GUI thread and forward safe UI snapshots."""

    progress = Signal(object)
    completed = Signal(object, bool)
    failed = Signal(str)

    def __init__(self, fen: str, engine_path: str, seconds: float, depth: int | None, multipv: int):
        super().__init__()
        self.fen = fen
        self.engine_path = engine_path
        self.seconds = seconds
        self.depth = depth
        self.multipv = multipv
        self.cancel_event = threading.Event()
        self._analysis = None
        self._last_update = 0.0

    def stop(self) -> None:
        self.cancel_event.set()
        analysis = self._analysis
        if analysis is not None:
            analysis.stop()

    def _set_analysis(self, analysis) -> None:
        self._analysis = analysis
        if self.cancel_event.is_set():
            analysis.stop()

    def _on_progress(self, lines: tuple[AnalysisLine, ...]) -> None:
        now = time.monotonic()
        if now - self._last_update >= 0.15:
            self._last_update = now
            self.progress.emit(lines)

    def run(self) -> None:
        try:
            result = analyze_position(
                self.fen,
                engine_path=self.engine_path,
                seconds=self.seconds,
                depth=self.depth,
                multipv=self.multipv,
                cancel_event=self.cancel_event,
                progress_callback=self._on_progress,
                analysis_handle_callback=self._set_analysis,
            )
            self.completed.emit(result, self.cancel_event.is_set())
        except Exception as exc:  # Surface engine startup/exit and invalid FEN errors in the window.
            self.failed.emit(str(exc))


class GameAnalysisWorker(QThread):
    """Review all PGN positions without blocking the GUI thread."""

    progress = Signal(int, int)
    completed = Signal(object, bool)
    failed = Signal(str)

    def __init__(self, game, engine_path: str, seconds: float, depth: int | None,
                 multipv: int, cache: dict[tuple, AnalysisResult],
                 thresholds: ClassificationThresholds, *, mode: str = "personalizado",
                 threads: int = 1, hash_mb: int = 16):
        super().__init__()
        self.game = game
        self.engine_path = engine_path
        self.seconds = seconds
        self.depth = depth
        self.multipv = multipv
        self.cache = cache
        self.thresholds = thresholds
        self.mode = mode
        self.threads = threads
        self.hash_mb = hash_mb
        self.cancel_event = threading.Event()

    def stop(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            report = analyze_game(
                self.game, engine_path=self.engine_path, seconds=self.seconds,
                depth=self.depth, multipv=self.multipv, cache=self.cache,
                thresholds=self.thresholds,
                mode=self.mode, threads=self.threads, hash_mb=self.hash_mb,
                cancel_event=self.cancel_event,
                progress_callback=lambda done, total: self.progress.emit(done, total),
            )
            self.completed.emit(report, self.cancel_event.is_set())
        except Exception as exc:
            self.failed.emit(str(exc))


class PositionRecognitionWorker(QThread):
    """Run the local ONNX classifier away from the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, image, orientation: str, *, recover_turn=False, known_placement=None,
                 detect_premoves=False, premove_snapshot=None):
        super().__init__()
        self.image = image.copy()
        self.orientation = orientation
        self.recover_turn = recover_turn
        self.known_placement = known_placement
        self.last_move_highlight = None
        self.detect_premoves=detect_premoves
        self.premove_snapshot=premove_snapshot
        self.premove_highlight=None
        self.premove_evidence=None

    def run(self) -> None:
        try:
            # Calibrate the grid and inspect all visual evidence in the same coordinates.
            self.overlay_image_size = self.image.size()
            fitted = fitted_grid(self.image)
            self.overlay_grid_rect = fitted if fitted is not None else self.image.rect()
            if fitted != self.image.rect():
                self.image = self.image.copy(fitted)
            classification_image = self.image
            if self.premove_snapshot is not None and self.orientation != "auto":
                board, _, turn_known, _, _, _ = self.premove_snapshot
                if turn_known:
                    classification_image = without_move_dots(self.image, board, self.orientation)
            result = (recognize_position_auto_orientation(classification_image)
                      if self.orientation == "auto"
                      else recognize_position(classification_image, self.orientation))
            if result.timings:
                t = result.timings
                logger.info(
                    "Reconocimiento: total %.1f ms (sesión %.1f, localizar tablero %.1f, "
                    "preparar imagen %.1f, dividir casillas %.1f, ONNX %.1f, reconstrucción/validación %.1f)",
                    t.total_ms, t.model_session_ms, t.board_detection_ms,
                    t.image_prepare_ms, t.board_split_ms,
                    t.inference_ms, t.reconstruction_ms,
                )
            if self.recover_turn:
                started=time.perf_counter()
                try:
                    self.last_move_highlight=detect_highlights(self.image)
                    if self.last_move_highlight is None and self.known_placement is None:
                        self.last_move_highlight=detect_turn_highlights(
                            self.image,result.piece_placement,result.orientation,True)
                except Exception:
                    logger.exception("No se pudo obtener evidencia complementaria de resaltado")
                logger.debug("Detector de última jugada: %.3f ms",(time.perf_counter()-started)*1000)
            if self.detect_premoves:
                self.premove_highlight=detect_premove_cells(self.image)
                if self.premove_snapshot is not None:
                    board,pending,known,view,epoch,scope=self.premove_snapshot
                    if self.premove_highlight is not None or pending is not None:
                        self.premove_evidence=interpret_premove(result,self.premove_highlight,board,pending,
                            turn_known=known,orientation=view,epoch=epoch,scope=scope)
            self.completed_at = time.perf_counter()
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class LiveAnalysisWorker(QThread):
    """Own one UCI session and process only the latest requested live position."""

    completed = Signal(str, object, int)
    failed = Signal(str, str, int)
    progress = Signal(str, object, int)

    def __init__(self, engine_path: str, multipv: int, threads: int = 4, hash_mb: int = 256):
        super().__init__()
        self.engine_path = engine_path
        self.multipv = multipv
        self.threads = threads
        self.hash_mb = hash_mb
        self._requests: queue.Queue[tuple[str, float, int | None, int, int | None] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._current_cancel: threading.Event | None = None
        self._shutdown = threading.Event()
        self._request_id = 0
        self._requested_key: tuple[str, float, int | None, int | None] | None = None
        self._queued_request_key: tuple[str, float, int | None, int | None] | None = None
        self._last_progress_at = 0.0
        self._last_progress_count = 0
        self._last_progress_depth = None
        self._expected_live_pv_count = multipv

    def request_position(self, fen: str, seconds: float, depth: int | None, *, game_epoch=None) -> int:
        with self._lock:
            key = (fen, seconds, depth, game_epoch)
            if (key == self._requested_key and
                    ((self._current_cancel is not None and not self._current_cancel.is_set())
                     or self._queued_request_key == key)):
                return self._request_id
            if self._current_cancel is not None:
                logger.debug("UCI: cancelando solicitud anterior %s para priorizar %s", self._request_id, fen)
                self._current_cancel.set()
            while True:
                try:
                    self._requests.get_nowait()
                except queue.Empty:
                    break
            self._request_id += 1
            self._requested_key = key
            self._queued_request_key = key
            request_id = self._request_id
            self._requests.put((fen, seconds, depth, request_id, game_epoch))
            return request_id

    def cancel_pending(self) -> None:
        with self._lock:
            self._request_id += 1  # Invalidate even a dequeued-but-not-started request.
            if self._current_cancel is not None:
                self._current_cancel.set()
            while True:
                try:
                    self._requests.get_nowait()
                except queue.Empty:
                    break
            self._requested_key = None
            self._queued_request_key = None

    def stop(self) -> None:
        self._shutdown.set()
        self.cancel_pending()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        session = None
        session_game_epoch = None
        try:
            session = UCIEngineSession(
                self.engine_path, multipv=self.multipv, threads=self.threads, hash_mb=self.hash_mb,
            )
            while not self._shutdown.is_set():
                try:
                    request = self._requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if request is None:
                    break
                fen, seconds, depth, request_id, game_epoch = request
                with self._lock:
                    if self._queued_request_key == (fen, seconds, depth, game_epoch):
                        self._queued_request_key = None
                    if request_id != self._request_id:
                        continue
                    cancel = threading.Event()
                    self._current_cancel = cancel
                    self._last_progress_at = 0.0  # First PV of each new position is immediate.
                    self._last_progress_count = 0
                    self._last_progress_depth = None
                    self._expected_live_pv_count=min(self.multipv,chess.Board(fen).legal_moves.count())
                try:
                    if session is None:
                        session = UCIEngineSession(self.engine_path,multipv=self.multipv,
                                                   threads=self.threads,hash_mb=self.hash_mb)
                        session_game_epoch=None
                    if game_epoch is not None and game_epoch != session_game_epoch:
                        session.new_game()
                        session_game_epoch=game_epoch
                        logger.info("ENGINE_GAME_RESET worker=%x partida=%s",id(self),game_epoch)
                    logger.debug("ENGINE_SEARCH_STARTED worker=%x solicitud=%s FEN=%s",id(self),request_id,fen)
                    result = session.analyze(
                        fen, seconds=seconds, depth=depth, cancel_event=cancel,
                        progress_callback=lambda lines, f=fen, rid=request_id: self._emit_progress(f, lines, rid),
                    )
                    if not result.lines and chess.Board(fen).legal_moves.count() and not cancel.is_set():
                        raise RuntimeError("Stockfish terminó sin variantes para una posición con jugadas legales.")
                    if not cancel.is_set() and not self._shutdown.is_set():
                        self.completed.emit(fen, result, request_id)
                    else:
                        logger.debug("UCI: resultado cancelado solicitud=%s FEN=%s", request_id, fen)
                except Exception as exc:
                    if cancel.is_set() and not self._shutdown.is_set():
                        logger.warning("UCI cancelado falló; se reinicia la sesión para la solicitud más reciente: %s",exc)
                        session.close()
                        session = None
                    else:
                        self.failed.emit(fen, str(exc), request_id)
                        break
                finally:
                    with self._lock:
                        self._current_cancel = None
                        if request_id == self._request_id:
                            self._requested_key = None
        except Exception as exc:
            self.failed.emit("", str(exc), 0)
        finally:
            if session is not None:
                session.close()

    def _emit_progress(self, fen: str, lines: tuple[AnalysisLine, ...], request_id: int) -> None:
        with self._lock:
            if (request_id != self._request_id or self._shutdown.is_set()
                    or (self._current_cancel is not None and self._current_cancel.is_set())):
                return
        now = time.monotonic()
        complete_current_batch = (lines and len(lines) == self._expected_live_pv_count
                                  and len(lines) > self._last_progress_count
                                  and lines[0].depth == self._last_progress_depth)
        if now - self._last_progress_at >= 0.07 or complete_current_batch:
            self._last_progress_at = now
            self._last_progress_count = len(lines)
            self._last_progress_depth = lines[0].depth if lines else None
            self.progress.emit(fen, lines, request_id)


class MainWindow(QMainWindow):
    def __init__(self, *, responsive_layout: bool = False) -> None:
        super().__init__()
        self._responsive_layout = responsive_layout
        install_theme()
        self.setWindowTitle("Analizador de ajedrez")
        self.resize(1100, 740)
        self.setMinimumSize(420, 360) if responsive_layout else self.setMinimumSize(860, 620)
        self._analysis_turn = None
        self._selector_analysis_active = False
        self._selector_pending = False
        self.worker: AnalysisWorker | None = None
        self.engine_path: str | None = None
        self._capture_selector: RegionSelector | None = None
        self._preview_dialog: CapturePreviewDialog | None = None
        self._recognition_review: RecognitionReviewDialog | None = None
        self.recognition_worker: PositionRecognitionWorker | None = None
        self._capture_screen = None
        self._live_screen = None
        self._live_region = None
        self._live_anchor = None
        self._live_window_geometry = None
        self._live_region_colors = None
        self._live_search_area = None
        self._live_region_valid = True
        self._live_region_worker = None
        self._live_region_retry_at = 0.
        self._live_geometry_revision = 0
        self._live_recognition_geometry_revision = None
        self._live_orientation = "white"
        self._live_session_watch=LiveSessionWatch()
        self._highlight_turn_recovery=HighlightTurnRecovery()
        self._premove=PremoveMonitor()
        self._recognition_sequence=0
        self._last_live_frame_id=-1
        self._live_game_epoch=0
        self._live_tracker: LiveBoardTracker | None = None
        self._exploring_live=False
        self._exploration_board=None
        self._exploration_history=[]
        self._exploration_index=0
        self._exploration_base_fen=None
        self._live_monitoring = False
        self._live_paused = False
        self._live_setup_pending = False
        self._live_review_pending = False
        self._live_generation = 0
        self._live_pending_analysis = False
        self._live_recognition_context: tuple[str, int] | None = None
        self._live_analysis_cache: dict[tuple, AnalysisResult] = {}
        self.live_analysis_worker: LiveAnalysisWorker | None = None
        self._live_analysis_worker_key: tuple[str, int] | None = None
        self._live_analysis_request_id = 0
        self._live_analysis_cache_key: tuple | None = None
        self._live_last_signature: bytes | None = None
        self._live_signature_confirmed = False
        self._live_latest_recognition_started = 0.0
        self._live_frame_is_stable = False
        self._live_drag_active = False
        self._live_position_changed_at = 0.0
        self._live_first_eval_logged_for_fen: str | None = None
        self._live_position_revision = 0
        self._live_position_fen = None
        self._live_search_ticket = None
        self._live_search_finalized = False
        self._live_displayed_lines = ()
        self._live_engine_retries = 0
        self._closed = False
        self._vulnerability_worker = None
        self._vulnerability_token = None
        self._pin_worker = None
        self._pin_token = None
        self._pgn_games: tuple[PGNGame, ...] = ()
        self._pgn_game: PGNGame | None = None
        self._pgn_ply = 0
        self._analysis_generation = 0
        self.game_worker: GameAnalysisWorker | None = None
        self._game_analysis_generation = 0
        self._game_analysis_cache: dict[tuple, AnalysisResult] = {}
        self.classification_thresholds = DEFAULT_THRESHOLDS
        self._game_analysis_report: GameAnalysisReport | None = None
        self._game_report_dialog: GameReportDialog | None = None
        self._show_classification_badges = True
        self._build_ui()
        self._live_timer = QTimer(self)
        self._live_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._live_timer.timeout.connect(self._live_capture_tick)
        self._load_engine_path()
        self.board_widget.set_position(self.fen_edit.text())

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)
        self._outer_layout = outer
        root_outer = outer
        if self._responsive_layout:
            header = QWidget()
            header_layout = QVBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(10)
            self.header_scroll = QScrollArea()
            self.header_scroll.setWidgetResizable(True)
            self.header_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.header_scroll.setWidget(header)
            outer.addWidget(self.header_scroll)
            outer = header_layout

        title = QLabel("Analizador de ajedrez")
        title.setObjectName("title")
        subtitle = QLabel("Análisis local con Stockfish")
        subtitle.setObjectName("subtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        engine_row = QHBoxLayout()
        self.engine_status = QLabel("Buscando Stockfish…")
        self.engine_status.setObjectName("status")
        self.engine_path_label = QLabel("Motor: sin seleccionar")
        self.engine_path_label.setObjectName("muted")
        self.engine_path_label.setMinimumWidth(0)
        self.engine_path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.engine_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.browse_engine_button = QPushButton("Elegir motor…")
        self.browse_engine_button.clicked.connect(self._choose_engine)
        engine_row.addWidget(self.engine_status)
        engine_row.addWidget(self.engine_path_label, 1)
        engine_row.addWidget(self.browse_engine_button)
        outer.addLayout(engine_row)

        capture_row = QHBoxLayout()
        capture_row.addWidget(QLabel("Pantalla"))
        self.screen_combo = QComboBox()
        self.capture_button = QPushButton("Capturar región…")
        self.capture_button.clicked.connect(self._begin_capture)
        self.read_board_button = QPushButton("Leer tablero en pantalla")
        self.read_board_button.setObjectName("primaryButton")
        self.read_board_button.clicked.connect(self._begin_board_reading)
        self.capture_status = QLabel("Sin captura")
        self.capture_status.setObjectName("muted")
        self._populate_screens()
        capture_row.addWidget(self.screen_combo, 1)
        capture_row.addWidget(self.capture_button)
        capture_row.addWidget(self.read_board_button)
        capture_row.addWidget(self.capture_status)
        outer.addLayout(capture_row)

        live_row = QHBoxLayout()
        live_row.addWidget(QLabel("Actualizar cada (ms)"))
        self.live_interval_spin = QSpinBox()
        self.live_interval_spin.setRange(100, 5000)
        self.live_interval_spin.setSingleStep(25)
        self.live_interval_spin.setValue(125)
        self.live_interval_spin.valueChanged.connect(self._update_live_interval)
        live_row.addWidget(self.live_interval_spin)
        self.live_pause_button = QPushButton("Pausar lectura")
        self.live_pause_button.setEnabled(False)
        self.live_pause_button.clicked.connect(self._toggle_live_reading)
        self.live_stop_button = QPushButton("Detener lectura")
        self.live_stop_button.setEnabled(False)
        self.live_stop_button.clicked.connect(self._stop_live_reading)
        live_row.addWidget(self.live_pause_button)
        live_row.addWidget(self.live_stop_button)
        self.live_review_button = QPushButton("Corregir posición")
        self.live_review_button.setEnabled(False)
        self.live_review_button.clicked.connect(self._begin_manual_live_review)
        live_row.addWidget(self.live_review_button)
        self.live_status = QLabel("Lectura automática detenida")
        self.live_status.setObjectName("muted")
        live_row.addWidget(self.live_status, 1)
        outer.addLayout(live_row)

        pgn_row = QHBoxLayout()
        self.open_pgn_button = QPushButton("Abrir PGN…")
        self.open_pgn_button.clicked.connect(self._open_pgn_dialog)
        pgn_row.addWidget(self.open_pgn_button)
        self.pgn_game_label = QLabel("Sin partida PGN cargada")
        self.pgn_game_label.setObjectName("muted")
        pgn_row.addWidget(self.pgn_game_label, 1)
        self.pgn_turn_label = QLabel("Turno: —")
        self.pgn_turn_label.setObjectName("muted")
        pgn_row.addWidget(self.pgn_turn_label)
        outer.addLayout(pgn_row)

        game_analysis_row = QHBoxLayout()
        self.game_analyze_button = QPushButton("Analizar partida")
        self.game_analyze_button.setObjectName("primaryButton")
        self.game_analyze_button.setEnabled(False)
        self.game_analyze_button.clicked.connect(self._start_game_analysis)
        self.game_cancel_button = QPushButton("Cancelar análisis de partida")
        self.game_cancel_button.setEnabled(False)
        self.game_cancel_button.clicked.connect(self._stop_game_analysis)
        self.threshold_button = QPushButton("Umbrales…")
        self.threshold_button.clicked.connect(self._configure_classification)
        self.game_mode_combo = QComboBox()
        self.game_mode_combo.addItem("Rápido · tiempo por posición", "quick")
        self.game_mode_combo.addItem("Profundo · profundidad fija", "deep")
        self.game_mode_combo.currentIndexChanged.connect(self._update_depth_controls)
        self.game_progress = QProgressBar()
        self.game_progress.setRange(0, 100)
        self.game_progress.setValue(0)
        self.game_progress.setMinimumWidth(150)
        self.game_analysis_status = QLabel("Carga un PGN para analizar la partida completa.")
        self.game_analysis_status.setObjectName("muted")
        game_analysis_row.addWidget(self.game_analyze_button)
        game_analysis_row.addWidget(self.game_cancel_button)
        game_analysis_row.addWidget(self.threshold_button)
        game_analysis_row.addWidget(self.game_mode_combo)
        game_analysis_row.addWidget(self.game_progress)
        game_analysis_row.addWidget(self.game_analysis_status, 1)
        outer.addLayout(game_analysis_row)

        navigation_row = QHBoxLayout()
        self.pgn_start_button = QPushButton("|< Inicio")
        self.pgn_prev_button = QPushButton("< Anterior")
        self.pgn_next_button = QPushButton("Siguiente >")
        self.pgn_end_button = QPushButton("Final")
        self.pgn_start_button.clicked.connect(lambda: self._navigate_pgn(0))
        self.pgn_prev_button.clicked.connect(lambda: self._step_calculation_or_pgn(-1))
        self.pgn_next_button.clicked.connect(lambda: self._step_calculation_or_pgn(1))
        self.pgn_end_button.clicked.connect(lambda: self._navigate_pgn(self._pgn_game.move_count if self._pgn_game else 0))
        self.pgn_move_label = QLabel("Sin jugada seleccionada")
        self.pgn_move_label.setObjectName("muted")
        for button in (self.pgn_start_button, self.pgn_prev_button, self.pgn_next_button, self.pgn_end_button):
            button.setEnabled(False)
            navigation_row.addWidget(button)
        navigation_row.addWidget(self.pgn_move_label, 1)
        outer.addLayout(navigation_row)
        self.visual_options_row=QHBoxLayout()
        outer.addLayout(self.visual_options_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        splitter.setChildrenCollapsible(False)
        self.board_widget = BoardWidget()
        if not self._responsive_layout:
            self.board_widget.setMinimumSize(400, 400)
        self.board_widget.setObjectName("board")
        self.eval_bar = EvaluationBar()
        self.board_panel = BoardEvaluationPanel(self.board_widget,self.eval_bar,
                                               reserve_artwork=self._responsive_layout)
        splitter.addWidget(self.board_panel)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 0, 0, 0)
        panel_layout.setSpacing(12)

        self.return_to_game_button=QPushButton('Volver a la posición del juego')
        self.return_to_game_button.setObjectName('primaryButton')
        self.return_to_game_button.setMinimumHeight(max(36,self.return_to_game_button.fontMetrics().height()+16))
        self.return_to_game_button.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.return_to_game_button.setEnabled(False)
        self.return_to_game_button.clicked.connect(self._return_to_game_position)
        panel_layout.addWidget(self.return_to_game_button)
        self.board_widget.square_clicked.connect(self._explore_live_square)

        self.show_evaluation_bar_check=QCheckBox('Mostrar barra de evaluación')
        self.show_analysis_arrows_check=QCheckBox('Mostrar flechas de evaluación')
        for checkbox in (self.show_evaluation_bar_check,self.show_analysis_arrows_check):
            checkbox.setChecked(True)
            self.visual_options_row.addWidget(checkbox)
        self.show_evaluation_bar_check.toggled.connect(self.eval_bar.setVisible)
        self.show_analysis_arrows_check.toggled.connect(self._set_analysis_arrows_visible)
        self.custom_images_button=QPushButton('Imágenes…')
        self.custom_images_button.clicked.connect(self._configure_images)
        self.visual_options_row.addWidget(self.custom_images_button)
        self.visual_options_row.addStretch()

        turn_row=QHBoxLayout()
        self.turn_buttons=QButtonGroup(self)
        self.turn_buttons.setExclusive(True)
        self.auto_turn_button=QPushButton('Auto')
        self.white_turn_button=QPushButton('White to play')
        self.black_turn_button=QPushButton('Black to play')
        for button,key in ((self.auto_turn_button,0),(self.white_turn_button,1),(self.black_turn_button,2)):
            button.setCheckable(True);button.setObjectName('turnButton')
            self.turn_buttons.addButton(button,key);turn_row.addWidget(button)
        self.auto_turn_button.setChecked(True)
        self.turn_buttons.idClicked.connect(self._analysis_turn_selected)
        panel_layout.addLayout(turn_row)
        fen_title = QLabel("Posición FEN")
        fen_title.setObjectName("sectionTitle")
        self.fen_edit = QLineEdit(chess.STARTING_FEN)
        self.fen_edit.setClearButtonEnabled(True)
        self.fen_edit.setPlaceholderText("Pega aquí una posición FEN")
        self.fen_edit.returnPressed.connect(self._start_analysis)
        panel_layout.addWidget(fen_title)
        panel_layout.addWidget(self.fen_edit)

        panel_layout.addWidget(QLabel("Jugadas de la partida"))
        self.pgn_moves = QListWidget()
        self.pgn_moves.setMaximumHeight(130)
        self.pgn_moves.currentRowChanged.connect(self._pgn_list_selection_changed)
        panel_layout.addWidget(self.pgn_moves)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Tiempo (s)"))
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.1, 30.0)
        self.time_spin.setSingleStep(0.5)
        self.time_spin.setValue(1.5)
        settings_row.addWidget(self.time_spin)
        settings_row.addSpacing(8)
        settings_row.addWidget(QLabel("Variantes"))
        self.multipv_spin = QSpinBox()
        self.multipv_spin.setRange(1, 5)
        self.multipv_spin.setValue(3)
        settings_row.addWidget(self.multipv_spin)
        panel_layout.addLayout(settings_row)

        depth_row = QHBoxLayout()
        self.depth_check = QCheckBox("Usar profundidad fija")
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 40)
        self.depth_spin.setValue(18)
        self.depth_spin.setEnabled(False)
        self.depth_check.toggled.connect(self._depth_check_changed)
        self.depth_profile = QComboBox()
        for label, key, _ in POSITION_PROFILES:
            self.depth_profile.addItem(label, key)
        self.depth_profile.setToolTip('Límite por posición. Los perfiles de partida completa se ajustan por separado.')
        self.depth_profile.currentIndexChanged.connect(self._position_profile_changed)
        panel_layout.addWidget(self.depth_profile)
        depth_row.addWidget(self.depth_check)
        depth_row.addWidget(self.depth_spin)
        depth_row.addStretch(1)
        panel_layout.addLayout(depth_row)
        self._update_depth_controls()

        buttons = QHBoxLayout()
        self.analyze_button = QPushButton("Analizar posición")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self._start_analysis)
        self.stop_button = QPushButton("Detener análisis")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_analysis)
        buttons.addWidget(self.analyze_button)
        buttons.addWidget(self.stop_button)
        panel_layout.addLayout(buttons)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        panel_layout.addWidget(divider)

        self.eval_label = QLabel("Evaluación: esperando análisis")
        self.eval_label.setObjectName("evaluationValue")
        self.eval_label.setWordWrap(True)
        self.best_move_label = QLabel("Mejor jugada: —")
        self.best_move_label.setObjectName("bestMove")
        self.depth_label = QLabel("Profundidad: —")
        self.depth_label.setObjectName("muted")
        panel_layout.addWidget(self.eval_label)
        panel_layout.addWidget(self.best_move_label)
        panel_layout.addWidget(self.depth_label)

        variants_title = QLabel("Variantes principales")
        variants_title.setObjectName("sectionTitle")
        panel_layout.addWidget(variants_title)
        self.variants_view = QPlainTextEdit()
        self.variants_view.setReadOnly(True)
        self.variants_view.setPlaceholderText("Aquí aparecerán las líneas de análisis.")
        panel_layout.addWidget(self.variants_view, 1)
        if self._responsive_layout:
            self.analysis_scroll = QScrollArea()
            self.analysis_scroll.setWidgetResizable(True)
            self.analysis_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.analysis_scroll.setMinimumSize(0, 80)
            self.analysis_scroll.setWidget(panel)
            splitter.addWidget(self.analysis_scroll)
        else:
            splitter.addWidget(panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        outer = root_outer
        outer.addWidget(splitter, 1)

        self.message_label = QLabel("Introduce una FEN y pulsa Analizar.")
        self.message_label.setObjectName("muted")
        self.message_label.setWordWrap(True)
        if self._responsive_layout:
            self.message_label.setMaximumHeight(48)
        outer.addWidget(self.message_label)
        from .four_player_tab import FourPlayerTab
        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(root, "Ajedrez clásico")
        self.four_player_tab = FourPlayerTab()
        self.mode_tabs.addTab(self.four_player_tab, "4 jugadores · FFA")
        from .overlay import OverlayTab
        self.overlay_tab = OverlayTab(self)
        self.mode_tabs.addTab(self.overlay_tab, "Análisis Overlay")
        self.mode_tabs.currentChanged.connect(lambda _: self.overlay_tab.activate(self.mode_tabs.currentWidget() is self.overlay_tab))
        self.setCentralWidget(self.mode_tabs)
        self._adapt_layout()

    def _fit_available_screen(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        # availableGeometry already uses Qt logical units, including DPI scaling.
        self.resize(min(1280, max(420, available.width()-32)),
                    min(900, max(360, available.height()-64)))
        self.move(available.x()+(available.width()-self.width())//2,
                  available.y()+(available.height()-self.height())//2)

    def _adapt_layout(self):
        if not hasattr(self, 'header_scroll'):
            return
        compact = self.height() < 850 or self.width() < 1500
        margin = 10 if compact else 24
        self._outer_layout.setContentsMargins(margin, 8 if compact else 20, margin, 8 if compact else 20)
        self._outer_layout.setSpacing(8 if compact else 14)
        header_height = (96 if self.height() < 520 else 150) if compact else 350
        self.header_scroll.setFixedHeight(header_height)
        vertical = self.width() < 900
        orientation = Qt.Orientation.Vertical if vertical else Qt.Orientation.Horizontal
        if self.main_splitter.orientation() != orientation:
            self.main_splitter.setOrientation(orientation)
            self.main_splitter.setSizes([3, 2] if vertical else [5, 4])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt_layout()

    def _effective_analysis_board(self,board=None):
        return analysis_position(self.board_widget.board if board is None else board,
                                 None if self._exploring_live else self._analysis_turn)

    def _configure_images(self):
        from .image_settings import ImageSettingsDialog
        dialog=ImageSettingsDialog(self.board_panel,self)
        dialog.exec()

    def _explore_live_square(self,square):
        if not self._live_monitoring or self._live_tracker is None:return
        board=(self._exploration_board.copy(stack=True) if self._exploring_live
               else self._effective_analysis_board())
        origin=self.board_widget.selected_square
        piece=board.piece_at(square)
        if origin is None or (piece and piece.color==board.turn):
            self.board_widget.selected_square=square if piece and piece.color==board.turn else None
            self.board_widget.update();return
        candidates=[move for move in board.legal_moves if move.from_square==origin and move.to_square==square]
        if not candidates:return
        move=candidates[0]
        if any(m.promotion for m in candidates):
            names={'Dama':chess.QUEEN,'Torre':chess.ROOK,'Alfil':chess.BISHOP,'Caballo':chess.KNIGHT}
            name,ok=QInputDialog.getItem(self,'Promoción','Elige una pieza:',list(names),0,False)
            if not ok:return
            move=next(m for m in candidates if m.promotion==names[name])
        if not self._exploring_live:
            self._exploration_base_fen=self._live_tracker.board.fen()
            self._exploration_history=[board.copy(stack=True)]
            self._exploration_index=0
        self._exploring_live=True
        board.push(move)
        self._exploration_history=self._exploration_history[:self._exploration_index+1]
        self._exploration_history.append(board.copy(stack=True))
        self._exploration_index+=1
        self._show_exploration_position(board)

    def _update_exploration_navigation(self):
        if self._live_monitoring:
            active=self._exploring_live and bool(self._exploration_history)
            self.pgn_prev_button.setEnabled(active and self._exploration_index>0)
            self.pgn_next_button.setEnabled(active and self._exploration_index<len(self._exploration_history)-1)
            self.pgn_start_button.setEnabled(False);self.pgn_end_button.setEnabled(False)
        elif self._pgn_game is None:
            self.pgn_prev_button.setEnabled(False);self.pgn_next_button.setEnabled(False)

    def _step_calculation_or_pgn(self,direction):
        if not self._live_monitoring:
            self._navigate_pgn(self._pgn_ply+direction);return
        index=self._exploration_index+direction
        if not self._exploring_live or not 0<=index<len(self._exploration_history):return
        self._exploration_index=index
        self._show_exploration_position(self._exploration_history[index].copy(stack=True))

    def _show_exploration_position(self,board):
        self._exploration_board=board.copy(stack=True)
        self._invalidate_live_game_search('exploración temporal')
        self._invalidate_analysis_for_position()
        self._live_position_revision+=1
        self._live_displayed_lines=()
        self._clear_vulnerability_badges();self._clear_pin_badges()
        self.board_widget.selected_square=None
        self.board_widget.set_premove_squares(())
        self.board_widget.set_board(board);self.board_widget.set_best_moves(())
        self.board_widget.set_review_marker(None)
        self.fen_edit.setText(board.fen())
        self.return_to_game_button.setEnabled(True)
        self._update_exploration_navigation()
        self.pgn_move_label.setText(f'Exploración temporal · {self._exploration_index}/{len(self._exploration_history)-1} · la lectura del juego sigue activa')
        self.pgn_turn_label.setText("Turno: "+('Blancas' if board.turn else 'Negras'))
        self._request_live_analysis(board)

    def _return_to_game_position(self):
        if not self._live_monitoring or self._live_tracker is None:return
        self._exploring_live=False;self._exploration_base_fen=None
        self._exploration_board=None
        self._live_position_revision+=1
        self._invalidate_analysis_for_position()
        self._apply_live_position(self._live_tracker.board.copy(stack=True),'posición del juego restaurada.',
                                  turn_known=self._live_tracker.turn_known)

    def _set_analysis_arrows_visible(self,visible):
        self.board_widget.analysis_arrows_visible=visible
        self.board_widget.update()

    def _analysis_turn_selected(self,selection):
        turn={0:None,1:chess.WHITE,2:chess.BLACK}[selection]
        if turn==self._analysis_turn:
            return
        self._analysis_turn=turn
        if self._exploring_live and turn is not None:
            branch=analysis_position(self._exploration_board,turn)
            self._exploration_board=branch
            self._exploration_history=self._exploration_history[:self._exploration_index+1]
            self._exploration_history[self._exploration_index]=branch.copy(stack=True)
            self._update_exploration_navigation()
            self.board_widget.set_board(branch);self.board_widget.selected_square=None
            self.fen_edit.setText(branch.fen())
            self._live_position_revision+=1
        if turn is None and self._live_tracker is not None and not self._live_tracker.turn_known:
            self._highlight_turn_recovery.clear_pending()
            self._live_signature_confirmed=False
        self._selector_analysis_active=True
        self._invalidate_live_game_search('turno de análisis seleccionado')
        self._invalidate_analysis_for_position()
        self._live_displayed_lines=()
        self.board_widget.analysis_turn=turn
        self.board_widget.set_best_moves(())
        if ((self.worker and self.worker.isRunning())
                or (self.game_worker and self.game_worker.isRunning())):
            self._selector_pending=True
            return
        self._analyze_selected_turn()

    def _analyze_selected_turn(self):
        if self._closed:return
        if self.game_worker and self.game_worker.isRunning():
            self._selector_pending=True
            return
        if self._live_monitoring and self._live_tracker is not None:
            if not self._exploring_live and self._analysis_turn is None and not self._live_tracker.turn_known:
                self._apply_live_position(self._live_tracker.board.copy(stack=True),
                                          'lectura automática.',turn_known=False)
                return
        self._live_pending_analysis=False
        self._request_live_analysis(self.board_widget.board)

    def _populate_screens(self) -> None:
        self.screen_combo.clear()
        self._screens = list(QGuiApplication.screens())
        if not self._screens:
            self.screen_combo.addItem("No hay pantallas disponibles")
            self.screen_combo.setEnabled(False)
            self.capture_button.setEnabled(False)
            return
        for index, screen in enumerate(self._screens, 1):
            geometry = screen.geometry()
            self.screen_combo.addItem(
                f"Pantalla {index} · {screen.name()} ({geometry.width()} × {geometry.height()})",
                index - 1,
            )

    def _begin_board_reading(self) -> None:
        if not self._screens:
            self.live_status.setText("No hay pantallas disponibles para leer.")
            return
        if self.game_worker and self.game_worker.isRunning():
            self.live_status.setText("Cancela primero el análisis completo de la partida.")
            return
        if self._live_monitoring:
            self._stop_live_reading()
        self._live_generation += 1
        self._live_setup_pending = True
        self._live_review_pending = False
        self._live_tracker = None
        self._live_session_watch.clear()
        self.live_status.setText("Selecciona el monitor y arrastra una vez sobre el tablero.")
        self._begin_capture()

    def _begin_capture(self) -> None:
        if not self._screens or self._closed:
            return
        index = self.screen_combo.currentData()
        if index is None or not 0 <= int(index) < len(self._screens):
            self.message_label.setText("Selecciona una pantalla disponible para capturar.")
            return
        screen = self._screens[int(index)]
        self._capture_screen = screen
        self.capture_button.setEnabled(False)
        self.capture_status.setText("Preparando captura…")
        self.message_label.setText(
            "Se ocultará esta ventana brevemente. Deja visible el tablero que quieres capturar."
        )
        self.showMinimized()
        QTimer.singleShot(450, lambda selected_screen=screen: self._show_region_selector(selected_screen))

    def _show_region_selector(self, screen) -> None:
        if self._closed:
            return
        try:
            frame = capture_screen(screen)
            selector = RegionSelector(frame)
        except Exception as exc:
            self._capture_failed(f"No se pudo capturar la pantalla: {exc}")
            return
        self._capture_selector = selector
        selector.selected.connect(
            lambda image, region, source=frame: self._capture_selected(source, image, region)
        )
        selector.cancelled.connect(self._capture_cancelled)
        selector.selection_error.connect(self._capture_selection_error)
        selector.show()
        selector.raise_()
        selector.activateWindow()

    def _capture_selection_error(self, message: str) -> None:
        self.message_label.setText(message)

    def _capture_selected(self, frame: ScreenFrame, image, region) -> None:
        self._dispose_selector()
        if self._live_setup_pending:
            self._live_anchor = WindowAnchor.bind(region, self._capture_screen)
            self._live_window_geometry = self._live_anchor.current() if self._live_anchor else None
            self._live_region_colors = background_colors(image)
            self._live_region_valid = grid_score(image,self._live_region_colors)>=.80
            self._live_search_area = None
            if not self._live_region_valid:
                # A broad video selection is a viewport; keep it for subsequent searches.
                selected = to_physical(region, self._capture_screen)
                reference = self._live_window_geometry.client if self._live_window_geometry else None
                self._live_search_area = (selected, reference)
            self._live_region_retry_at = 0.
            self._live_geometry_revision += 1
        self._restore_after_capture()
        self._selected_screen = self._capture_screen
        if self._live_setup_pending:
            self._live_screen = self._capture_screen
            self._live_region = region
            self._live_monitoring = True
            self._live_timer.start(self.live_interval_spin.value())
            self._live_last_signature = self._image_signature(image)
            self._live_signature_confirmed = False
            self.capture_status.setText("Lectura automática iniciada")
            self.live_status.setText("Reconociendo el tablero y su orientación; no hace falta confirmar la posición…")
            self._prewarm_overlay_engine()
            if not self._live_region_valid:
                self._search_live_region(frame)
                return
            self._start_recognition(image, "auto", "live_setup")
            return
        self.capture_status.setText("Vista previa disponible")
        self.message_label.setText("Captura lista. El reconocimiento local está preparado para revisar la posición.")
        dialog = CapturePreviewDialog(image, frame.screen_name, region, self)
        self._preview_dialog = dialog
        dialog.recapture_requested.connect(self._begin_capture)
        dialog.recognition_requested.connect(self._recognize_capture)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _recognize_capture(self, image, orientation: str) -> None:
        if self.recognition_worker and self.recognition_worker.isRunning():
            return
        context = "live_setup" if self._live_setup_pending else "static"
        self._start_recognition(image, orientation, context)

    def _start_recognition(self, image, orientation: str, context: str) -> None:
        if self.recognition_worker is not None:
            logger.debug("Reconocimiento en curso: no se sustituye su worker por otra captura")
            return
        if context in ("live_setup", "live_setup_poll", "live_poll"):
            self._live_recognition_geometry_revision = self._live_geometry_revision
            self._live_latest_recognition_started = time.perf_counter()
        live=context in ("live_setup", "live_setup_poll", "live_poll")
        known=(self._live_tracker.board.board_fen() if self._live_tracker is not None
               and self._live_tracker.turn_known else None)
        worker = PositionRecognitionWorker(image, orientation,
                    recover_turn=live and self._analysis_turn is None,known_placement=known,
                    detect_premoves=live,premove_snapshot=(
                        self._live_tracker.board.copy(stack=True),self._premove.pending,
                        self._live_tracker.turn_known,self._live_orientation,self._premove.revision,
                        (self._live_generation,self._live_geometry_revision))
                        if live and self._live_tracker is not None else None)
        self._recognition_sequence+=1
        worker.frame_id=self._recognition_sequence
        self.recognition_worker = worker
        self._live_recognition_context = (context, self._live_generation)
        worker.completed.connect(
            lambda result, token=self._live_recognition_context, source=worker:
                self._deliver_recognition(result, token, source)
        )
        worker.failed.connect(
            lambda reason, token=self._live_recognition_context: self._recognition_failed(reason, token)
        )
        worker.finished.connect(lambda source=worker:self._recognition_worker_finished(source))
        if context == "live_poll":
            self.live_status.setText("Leyendo piezas en la captura local…")
        elif self._preview_dialog:
            self._preview_dialog.recognize_button.setEnabled(False)
        worker.start()

    def _deliver_recognition(self, result, context, source):
        if context and context[0].startswith('live_'):
            frame_id=getattr(source,'frame_id',None)
            if frame_id is not None:
                if frame_id!=self._recognition_sequence or frame_id<=self._last_live_frame_id:return
                self._last_live_frame_id=frame_id
        self.overlay_tab.observe_geometry(source, context)
        emitted = getattr(source, "completed_at", None)
        if emitted is not None:
            logger.debug("Entrega worker → interfaz: %.3f ms", (time.perf_counter()-emitted)*1000)
        # Video encoders change background pixels even when every piece is
        # settled. Two confident classifications in the SAME capture scope
        # provide semantic stability without changing the exact-image fast path.
        old_stable=self._live_frame_is_stable
        geometry_revision=self._live_geometry_revision
        semantic=False
        if (context and context[0]=='live_poll' and context[1]==self._live_generation
                and self._live_monitoring and not self._live_paused
                and self._live_recognition_geometry_revision in (None,geometry_revision)
                and not result.uncertain_squares and not left_mouse_button_down()
                and getattr(source,'premove_evidence',None) is None
                and getattr(source,'premove_highlight',None) is None):
            key=(context[1],self._live_geometry_revision,result.orientation,result.piece_placement)
            semantic=key==getattr(self,'_last_piece_observation',None)
            self._last_piece_observation=key
        else:
            self._last_piece_observation=None
        try:
            self._live_frame_is_stable=old_stable or semantic
            self._recognition_completed(result, context, source.last_move_highlight,
                                        premove=getattr(source,'premove_evidence',None),red=getattr(source,'premove_highlight',None))
        finally:
            self._live_frame_is_stable=old_stable
            if self._live_geometry_revision != geometry_revision:
                self._last_piece_observation=None

    def _recognition_completed(self, result, context: tuple[str, int] | None, highlight=None, *, premove=None,red=None) -> None:
        if context is None or self._closed:
            return
        kind, generation = context
        if kind.startswith("live_") and generation!=self._live_generation:
            logger.debug("STALE_RESULT_DISCARDED tipo=reconocimiento generación=%s actual=%s",generation,self._live_generation)
            return
        if kind.startswith("live_") and kind != "live_manual":
            geometry_ok = self._refresh_capture_geometry()
            if (not geometry_ok or (self._live_recognition_geometry_revision is not None
                    and self._live_recognition_geometry_revision != self._live_geometry_revision)):
                logger.debug("Captura descartada: la ventana cambió durante el reconocimiento")
                return
        if kind == "live_poll":
            if (not self._live_monitoring or self._live_paused or generation != self._live_generation
                    or self._live_tracker is None):
                return
            if left_mouse_button_down():
                self._live_drag_active = True
                self._live_tracker.discard_pending_candidate()
                self._highlight_turn_recovery.clear_pending()
                self._live_signature_confirmed=False
                self.live_status.setText("Arrastre en curso; se conserva la posición confirmada.")
                return
            self.overlay_tab.observe_position(result, context)
            if premove is not None:
                if premove.scope!=(self._live_generation,self._live_geometry_revision):
                    self._live_signature_confirmed=False
                    return
                if not self._premove.consume(premove,self._live_tracker.board,visual_stable=self._live_frame_is_stable):
                    self._live_signature_confirmed=False
                    return
                if not self._exploring_live:self.board_widget.set_premove_squares(self._premove.squares)
                logger.debug('PREMOVE estado=%s motivo=%s casillas=%s FEN real=%s',self._premove.state,premove.reason,
                             [chess.square_name(s) for s in self._premove.squares],self._live_tracker.board.fen())
                if premove.hold:
                    # An unresolved frame is never a confirmed image-cache hit.
                    # Keep polling even if the red overlay itself stops changing.
                    self._live_signature_confirmed=False
                    self.live_status.setText('Premove o transición pendiente; se conservan posición, turno y análisis reales.')
                    return
                result=premove.interpreted
            decision=self._live_session_watch.inspect(result,self._live_tracker,self._live_orientation,
                                                       visual_stable=self._live_frame_is_stable)
            if decision.event=="waiting":
                self._highlight_turn_recovery.clear_pending()
                self._live_signature_confirmed=False
                self.live_status.setText(decision.message)
                return
            if decision.event in ("new_game","unlinked_game"):
                if decision.event=="new_game":
                    board=chess.Board();known=True
                else:
                    candidates=legal_turn_candidates(decision.result.piece_placement)
                    board=candidates[0];known=len(candidates)==1
                logger.info("%s tipo=%s vista=%s anterior=%s nueva=%s turno_confirmado=%s",
                            "NEW_GAME_DETECTED" if decision.event=="new_game" else "GAME_SESSION_UNLINKED",
                            decision.event,decision.result.orientation,self._live_tracker.board.fen(),board.fen(),known)
                self._reset_live_game(board,decision.result.orientation,known,decision.message)
                self._try_highlight_turn_recovery(decision.result,highlight)
                return
            if decision.event=="orientation":
                self._change_live_orientation(decision)
                self._try_highlight_turn_recovery(decision.result,highlight)
                return
            validation_started = time.perf_counter()
            previous_fen = self._live_tracker.board.fen()
            update = self._live_tracker.observe(result, visual_stable=self._live_frame_is_stable)
            if update.status in ("updated", "resynced"):
                logger.info("POSITION_CONFIRMED FEN antes=%s después=%s UCI=%s turno_confirmado=%s",
                            previous_fen, update.board.fen(),
                            " ".join(move.uci() for move in update.moves), self._live_tracker.turn_known)
            validation_ms = (time.perf_counter() - validation_started) * 1000
            logger.debug("Seguimiento: estado=%s, confianza=%.3f, orientación=%s, candidato=%s, motivo=%s",
                         update.status, result.mean_confidence, result.orientation, result.piece_placement, update.message)
            if update.status in ("updated", "resynced", "review"):
                logger.info("Reconstrucción legal de transición: %.2f ms · %s",
                            validation_ms, update.status)
            if update.status == "unchanged":
                self._live_signature_confirmed = True
                self.live_status.setText("Tablero sincronizado · sin cambios.")
            elif update.status == "waiting":
                self.live_status.setText("Esperando que termine la animación del movimiento…")
            elif update.status == "retry":
                self.live_status.setText(
                    f"Captura dudosa; se conserva la posición y se reintentará. {update.message}"
                )
            elif update.status in ("updated", "resynced"):
                self._live_signature_confirmed = True
                self._apply_live_position(
                    update.board,
                    f"{update.message} Confianza: {result.mean_confidence:.0%}.",
                    turn_known=self._live_tracker.turn_known,
                )
            else:
                self._live_signature_confirmed = False
                self.live_status.setText(
                    f"La captura sigue sin ser válida; se conserva la última posición y se reintentará. "
                    f"{update.message} Usa «Corregir posición» solo si persiste."
                )
            if update.status in ("unchanged","resynced","updated"):
                self._try_highlight_turn_recovery(result,highlight)
            else:
                self._highlight_turn_recovery.clear_pending()
            if premove is not None:
                self._premove.after_tracking(self._live_tracker.board,visual_stable=self._live_frame_is_stable)
                if not self._exploring_live:self.board_widget.set_premove_squares(self._premove.squares)
                if self._premove.needs_retry:self._live_signature_confirmed=False
            return
        if kind in ("live_setup", "live_setup_poll") and generation != self._live_generation:
            return
        if kind in ("live_setup", "live_setup_poll"):
            if red is not None:
                self._live_signature_confirmed=False
                self._schedule_live_bootstrap_retry()
                self.live_status.setText('Resaltado rojo sin posición real previa; se espera una lectura no provisional.')
                return
            self._bootstrap_live_position(result)
            self._try_highlight_turn_recovery(result,highlight)
            return
        if kind == "live_manual":
            if self._live_monitoring:
                if self._live_tracker is not None:
                    self._live_review_pending = True
                self._live_paused = True
                self._live_timer.stop()
                self._open_recognition_review(
                    result, self._live_tracker.board if self._live_tracker is not None else None
                )
            return
        if self._preview_dialog:
            self._preview_dialog.close()
        self._open_recognition_review(result)

    def _try_highlight_turn_recovery(self,result,highlight):
        tracker=self._live_tracker
        if (self._analysis_turn is not None or tracker is None
                or result.uncertain_squares or result.piece_placement!=tracker.board.board_fen()
                or not self._live_monitoring or self._live_paused):
            self._highlight_turn_recovery.clear_pending()
            if hasattr(self,'_highlight_conflict_recovery'):self._highlight_conflict_recovery.clear_pending()
            return
        if tracker.turn_known:
            from ..last_move_highlight import resolve_last_move
            evidence=resolve_last_move(tracker.board,highlight,self._live_orientation)
            if (evidence is None or evidence.next_turn==tracker.board.turn
                    or not tracker.board.move_stack or self._premove.pending is not None):
                self._highlight_turn_recovery.clear_pending()
                if hasattr(self,'_highlight_conflict_recovery'):self._highlight_conflict_recovery.clear_pending()
                return
            # A stale legal reconstruction must not override a newer stable
            # visible last move. Drop unsupported history, never fabricate it.
            last=tracker.board.peek()
            if (last.from_square,last.to_square)==(evidence.origin,evidence.destination):return
            if not hasattr(self,'_highlight_conflict_recovery'):
                self._highlight_conflict_recovery=HighlightTurnRecovery()
            proven=self._highlight_conflict_recovery.observe(tracker.board,highlight,self._live_orientation,
                        (self._live_generation,self._live_geometry_revision),visual_stable=self._live_frame_is_stable)
            if proven is None:
                self._live_signature_confirmed=False;return
            corrected=tracker.board.copy(stack=False);corrected.turn=proven.next_turn;corrected.ep_square=None
            if corrected.is_valid():
                tracker.reset(corrected)
                self._highlight_conflict_recovery.reset()
                self._apply_live_position(corrected,'Turno corregido por última jugada resaltada estable.')
            return
        evidence=self._highlight_turn_recovery.observe(
            tracker.board,highlight,self._live_orientation,
            (self._live_generation,self._live_geometry_revision),visual_stable=self._live_frame_is_stable)
        if evidence is not None and tracker.confirm_visual_turn(evidence.next_turn,tracker.board.fen()):
            logger.info("TURN_RECOVERED_FROM_HIGHLIGHT origen=%s destino=%s movió=%s turno=%s FEN=%s",
                        chess.square_name(evidence.origin),chess.square_name(evidence.destination),
                        "blancas" if evidence.mover else "negras",
                        "blancas" if evidence.next_turn else "negras",tracker.board.fen())
            self._live_signature_confirmed=True
            self._apply_live_position(tracker.board.copy(stack=True),
                                      "Turno recuperado mediante la última jugada resaltada.",turn_known=True)
        elif self._highlight_turn_recovery.pending:
            # One ordinary next capture supplies temporal confirmation. With no
            # evidence, keep the old unchanged-image fast path intact.
            self._live_signature_confirmed=False

    def _recognition_failed(self, reason: str, context: tuple[str, int] | None) -> None:
        if self._closed:
            return
        if context and context[0].startswith("live_") and context[1]!=self._live_generation:
            return
        kind = context[0] if context else "static"
        if kind in ("live_poll", "live_setup_poll"):
            self.live_status.setText(f"No se pudo leer la captura; se conserva la posición anterior: {reason}")
        elif kind == "live_setup":
            self.live_status.setText(f"No se pudo iniciar todavía; se reintentará automáticamente: {reason}")
            if self._live_screen is not None and self._live_region is not None:
                self._live_monitoring = True
                self._live_paused = False
                self.live_pause_button.setEnabled(True)
                self.live_stop_button.setEnabled(True)
                self.live_review_button.setEnabled(False)
                self._live_timer.start(self.live_interval_spin.value())
        else:
            if self._preview_dialog:
                self._preview_dialog.recognize_button.setEnabled(True)
            QMessageBox.warning(self, "No se pudo reconocer", reason)

    def _recognition_worker_finished(self, source=None) -> None:
        worker = source if source is not None else self.recognition_worker
        if self.recognition_worker is worker:
            self.recognition_worker = None
            if (worker is not None and not getattr(worker,"recover_turn",False)
                    and self._analysis_turn is None and self._live_monitoring and not self._closed
                    and self._live_tracker is not None and not self._live_tracker.turn_known):
                # Auto may have been selected while this manual-mode read was
                # finishing. Its missing hint must not suppress the next read.
                self._live_signature_confirmed=False
        if worker:
            worker.deleteLater()
        if self._preview_dialog and self._preview_dialog.isVisible():
            self._preview_dialog.recognize_button.setEnabled(True)

    def _open_recognition_review(self, result, initial_board: chess.Board | None = None) -> None:
        """Open review without a nested exec loop; acceptance drives analysis once."""
        review = RecognitionReviewDialog(result, self, initial_board=initial_board)
        review.setWindowModality(Qt.WindowModality.ApplicationModal)
        review.accepted.connect(lambda dialog=review: self._recognized_position_accepted(dialog))
        review.finished.connect(lambda code, dialog=review: self._recognition_review_finished(dialog, code))
        self._recognition_review = review
        review.open()

    def _recognized_position_accepted(self, review: RecognitionReviewDialog) -> None:
        fen = review.fen
        self.fen_edit.setText(fen)
        self.board_widget.set_best_move(None)
        self.message_label.setText("FEN creada a partir de la captura; se inicia el análisis.")
        if self._live_setup_pending or self._live_review_pending:
            self._clear_premove()
            board = chess.Board(fen)
            if self._live_setup_pending:
                self._live_orientation = review.result.orientation
                if self._live_screen is None or self._live_region is None:
                    self.live_status.setText("No se guardó el monitor o la región; vuelve a seleccionar el tablero.")
                    self._live_setup_pending = False
                    self._start_analysis()
                    return
                self._live_tracker = LiveBoardTracker(board)
            else:
                assert self._live_tracker is not None
                self._live_tracker.reset(board)
            self._live_setup_pending = False
            self._live_review_pending = False
            self._live_session_watch.clear()
            self._live_signature_confirmed = True
            self._live_monitoring = True
            self._live_paused = False
            self._live_generation += 1
            self._live_timer.start(self.live_interval_spin.value())
            self.live_pause_button.setText("Pausar lectura")
            self.live_pause_button.setEnabled(True)
            self.live_stop_button.setEnabled(True)
            self.live_review_button.setEnabled(True)
            self.live_status.setText(
                "Posición confirmada. Mantén esta ventana fuera del tablero capturado para no taparlo."
            )
            for control in (self.pgn_start_button, self.pgn_prev_button, self.pgn_next_button,
                            self.pgn_end_button, self.pgn_moves):
                control.setEnabled(False)
            self._apply_live_position(board, "Posición confirmada por el usuario.")
        else:
            self._start_analysis()

    def _clear_recognition_review(self, review: RecognitionReviewDialog) -> None:
        if self._recognition_review is review:
            self._recognition_review = None

    def _recognition_review_finished(self, review: RecognitionReviewDialog, code: int) -> None:
        self._clear_recognition_review(review)
        if code == QDialog.DialogCode.Accepted:
            return
        if self._live_setup_pending:
            if self._live_monitoring:
                self._live_paused = False
                self._live_timer.start(self.live_interval_spin.value())
                self.live_status.setText("Se conserva la selección; se volverá a intentar la lectura automática.")
            else:
                self._live_setup_pending = False
                self.live_status.setText("Lectura automática no iniciada; no se confirmó la posición.")
        if self._live_review_pending:
            self._live_review_pending = False
            self._live_paused = False
            # Keep the last confirmed board and avoid reopening the same review
            # on an identical frozen frame; a new frame or explicit resume retries.
            self._live_signature_confirmed = True
            if self._live_monitoring:
                self._live_timer.start(self.live_interval_spin.value())
            self.live_pause_button.setText("Pausar lectura")
            self.live_status.setText("Se conserva la última posición válida; lectura reanudada y esperando cambios.")

    def _refresh_capture_geometry(self):
        """Observe native geometry without changing the chess/session state."""
        if self._live_anchor is None:
            return True
        current = self._live_anchor.current()
        if current is None:
            self._live_region_valid = False
            self.live_status.setText("Esperando que la ventana del tablero vuelva a estar visible.")
            return False
        if current != self._live_window_geometry:
            self._live_window_geometry = current
            self._live_geometry_revision += 1
            self._live_region_valid = False
            self._live_last_signature = None
            self._live_signature_confirmed = False
            if self._live_tracker is not None:
                self._live_tracker.discard_pending_candidate()
            self._live_session_watch.clear()
            predicted = self._live_anchor.predicted(current)
            screens = list(QGuiApplication.screens())
            preferred = next((screen for screen in screens if physical_bounds(screen).contains(predicted)),None)
            if preferred is None:
                preferred = next((screen for screen in screens if physical_bounds(screen).intersects(current.client)),None)
            if preferred is not None:
                self._live_screen = preferred
                self._live_region = to_logical(predicted,preferred)
            logger.info("Geometría de captura cambiada: revisión=%s cliente=%s DPI=%s",
                        self._live_geometry_revision,current.client,current.dpi)
        return True

    def _search_live_region(self, frame):
        self.overlay_tab.invalidate()
        if self._live_region_worker is not None or time.monotonic() < self._live_region_retry_at:
            return
        self._live_region_retry_at = time.monotonic()+1.
        self._live_signature_confirmed = False
        self.live_status.setText("Reubicando el tablero visible; se conserva la partida.")
        screens = list(QGuiApplication.screens()); searches=[]
        owner = self._live_window_geometry.client if self._live_window_geometry else None
        for index,screen in enumerate(screens):
            bounds = physical_bounds(screen)
            search = bounds.intersected(owner) if owner is not None else bounds
            if self._live_search_area is not None:
                selected, reference = self._live_search_area
                if reference is not None and owner is not None:
                    sx, sy = owner.width()/reference.width(), owner.height()/reference.height()
                    selected = QRect(owner.x()+round((selected.x()-reference.x())*sx),
                                     owner.y()+round((selected.y()-reference.y())*sy),
                                     round(selected.width()*sx), round(selected.height()*sy))
                search = search.intersected(selected)
            if search.width()<128 or search.height()<128:
                continue
            try:
                snapshot = frame if screen is self._live_screen else capture_screen(screen)
                local = search.translated(-bounds.x(),-bounds.y())
                image = snapshot.image.copy(local)
                excluded=[]
                if self.isVisible() and not self.isMinimized():
                    visible_own = self.frameGeometry().intersected(screen.geometry())
                    if not visible_own.isEmpty():
                        own = to_physical(visible_own,screen).intersected(search)
                        if not own.isEmpty():excluded.append(own.translated(-search.x(),-search.y()))
                searches.append((index,search.topLeft(),image,tuple(excluded)))
            except (RuntimeError,ValueError):
                continue
        if not searches:return
        worker=RegionSearchWorker(searches,self._live_region_colors,self)
        self._live_region_worker=worker
        token=(self._live_generation,self._live_geometry_revision,self._live_window_geometry)
        worker.completed.connect(lambda match,source=worker:self._live_region_found(match,token,screens,source))
        worker.finished.connect(lambda source=worker:self._live_region_search_finished(source))
        worker.start()

    def _live_region_found(self,match,token,screens,source):
        if (source is not self._live_region_worker or self._closed or self._live_paused
                or not self._live_monitoring or token[:2] != (self._live_generation,self._live_geometry_revision)):
            return
        if self._live_anchor and self._live_anchor.current()!=token[2]:
            return
        if match is None:return
        screen=screens[match.screen_index]
        if screen not in QGuiApplication.screens():return
        region=to_logical(match.region,screen)
        if self.isVisible() and not self.isMinimized() and self.frameGeometry().intersects(region):return
        self._live_screen=screen;self._live_region=region
        if getattr(match,'colors',None) is not None:self._live_region_colors=match.colors
        self._live_region_valid=True
        self._live_geometry_revision+=1
        self._live_recognition_geometry_revision=None
        self._live_last_signature=None;self._live_signature_confirmed=False
        if self._live_tracker is not None:self._live_tracker.discard_pending_candidate()
        self._live_session_watch.clear()
        if self._live_anchor and self._live_window_geometry:
            self._live_anchor.update(self._live_window_geometry,match.region)
        logger.info("Región reubicada sin reiniciar partida: pantalla=%s región=%s confianza=%.3f",
                    screen.name(),region,match.score)

    def _live_region_search_finished(self,source):
        if self._live_region_worker is source:self._live_region_worker=None
        source.deleteLater()

    def _live_capture_tick(self) -> None:
        if (not self._live_monitoring or self._live_paused or self._live_screen is None
                or self._live_region is None or self.recognition_worker is not None):
            return
        if self._live_region_worker is not None:
            return
        if not self._refresh_capture_geometry():
            return
        if self._live_region_valid and self.isVisible() and not self.isMinimized() and self.frameGeometry().intersects(self._live_region):
            self._live_signature_confirmed=False
            self.overlay_tab.invalidate()
            self.live_status.setText("El analizador se superpone a la región seleccionada; aparta o minimiza esta ventana para evitar capturar sus propias piezas y flechas.")
            logger.debug("Captura omitida: el analizador se superpone a la región %s",self._live_region)
            return
        if left_mouse_button_down():
            self._live_drag_active = True
            self._live_signature_confirmed = False
            if self._live_tracker is not None:
                self._live_tracker.discard_pending_candidate()
            self.live_status.setText("Arrastre detectado; se conserva la última posición confirmada.")
            return
        if self._live_drag_active:
            self._live_drag_active = False
            self._live_signature_confirmed = False
        capture_started = time.perf_counter()
        try:
            frame = capture_screen(self._live_screen)
            screen_capture_ms = (time.perf_counter() - capture_started) * 1000
            if not self._live_region_valid:
                self._search_live_region(frame)
                return
            if self._live_region_colors is not None and (
                    not frame.geometry.contains(self._live_region)
                    or (not self._live_region_valid and self.isVisible() and not self.isMinimized()
                        and self.frameGeometry().intersects(self._live_region))):
                self._live_region_valid = False
                self._search_live_region(frame)
                return
            crop_started = time.perf_counter()
            image = crop_region(frame, self._live_region)
            crop_ms = (time.perf_counter() - crop_started) * 1000
        except Exception as exc:
            self.live_status.setText("Recuperando la región visible; se conserva la posición confirmada.")
            self._live_region_valid = False
            if 'frame' in locals():
                self._search_live_region(frame)
            logger.debug("Captura pendiente de reubicación: %s",exc)
            return
        capture_ms = (time.perf_counter() - capture_started) * 1000
        logger.info("Captura de pantalla %.1f ms · recorte %.1f ms · total %.1f ms",
                    screen_capture_ms, crop_ms, capture_ms)
        if self._live_region_colors is not None:
            score = grid_score(image, self._live_region_colors)
            if score < .80:
                self._live_region_valid = False
                self._search_live_region(frame)
                return
            if not self._live_region_valid:
                self._live_region_valid = True
                self._live_last_signature = None
                self._live_signature_confirmed = False
                if self._live_anchor and self._live_window_geometry:
                    self._live_anchor.update(self._live_window_geometry, to_physical(self._live_region,self._live_screen))
        signature = self._image_signature(image)
        self._live_frame_is_stable = signature == self._live_last_signature
        if signature == self._live_last_signature and self._live_signature_confirmed:
            self.live_status.setText(
                f"Tablero sin cambios · captura {capture_ms:.0f} ms · no se repite el reconocimiento."
            )
            return
        if signature != self._live_last_signature:
            self._live_last_signature = signature
            self._live_signature_confirmed = False
        self._live_latest_recognition_started = time.perf_counter()
        context = "live_setup_poll" if self._live_setup_pending else "live_poll"
        orientation = "auto" if self._live_setup_pending else self._live_orientation
        self._start_recognition(image, orientation, context)

    def _bootstrap_live_position(self, result) -> None:
        """Start live tracking directly from a stable initial or legal midgame view."""
        if result.uncertain_squares:
            self.live_status.setText(
                f"Lectura inicial dudosa ({len(result.uncertain_squares)} casillas); "
                "se conserva la selección y se volverá a intentar automáticamente."
            )
            self._schedule_live_bootstrap_retry()
            return
        placement = result.piece_placement
        start_placement = chess.Board().board_fen()
        turn_known = True
        if placement == start_placement:
            board = chess.Board()
        else:
            candidates = list(legal_turn_candidates(placement))
            if not candidates:
                self.live_status.setText(
                    "La posición inicial observada no es legal; se conserva la región y se reintentará."
                )
                self._schedule_live_bootstrap_retry()
                return
            if len(candidates) == 1:
                board = candidates[0]
            else:
                # With no visible history, turn cannot always be derived from
                # piece placement. Choose an explicit provisional default and
                # disclose it rather than presenting it as a visual fact.
                board = candidates[0]
                turn_known = False
        self.live_pause_button.setText("Pausar lectura")
        self.live_pause_button.setEnabled(True)
        self.live_stop_button.setEnabled(True)
        self.live_review_button.setEnabled(True)
        for control in (self.pgn_start_button, self.pgn_prev_button, self.pgn_next_button,
                        self.pgn_end_button, self.pgn_moves):
            control.setEnabled(False)
        if placement == start_placement:
            explanation = "Posición inicial estándar detectada; turno blanco y enroques iniciales confirmados."
        else:
            explanation = "Posición legal detectada; historial desconocido, derechos de enroque desactivados."
            if not turn_known:
                explanation += " Turno provisional: blancas."
        self._reset_live_game(board,result.orientation,turn_known,explanation)
        self.capture_status.setText("Lectura automática activa")

    def _invalidate_live_game_search(self,reason):
        logger.info("ENGINE_SEARCH_CANCELLED motivo=%s posición=%s búsqueda=%s",reason,
                    (self._live_generation,self._live_position_revision),self._live_analysis_request_id)
        if self.live_analysis_worker is not None:
            self.live_analysis_worker.cancel_pending()
        self._live_search_ticket=None
        self._live_analysis_request_id+=1
        self._live_search_finalized=False
        self._live_pending_analysis=False

    def _reset_live_game(self,board,orientation,turn_known,message):
        """Reset only chess state; keep the selected screen/region and UCI worker."""
        self._exploring_live=False;self._exploration_base_fen=None
        self._exploration_board=None
        self._clear_premove()
        self._invalidate_live_game_search("nueva sesión de partida")
        self._live_generation+=1
        self._live_game_epoch+=1
        self._live_orientation=orientation
        self._live_tracker=LiveBoardTracker(board,turn_known=turn_known)
        self._highlight_turn_recovery.reset()
        self._live_session_watch.clear()
        self._live_analysis_cache.clear()
        self._live_position_fen=None
        self._live_engine_retries=0
        self._live_setup_pending=False;self._live_review_pending=False
        self._live_monitoring=True;self._live_paused=False
        self._live_signature_confirmed=True
        self._live_drag_active=False
        self._live_timer.start(self.live_interval_spin.value())
        logger.info("GAME_STATE_RESET partida=%s generación=%s vista=%s FEN=%s turno_confirmado=%s",
                    self._live_game_epoch,self._live_generation,orientation,board.fen(),turn_known)
        if not turn_known:message+=" Turno provisional; Stockfish espera evidencia legal."
        self._apply_live_position(board,message,turn_known=turn_known)

    def _change_live_orientation(self,decision):
        old=self._live_orientation
        self._invalidate_live_game_search("cambio de orientación confirmado")
        self._live_generation+=1
        self._live_orientation=decision.result.orientation
        self._live_session_watch.clear()
        if decision.result.piece_placement!=self._live_tracker.board.board_fen():
            update=self._live_tracker.observe_stable(decision.result,decision.observations)
            message=f"{decision.message} {update.message}"
        else:message=decision.message
        self._live_signature_confirmed=decision.result.piece_placement==self._live_tracker.board.board_fen()
        logger.info("ORIENTATION_CHANGED anterior=%s nueva=%s partida=%s FEN=%s turno_confirmado=%s",
                    old,self._live_orientation,self._live_game_epoch,self._live_tracker.board.fen(),self._live_tracker.turn_known)
        self._apply_live_position(self._live_tracker.board.copy(stack=True),message,turn_known=self._live_tracker.turn_known)

    def _schedule_live_bootstrap_retry(self) -> None:
        self._live_setup_pending = True
        self._live_monitoring = True
        self._live_paused = False
        self.live_pause_button.setEnabled(True)
        self.live_stop_button.setEnabled(True)
        self.live_review_button.setEnabled(True)
        self._live_timer.start(self.live_interval_spin.value())

    def _begin_manual_live_review(self) -> None:
        if not self._live_monitoring or self._live_screen is None or self._live_region is None:
            return
        try:
            image = crop_region(capture_screen(self._live_screen), self._live_region)
        except Exception as exc:
            self.live_status.setText(f"No se pudo preparar la corrección manual: {exc}")
            return
        self._start_recognition(image, self._live_orientation, "live_manual")

    @staticmethod
    def _image_signature(image) -> bytes:
        """Hash a 64×64 sample grid without allocating or retaining Qt buffers."""
        width, height = image.width(), image.height()
        # Identical sample positions and QRgb bytes to the scalar version.
        # Convert once only when needed; no per-pixel Python/Qt calls.
        native = (QImage.Format.Format_RGB32, QImage.Format.Format_ARGB32,
                  QImage.Format.Format_ARGB32_Premultiplied)
        rgba = image if image.format() in native else image.convertToFormat(QImage.Format.Format_ARGB32)
        rows = np.frombuffer(rgba.constBits(), dtype=np.uint32).reshape(height, rgba.bytesPerLine()//4)
        xs = np.minimum(width-1, (np.arange(64)*2+1)*width//128)
        ys = np.minimum(height-1, (np.arange(64)*2+1)*height//128)
        pixels = rows[ys[:,None], xs[None,:]].astype('<u4', copy=False).tobytes()
        return hashlib.blake2b(pixels, digest_size=12).digest()

    def _apply_live_position(self, board: chess.Board, message: str, *, turn_known: bool = True) -> None:
        if self._exploring_live and board.fen()==self._exploration_base_fen:
            self.board_widget.set_orientation(self._live_orientation)
            return
        self._exploring_live=False;self._exploration_base_fen=None
        self._exploration_board=None
        self.board_widget.selected_square=None
        self._exploration_history=[];self._exploration_index=0
        self._update_exploration_navigation()
        self.board_widget.movable=self._live_monitoring
        self.return_to_game_button.setEnabled(False)
        if turn_known and self._live_monitoring:
            self._highlight_turn_recovery.remember_known(board)
        if board.fen() != self._live_position_fen:
            self.overlay_tab.invalidate()
            self._live_position_revision += 1
            self._live_position_fen = board.fen()
            self._live_engine_retries = 0
        self._live_search_ticket = None
        self._live_search_finalized = False
        self._live_displayed_lines = ()
        logger.debug("Tablero lógico actualizado; se limpian flechas: FEN=%s turno=%s enroques=%s confirmado=%s",
                     board.fen(), "blancas" if board.turn else "negras", board.castling_xfen(), turn_known)
        if self._live_latest_recognition_started:
            latency_ms = (time.perf_counter() - self._live_latest_recognition_started) * 1000
            logger.info("Latencia captura estable → tablero actualizado: %.1f ms", latency_ms)
        self.fen_edit.setText(board.fen())
        self._live_position_changed_at = time.perf_counter()
        self._live_first_eval_logged_for_fen = None
        self.board_widget.set_board(board)
        self.board_widget.set_orientation(self._live_orientation)
        self.board_widget.set_best_move(None)
        self.board_widget.set_review_marker(None)
        self._request_vulnerability_badges(board, turn_known)
        self._request_pin_badges(board, turn_known)
        self.pgn_turn_label.setText(f"Turno: {'Blancas' if board.turn else 'Negras'}")
        self.pgn_move_label.setText(f"Posición sincronizada · {message}")
        self.live_status.setText(f"Tablero actualizado · {message}")
        if not turn_known and self._analysis_turn is None:
            if self.live_analysis_worker is not None:
                self.live_analysis_worker.cancel_pending()
            self._live_analysis_request_id += 1
            self._live_pending_analysis = False
            self.board_widget.set_best_moves(())
            self.best_move_label.setText("Mejor jugada: esperando confirmar el turno")
            self.eval_label.setText("Evaluación: pendiente; el turno es provisional")
            self.eval_bar.setValue(50)
            self.depth_label.setText("Profundidad: —")
            self.variants_view.clear()
            self.engine_status.setText("Stockfish en espera · turno no confirmado")
            self.message_label.setText(
                "La posición de piezas se sincronizó, pero la imagen no demuestra quién mueve. "
                "Se conserva el turno como provisional hasta una transición legal que lo aclare."
            )
            return
        if self.worker and self.worker.isRunning():
            self._live_pending_analysis = True
            self._invalidate_analysis_for_position()
            self.engine_status.setText("Cancelando búsqueda de la posición anterior…")
        else:
            self._request_live_analysis(board)

    def _clear_premove(self):
        self._premove.reset()
        self.board_widget.set_premove_squares(())

    def _request_vulnerability_badges(self, board, turn_known):
        if not self._live_monitoring or self._closed:
            return
        key = vulnerability_key(board, turn_known)
        token = (self._live_generation, board.fen(), key)
        if token == self._vulnerability_token:
            return
        self._vulnerability_token = token
        worker = self._vulnerability_worker
        if worker is None:
            worker = VulnerabilityWorker(self)
            worker.completed.connect(self._vulnerability_badges_ready)
            self._vulnerability_worker = worker
            worker.start()
        worker.request_position(board, turn_known, token)

    def _vulnerability_badges_ready(self, token, key, result):
        if (self._closed or not self._live_monitoring or token != self._vulnerability_token
                or token[0] != self._live_generation or token[1] != self.board_widget.board.fen()
                or token[2] != key):
            return
        self.board_widget.set_vulnerable_squares(result.squares, expected_fen=token[1])
        logger.debug("Indicadores materiales: %.3f ms · nodos=%s · completo=%s · casillas=%s",
                     result.elapsed_ms, result.nodes, result.complete,
                     [chess.square_name(square) for square in result.squares])

    def _clear_vulnerability_badges(self):
        self._vulnerability_token = None
        if self._vulnerability_worker is not None:
            self._vulnerability_worker.cancel_pending()
        self.board_widget.set_vulnerable_squares(())

    def _request_pin_badges(self, board, turn_known):
        if not self._live_monitoring or self._closed:
            return
        key=vulnerability_key(board,turn_known)
        token=(self._live_generation,board.fen(),key)
        if token==self._pin_token:return
        self._pin_token=token
        if self._pin_worker is None:
            self._pin_worker=PinWorker(self)
            self._pin_worker.completed.connect(self._pin_badges_ready)
            self._pin_worker.start()
        self._pin_worker.request_position(board,turn_known,token)

    def _pin_badges_ready(self, token, key, result):
        if (self._closed or not self._live_monitoring or token!=self._pin_token
                or token[0]!=self._live_generation or token[1]!=self.board_widget.board.fen()
                or token[2]!=key):return
        self.board_widget.set_pinned_squares(result.squares,expected_fen=token[1])
        logger.debug("Chinchetas: %.3f ms · candidatas=%s · completo=%s · casillas=%s",
                     result.elapsed_ms,result.nodes,result.complete,
                     [chess.square_name(square) for square in result.squares])

    def _clear_pin_badges(self):
        self._pin_token=None
        if self._pin_worker is not None:self._pin_worker.cancel_pending()
        self.board_widget.set_pinned_squares(())

    def _update_live_interval(self, _value: int) -> None:
        if self._live_monitoring and not self._live_paused:
            self._live_timer.start(self.live_interval_spin.value())

    def _request_live_analysis(self, board: chess.Board) -> None:
        if self._exploring_live and board.fen()!=self.board_widget.board.fen():return
        board = self._effective_analysis_board(board)
        self.board_widget.analysis_turn = None if self._exploring_live else self._analysis_turn
        if self._live_analysis_cache_key and self._live_analysis_cache_key[0]!=board.fen():
            self._live_displayed_lines=()
            self.board_widget.set_best_moves(())
        try:
            engine_path = find_engine(self.engine_path)
        except FileNotFoundError as exc:
            self.engine_status.setText("Stockfish no está disponible")
            self.message_label.setText(str(exc))
            return
        mode = self.game_mode_combo.currentData()
        depth = self._position_depth()
        seconds = self.time_spin.value() if depth is None else 1.5
        # Keep five alternatives in the live view; Stockfish streams their
        # current-depth lines progressively from one MultiPV search.
        multipv = 5 if self._live_monitoring else self.multipv_spin.value()
        cache_key = (board.fen(), engine_path, round(seconds, 3), depth, multipv, mode, 4, 256)
        ticket = self._live_search_ticket
        if (ticket is not None and not self._live_search_finalized and ticket.cache_key == cache_key
                and ticket.position_id == (self._live_generation,self._live_position_revision)
                and ticket.source is self.live_analysis_worker and ticket.source.isRunning()):
            return
        self._live_analysis_cache_key = cache_key
        cached = self._live_analysis_cache.get(cache_key)
        if not self._live_displayed_lines:
            self.best_move_label.setText("Mejor jugada: calculando…")
            self.eval_label.setText("Evaluación: calculando…")
            self.eval_bar.setValue(50)
            self.depth_label.setText("Profundidad: —")
            self.variants_view.clear()
        if cached and cached.lines:
            if self.live_analysis_worker is not None:
                self.live_analysis_worker.cancel_pending()
            self._live_analysis_request_id += 1
            self._live_search_ticket = None
            self._live_search_finalized = True
            self._live_displayed_lines = cached.lines
            self._render_lines(cached.lines)
            self.engine_status.setText("Stockfish · resultado local reutilizado")
            self.message_label.setText("La posición ya tenía un análisis completo con estos ajustes.")
            self.stop_button.setEnabled(False)
            return
        worker = self._ensure_live_analysis_worker(engine_path, multipv)
        request_id = worker.request_position(board.fen(), seconds, depth,game_epoch=self._live_game_epoch)
        self._bind_live_search(board.fen(),request_id,worker,cache_key)
        self.stop_button.setEnabled(True)
        self.engine_status.setText("Stockfish conectado · analizando posición detectada")
        self.message_label.setText("La posición cambió; se está analizando en segundo plano.")

    def _ensure_live_analysis_worker(self, engine_path, multipv):
        worker_key = (engine_path, multipv)
        worker = self.live_analysis_worker
        if worker is None or self._live_analysis_worker_key != worker_key or not worker.isRunning():
            if worker is not None and worker.isRunning():
                worker.stop()
            worker = LiveAnalysisWorker(engine_path, multipv)
            worker.setParent(self)
            self.live_analysis_worker = worker
            self._live_analysis_worker_key = worker_key
            worker.completed.connect(
                lambda fen, result, request_id, source=worker:
                self._live_analysis_completed(fen, result, request_id, source)
            )
            worker.progress.connect(
                lambda fen, lines, request_id, source=worker:
                self._live_analysis_progress(fen, lines, request_id, source)
            )
            worker.failed.connect(
                lambda fen, reason, request_id, source=worker:
                self._live_analysis_failed(fen, reason, request_id, source)
            )
            worker.finished.connect(lambda source=worker: self._live_analysis_finished(source))
            worker.start()
        return worker

    def _prewarm_overlay_engine(self):
        if not self.overlay_tab.timer.isActive() or (self.worker and self.worker.isRunning()):
            return
        try:
            self._ensure_live_analysis_worker(find_engine(self.engine_path), 5)
        except FileNotFoundError:
            pass  # The normal position request reports a missing engine.

    def _bind_live_search(self,fen,request_id,source,cache_key=None):
        self._live_analysis_request_id=request_id
        self._live_search_ticket=LiveSearchTicket(
            (self._live_generation,self._live_position_revision),fen,request_id,
            getattr(source,"multipv",5),source,cache_key)
        self._live_search_finalized=False
        logger.debug("Solicitud vinculada: posición=%s worker=%x búsqueda=%s MultiPV=%s FEN=%s configuración=%s",
                     self._live_search_ticket.position_id,id(source),request_id,
                     self._live_search_ticket.multipv,fen,cache_key)

    def _live_ticket_matches(self,fen,request_id,source):
        ticket=self._live_search_ticket
        return (ticket is not None and ticket.source is source and source is self.live_analysis_worker
                and ticket.request_id==request_id==self._live_analysis_request_id
                and ticket.position_id==(self._live_generation,self._live_position_revision)
                and ticket.fen==fen==self._effective_analysis_board().fen()
                and self.fen_edit.text().strip()==self.board_widget.board.fen()
                and ticket.multipv==getattr(source,"multipv",5)
                and (self._live_monitoring or self._selector_analysis_active) and not self._closed)

    def _render_live_batch(self,lines,*,final=False):
        lines=tuple(sorted(lines,key=lambda line:line.index))
        if not lines:
            return False
        legal=self._effective_analysis_board().legal_moves
        if (len(lines)>self._live_search_ticket.multipv
                or tuple(line.index for line in lines)!=tuple(range(1,len(lines)+1))
                or len({line.depth for line in lines})!=1
                or len({line.best_move_uci for line in lines})!=len(lines)):
            logger.warning("Lote MultiPV rechazado: índices/profundidades/movimientos incoherentes")
            return False
        try:
            if any(not line.best_move_uci or chess.Move.from_uci(line.best_move_uci) not in legal for line in lines):
                raise ValueError
        except ValueError:
            logger.warning("Lote MultiPV rechazado: movimiento ajeno a la FEN confirmada %s",self.fen_edit.text())
            return False
        if not final and self._live_displayed_lines:
            previous_depth=self._live_displayed_lines[0].depth or 0
            if len(lines)<len(self._live_displayed_lines) or (lines[0].depth or 0)<previous_depth:
                logger.debug("Se conserva el lote completo actual mientras llega el nuevo: %s PV",len(lines))
                return False
        self._live_displayed_lines=lines
        logger.debug("Flechas reemplazadas: posición=%s búsqueda=%s FEN=%s PV=%s",
                     self._live_search_ticket.position_id,self._live_search_ticket.request_id,
                     self._live_search_ticket.fen,[(line.index,line.best_move_uci,line.depth) for line in lines])
        self._render_lines(lines)
        return True

    def _live_analysis_completed(self, fen: str, result: AnalysisResult, request_id: int,
                                 source: LiveAnalysisWorker) -> None:
        if not self._live_ticket_matches(fen,request_id,source) or result.fen!=fen:
            logger.debug("STALE_RESULT_DISCARDED solicitud=%s actual=%s FEN=%s actual=%s origen_vigente=%s lectura=%s",
                         request_id, self._live_analysis_request_id, fen, self.fen_edit.text().strip(),
                         source is self.live_analysis_worker, self._live_monitoring)
            return
        if result.lines:
            if not self._render_live_batch(result.lines,final=True):
                return
            if self._live_position_changed_at:
                logger.info("Stockfish refinó las variantes tras %.1f ms desde el cambio de posición",
                            (time.perf_counter() - self._live_position_changed_at) * 1000)
            if self._live_search_ticket.cache_key:
                self._live_analysis_cache[self._live_search_ticket.cache_key] = result
                if len(self._live_analysis_cache) > 128:
                    self._live_analysis_cache.pop(next(iter(self._live_analysis_cache)))
        elif not self._effective_analysis_board().legal_moves.count():
            board=self._effective_analysis_board()
            self.board_widget.set_best_moves(())
            self.best_move_label.setText("Mejor jugada: no hay jugadas legales")
            self.variants_view.clear()
            if board.is_checkmate():
                winner=not board.turn
                self.eval_label.setText(f"Mate: ganan {'blancas' if winner else 'negras'}")
                self.eval_bar.set_terminal(winner)
            else:
                self.eval_label.setText("Tablas: posición sin jugadas legales")
                self.eval_bar.setValue(50)
        self._live_search_finalized=True
        self.stop_button.setEnabled(False)
        self.engine_status.setText("Stockfish conectado · análisis finalizado")
        self.message_label.setText("La mejor jugada y la evaluación corresponden a la posición sincronizada.")

    def _live_analysis_progress(self, fen: str, lines: tuple[AnalysisLine, ...], request_id: int,
                                source: LiveAnalysisWorker) -> None:
        if (not self._live_ticket_matches(fen,request_id,source)
                or self._live_search_finalized or not lines):
            logger.debug("Progreso UCI descartado: solicitud=%s actual=%s FEN=%s actual=%s origen_vigente=%s",
                         request_id, self._live_analysis_request_id, fen, self.fen_edit.text().strip(),
                         source is self.live_analysis_worker)
            return
        if not self._render_live_batch(lines):
            return
        if self._live_first_eval_logged_for_fen != fen and self._live_position_changed_at:
            self._live_first_eval_logged_for_fen = fen
            logger.info("Primera evaluación de Stockfish visible tras %.1f ms desde el cambio de posición",
                        (time.perf_counter() - self._live_position_changed_at) * 1000)
            self.engine_status.setText("Stockfish · evaluación inicial; refinando variantes…")

    def _live_analysis_failed(self, fen: str, reason: str, request_id: int,
                              source: LiveAnalysisWorker) -> None:
        if not (self._live_monitoring or self._selector_analysis_active) or self._closed or self._live_search_finalized:
            return
        if source is not self.live_analysis_worker:
            return
        if request_id not in (0, self._live_analysis_request_id):
            return
        if fen and fen != self._effective_analysis_board().fen():
            return
        self.stop_button.setEnabled(False)
        self.engine_status.setText("Error del motor")
        self.engine_status.setStyleSheet("color: #FFAAA0; font-weight: 650")
        self.message_label.setText(f"No se pudo analizar la posición detectada: {reason}")

    def _live_analysis_finished(self, source: LiveAnalysisWorker) -> None:
        logger.debug("Worker finalizado=%x vigente=%s",id(source),self.live_analysis_worker is source)
        if self.live_analysis_worker is source:
            self.live_analysis_worker = None
            self._live_analysis_worker_key = None
            if (self._live_monitoring and not self._live_paused and not self._closed
                    and self._live_tracker is not None and self._live_tracker.turn_known
                    and not self._live_search_finalized and self._live_search_ticket is not None
                    and self._live_engine_retries<2):
                self._live_engine_retries+=1
                token=(self._live_generation,self._live_position_revision,self.fen_edit.text().strip())
                self.engine_status.setText("Recuperando Stockfish para la posición actual…")
                QTimer.singleShot(100*self._live_engine_retries,
                                  lambda state=token:self._retry_live_analysis(state))
        # Release finished QThreads and their signal closures on the GUI thread.
        # Otherwise closed windows retain workers until interpreter teardown.
        source.deleteLater()

    def _retry_live_analysis(self,state):
        if (state==(self._live_generation,self._live_position_revision,self.fen_edit.text().strip())
                and self._live_monitoring and not self._live_paused and not self._closed
                and self.live_analysis_worker is None and self._live_tracker is not None
                and (self._live_tracker.turn_known or self._analysis_turn is not None)
                and self._live_tracker.board.fen()==self.fen_edit.text().strip()):
            self._request_live_analysis(self._live_tracker.board.copy(stack=True))

    def _toggle_live_reading(self) -> None:
        if not self._live_monitoring:
            return
        if not self._live_paused:
            self._live_paused = True
            self._live_timer.stop()
            self.live_pause_button.setText("Reanudar lectura")
            self.live_status.setText("Lectura pausada; se conserva la última posición válida.")
        else:
            self._live_paused = False
            self._live_signature_confirmed = False
            self._live_timer.start(self.live_interval_spin.value())
            self.live_pause_button.setText("Pausar lectura")
            self.live_status.setText("Lectura reanudada.")
            if self._live_tracker is not None and self._live_tracker.turn_known and self.live_analysis_worker is None and not self._live_search_finalized:
                self._request_live_analysis(self._live_tracker.board.copy(stack=True))

    def _stop_live_reading(self) -> None:
        self._exploring_live=False;self._exploration_base_fen=None
        self._exploration_board=None
        self.board_widget.movable=False;self.board_widget.selected_square=None
        self.return_to_game_button.setEnabled(False)
        self._clear_premove()
        self._clear_vulnerability_badges()
        self._clear_pin_badges()
        self._live_session_watch.clear()
        self._live_search_ticket=None
        self._live_search_finalized=True
        self._live_generation += 1
        self._live_timer.stop()
        self._live_monitoring = False
        self._exploration_history=[];self._exploration_index=0
        self._update_exploration_navigation()
        self._live_paused = False
        self._live_setup_pending = False
        self._live_review_pending = False
        self._live_pending_analysis = False
        self._live_tracker = None
        self._live_analysis_request_id += 1
        if self.live_analysis_worker is not None:
            self.live_analysis_worker.cancel_pending()
        self.stop_button.setEnabled(False)
        self.live_pause_button.setText("Pausar lectura")
        self.live_pause_button.setEnabled(False)
        self.live_stop_button.setEnabled(False)
        self.live_review_button.setEnabled(False)
        self.live_status.setText("Lectura automática detenida.")
        self.read_board_button.setEnabled(True)
        if self._pgn_game is not None:
            for control in (self.pgn_start_button, self.pgn_prev_button, self.pgn_next_button,
                            self.pgn_end_button, self.pgn_moves):
                control.setEnabled(True)

    def _capture_cancelled(self) -> None:
        if self._closed:
            return
        self._dispose_selector()
        self._restore_after_capture()
        self.capture_status.setText("Captura cancelada")
        self.message_label.setText("Captura cancelada. No se cambió la posición ni el análisis.")
        if self._live_setup_pending:
            self._live_setup_pending = False
            self.live_status.setText("Selección cancelada; lectura automática no iniciada.")

    def _capture_failed(self, message: str) -> None:
        self._dispose_selector()
        self._restore_after_capture()
        self.capture_status.setText("Error de captura")
        self.capture_status.setStyleSheet("color: #FFAAA0; font-weight: 650")
        self.message_label.setText(message)
        if self._live_setup_pending:
            self._live_setup_pending = False
            self.live_status.setText("No se inició la lectura porque falló la captura.")

    def _restore_after_capture(self) -> None:
        self.capture_button.setEnabled(bool(self._screens))
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _dispose_selector(self) -> None:
        selector = self._capture_selector
        self._capture_selector = None
        if selector is not None:
            selector.hide()
            selector.deleteLater()

    def _load_engine_path(self) -> None:
        try:
            self.engine_path = find_engine()
            self.engine_path_label.setText(f"Motor: {self.engine_path}")
            self.engine_path_label.setToolTip(self.engine_path)
            self.engine_status.setText("Stockfish encontrado · se conectará al analizar")
        except FileNotFoundError as exc:
            self.engine_status.setText("Stockfish no encontrado")
            self.engine_status.setStyleSheet("color: #FFAAA0; font-weight: 650")
            self.message_label.setText(str(exc))

    def _choose_engine(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Selecciona Stockfish para Windows", "", "Ejecutable (*.exe);;Todos los archivos (*)"
        )
        if not path:
            return
        try:
            self.engine_path = find_engine(path)
        except FileNotFoundError as exc:
            self.message_label.setText(str(exc))
            self.engine_status.setText("La ruta del motor no es válida")
            return
        self.engine_path_label.setText(f"Motor: {self.engine_path}")
        self.engine_path_label.setToolTip(self.engine_path)
        self.engine_status.setText("Motor seleccionado · se comprobará al analizar")
        self.engine_status.setStyleSheet("")

    def _open_pgn_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir partida PGN", "", "Partidas PGN (*.pgn);;Todos los archivos (*)")
        if path:
            self._load_pgn(path)

    def _load_pgn(self, path: str) -> bool:
        try:
            games = load_pgn_games(path)
        except PGNImportError as exc:
            QMessageBox.warning(self, "No se pudo abrir el PGN", str(exc))
            self.message_label.setText(str(exc))
            return False
        self._pgn_games = games
        if len(games) == 1:
            selected = 0
        else:
            labels = [self._pgn_title(game, index) for index, game in enumerate(games, 1)]
            choice, accepted = QInputDialog.getItem(self, "Elegir partida", f"El archivo contiene {len(games)} partidas:", labels, 0, False)
            if not accepted:
                return False
            selected = labels.index(choice)
        self._set_pgn_game(self._pgn_games[selected], Path(path).name)
        return True

    @staticmethod
    def _pgn_title(game: PGNGame, index: int) -> str:
        headers = game.headers
        white, black = headers.get("White", "Blancas"), headers.get("Black", "Negras")
        event, result = headers.get("Event", "Evento desconocido"), headers.get("Result", "*")
        return f"{index}. {white} – {black} | {event} | {result}"

    def _set_pgn_game(self, game: PGNGame, file_name: str = "") -> None:
        if self._live_monitoring:
            self._stop_live_reading()
        self._game_analysis_generation += 1
        if self.game_worker and self.game_worker.isRunning():
            self.game_worker.stop()
        if self._game_report_dialog:
            self._game_report_dialog.close()
            self._game_report_dialog = None
        self._game_analysis_report = None
        self._invalidate_analysis_for_position()
        self._pgn_game = game
        self._pgn_ply = 0
        headers = game.headers
        details = f"{headers.get('White', 'Blancas')} – {headers.get('Black', 'Negras')}"
        if headers.get("Event"):
            details += f" · {headers['Event']}"
        if headers.get("Date"):
            details += f" · {headers['Date']}"
        if headers.get("Result"):
            details += f" · {headers['Result']}"
        self.pgn_game_label.setText(f"{file_name}: {details}" if file_name else details)
        self.pgn_moves.blockSignals(True)
        self.pgn_moves.clear()
        self.pgn_moves.addItem("Inicio")
        for ply in range(1, game.move_count + 1):
            self.pgn_moves.addItem(game.move_label(ply))
        self.pgn_moves.setCurrentRow(0)
        self.pgn_moves.blockSignals(False)
        for button in (self.pgn_start_button, self.pgn_prev_button, self.pgn_next_button, self.pgn_end_button):
            button.setEnabled(True)
        self._show_pgn_position(0)
        self.game_analyze_button.setEnabled(not (self.game_worker and self.game_worker.isRunning()))
        self.game_analysis_status.setText("Partida cargada. Pulsa «Analizar partida» para revisar sus jugadas.")
        self.game_progress.setValue(0)
        self.message_label.setText(f"Partida PGN cargada: {game.move_count} jugada(s). Selecciona una posición y pulsa Analizar.")

    def _start_game_analysis(self) -> None:
        if self._live_monitoring:
            self.game_analysis_status.setText("Detén primero la lectura automática del tablero.")
            return
        if self._pgn_game is None:
            self.game_analysis_status.setText("Primero abre una partida PGN.")
            return
        if self.worker and self.worker.isRunning():
            self.game_analysis_status.setText("Detén primero el análisis de posición actual.")
            return
        if self.game_worker and self.game_worker.isRunning():
            return
        try:
            engine_path = find_engine(self.engine_path)
        except FileNotFoundError as exc:
            self.game_analysis_status.setText(f"No se pudo iniciar el análisis: {exc}")
            return
        mode = self.game_mode_combo.currentData()
        # Batch searches use a fixed, modest resource profile for a 6-core CPU
        # and 16 GB RAM. Single-position analysis keeps its existing settings.
        depth = self.depth_spin.value() if mode == "deep" else None
        self._game_analysis_generation += 1
        generation = self._game_analysis_generation
        game = self._pgn_game
        self.game_progress.setValue(0)
        self.game_analysis_status.setText("Preparando análisis de la partida…")
        self.game_analyze_button.setEnabled(False)
        self.game_cancel_button.setEnabled(True)
        self.threshold_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.game_worker = GameAnalysisWorker(
            game, engine_path, self.time_spin.value(), depth, self.multipv_spin.value(),
            dict(self._game_analysis_cache), self.classification_thresholds,
            mode=mode, threads=4, hash_mb=256,
        )
        self.game_worker.generation = generation
        self.game_worker.progress.connect(
            lambda done, total, token=generation: self._game_analysis_progress(done, total, token)
        )
        self.game_worker.completed.connect(
            lambda report, cancelled, token=generation: self._game_analysis_completed(report, cancelled, token)
        )
        self.game_worker.failed.connect(lambda reason, token=generation: self._game_analysis_failed(reason, token))
        self.game_worker.finished.connect(self._game_analysis_finished)
        self.game_worker.start()

    def _configure_classification(self) -> None:
        dialog = ClassificationSettingsDialog(self.classification_thresholds, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.classification_thresholds = dialog.thresholds
            values = self.classification_thresholds
            self.game_analysis_status.setText(
                f"Umbrales configurados: {values.acceptable_max / 100:.2f} / "
                f"{values.inaccuracy_max / 100:.2f} / {values.error_max / 100:.2f} peones."
            )

    def _position_depth(self):
        profile = self.depth_profile.currentData()
        return self.depth_spin.value() if profile == 'custom' else PROFILE_DEPTHS[profile]

    def _position_profile_changed(self, *_args):
        from PySide6.QtCore import QSignalBlocker
        with QSignalBlocker(self.depth_check):
            self.depth_check.setChecked(self.depth_profile.currentData() != 'time')
        self._update_depth_controls()

    def _depth_check_changed(self, checked):
        self.depth_profile.setCurrentIndex(self.depth_profile.findData('custom' if checked else 'time'))
        self._update_depth_controls()

    def _update_depth_controls(self, *_args) -> None:
        batch_deep = getattr(self, "game_mode_combo", None) is not None and self.game_mode_combo.currentData() == "deep"
        self.depth_spin.setEnabled(self.depth_profile.currentData() == 'custom' or batch_deep)

    def _game_analysis_progress(self, done: int, total: int, generation: int) -> None:
        if generation != self._game_analysis_generation:
            return
        percent = round(done * 100 / total) if total else 100
        self.game_progress.setValue(percent)
        self.game_analysis_status.setText(f"Analizando posiciones: {done} de {total}…")

    def _game_analysis_completed(self, report: GameAnalysisReport, cancelled: bool, generation: int) -> None:
        if generation != self._game_analysis_generation or report.game is not self._pgn_game:
            return
        self._game_analysis_cache.update(report.cache_entries)
        self._game_analysis_report = report
        self.game_progress.setValue(round(report.processed_positions * 100 / report.total_positions)
                                    if report.total_positions else 100)
        if cancelled:
            self.game_analysis_status.setText(
                f"Análisis cancelado; {report.processed_positions} de {report.total_positions} posiciones procesadas."
            )
        else:
            average = (report.elapsed_seconds / report.analyzed_positions
                       if report.analyzed_positions else 0.0)
            self.game_analysis_status.setText(
                f"Análisis terminado: {len(report.moves)} jugadas; "
                f"{report.analyzed_positions} búsquedas nuevas, {report.cache_hits} reutilizadas; "
                f"{report.elapsed_seconds:.1f} s total ({average:.2f} s/búsqueda)."
            )
        dialog = GameReportDialog(report, self)
        dialog.move_selected.connect(self._game_report_move_selected)
        dialog.badge_visibility_changed.connect(self._set_classification_badges_visible)
        dialog.closed.connect(lambda source=dialog: self._game_report_dialog_closed(source))
        self._game_report_dialog = dialog
        if report.moves:
            self._game_report_move_selected(report.moves[0].ply)
        dialog.show()

    def _game_analysis_failed(self, reason: str, generation: int) -> None:
        if generation != self._game_analysis_generation:
            return
        self.game_analysis_status.setText(f"Error al analizar la partida: {reason}")

    def _game_analysis_finished(self) -> None:
        worker = self.game_worker
        if worker is not None:
            generation = getattr(worker, "generation", -1)
            worker.deleteLater()
        self.game_worker = None
        self.game_cancel_button.setEnabled(False)
        self.threshold_button.setEnabled(True)
        self.game_analyze_button.setEnabled(
            self._pgn_game is not None and not (self.worker and self.worker.isRunning())
        )
        self.analyze_button.setEnabled(not (self.worker and self.worker.isRunning()))
        if self._selector_pending:
            self._selector_pending=False
            QTimer.singleShot(0,self._analyze_selected_turn)

    def _stop_game_analysis(self) -> None:
        if self.game_worker and self.game_worker.isRunning():
            self.game_worker.stop()
            self.game_cancel_button.setEnabled(False)
            self.game_analysis_status.setText("Cancelando; se guardará el progreso ya completado…")

    def _game_report_move_selected(self, ply: int) -> None:
        report = self._game_analysis_report
        game = self._pgn_game
        if report is None or game is None or report.game is not game or not 1 <= ply <= game.move_count:
            return
        self._navigate_pgn(ply - 1)
        self.pgn_move_label.setText(f"Antes de {game.move_label(ply)}")
        review = report.moves[ply - 1]
        if review.category_id != "unrated":
            self.board_widget.set_review_marker(
                review.move_uci, review.symbol, review.category_id, self._show_classification_badges,
            )
        else:
            self.board_widget.set_review_marker(None)
        board = game.board_at(ply - 1)
        if self._analysis_turn is not None:
            self._selector_analysis_active=True
            self._analyze_selected_turn()
            return
        result = report.position_results.get(board.fen())
        if result and result.lines:
            self._render_lines(result.lines)
            self.message_label.setText(f"Análisis de Stockfish previo a {game.move_label(ply)}.")
        else:
            self.board_widget.set_best_move(None)
            self.best_move_label.setText("Mejor jugada: no disponible")
            self.eval_label.setText("Evaluación: posición no analizada")
            self.eval_bar.setValue(50)
            self.depth_label.setText("Profundidad: —")
            self.variants_view.clear()
            self.message_label.setText("Esta posición no alcanzó a analizarse antes de cancelar.")

    def _set_classification_badges_visible(self, visible: bool) -> None:
        self._show_classification_badges = visible
        dialog = self._game_report_dialog
        row = dialog.table.currentRow() if dialog is not None else -1
        report = self._game_analysis_report
        if report is not None and 0 <= row < len(report.moves):
            self._game_report_move_selected(report.moves[row].ply)

    def _game_report_dialog_closed(self, dialog: GameReportDialog) -> None:
        if self._game_report_dialog is dialog:
            self._game_report_dialog = None

    def _pgn_list_selection_changed(self, row: int) -> None:
        if self._pgn_game is not None and row >= 0:
            self._navigate_pgn(row, update_list=False)

    def _navigate_pgn(self, ply: int, update_list: bool = True) -> None:
        if self._live_monitoring:return
        if self._pgn_game is None:
            return
        ply = max(0, min(ply, self._pgn_game.move_count))
        if (ply == self._pgn_ply and update_list
                and self.board_widget.board.fen() == self._pgn_game.board_at(ply).fen()):
            return
        self._pgn_ply = ply
        if update_list:
            self.pgn_moves.blockSignals(True)
            self.pgn_moves.setCurrentRow(ply)
            self.pgn_moves.blockSignals(False)
        self._show_pgn_position(ply)

    def _show_pgn_position(self, ply: int) -> None:
        assert self._pgn_game is not None
        board = self._pgn_game.board_at(ply)
        self._invalidate_analysis_for_position()
        self.board_widget.set_board(board)
        self.board_widget.set_best_move(None)
        report = self._game_analysis_report
        if report is not None and report.game is self._pgn_game and ply > 0:
            review = report.moves[ply - 1]
            if review.category_id != "unrated":
                self.board_widget.set_review_marker(
                    review.move_uci, review.symbol, review.category_id, self._show_classification_badges,
                    allow_post_move=True,
                )
            else:
                self.board_widget.set_review_marker(None)
        else:
            self.board_widget.set_review_marker(None)
        self.fen_edit.setText(board.fen())
        self.pgn_turn_label.setText(f"Turno: {'Blancas' if board.turn else 'Negras'}")
        self.pgn_move_label.setText("Posición inicial" if ply == 0 else f"Jugada {self._pgn_game.move_label(ply)}")
        self.best_move_label.setText("Mejor jugada: —")
        self.eval_label.setText("Evaluación: analiza esta posición para ver su valor")
        self.eval_bar.setValue(50)
        self.depth_label.setText("Profundidad: —")
        self.variants_view.clear()

    def _invalidate_analysis_for_position(self) -> None:
        if self._selector_analysis_active and not self._live_monitoring:
            self._invalidate_live_game_search('posición de análisis cambiada')
            self._live_displayed_lines=()
        self._analysis_generation += 1
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.engine_status.setText("Deteniendo análisis de posición anterior…")
            self.message_label.setText("La posición cambió; se descartará el análisis anterior.")

    def _start_analysis(self) -> None:
        if self._closed:
            return
        if self.game_worker and self.game_worker.isRunning():
            self.message_label.setText("Espera o cancela primero el análisis completo de la partida.")
            return
        if self.worker and self.worker.isRunning():
            return
        fen = self.fen_edit.text().strip()
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                raise ValueError("La posición FEN no representa una posición válida de ajedrez.")
            engine_path = find_engine(self.engine_path)
        except (ValueError, FileNotFoundError) as exc:
            self.engine_status.setText("No se pudo iniciar el análisis")
            self.engine_status.setStyleSheet("color: #FFAAA0; font-weight: 650")
            self.message_label.setText(str(exc))
            return

        if self._live_monitoring and (self._exploring_live or self._analysis_turn is not None):
            self._request_live_analysis(self.board_widget.board)
            return
        if self._live_monitoring:
            if self._live_tracker is not None:
                try:
                    self._live_tracker.reset(board)
                except ValueError as exc:
                    self.message_label.setText(str(exc))
                    return
            self._apply_live_position(board, "posición revisada manualmente.")
            return

        if self._analysis_turn is not None or self._selector_analysis_active:
            self.board_widget.set_board(board)
            self.fen_edit.setText(board.fen())
            self._selector_analysis_active=True
            self._request_live_analysis(board)
            return

        depth = self._position_depth()

        self.board_widget.set_position(board.fen())
        self.board_widget.set_best_move(None)
        self.best_move_label.setText("Mejor jugada: calculando…")
        self.eval_label.setText("Evaluación: calculando…")
        self.eval_bar.setValue(50)
        self.depth_label.setText("Profundidad: —")
        self.variants_view.clear()
        self.message_label.setText("Stockfish está analizando la posición.")
        self.engine_status.setText("Conectando con Stockfish…")
        self.engine_status.setStyleSheet("")
        self._analysis_generation += 1
        generation = self._analysis_generation
        self.worker = AnalysisWorker(
            board.fen(), engine_path, self.time_spin.value(), depth, self.multipv_spin.value()
        )
        self.worker.generation = generation
        self.worker.progress.connect(lambda lines, token=generation: self._show_progress(lines, token))
        self.worker.completed.connect(lambda result, cancelled, token=generation: self._analysis_completed(result, cancelled, token))
        self.worker.failed.connect(lambda reason, token=generation: self._analysis_failed(reason, token))
        self.worker.finished.connect(self._worker_finished)
        self._set_running(True)
        self.worker.start()

    def _stop_analysis(self) -> None:
        if (self._live_monitoring or self._selector_analysis_active) and self.live_analysis_worker and self.live_analysis_worker.isRunning():
            self.live_analysis_worker.cancel_pending()
            self._live_analysis_request_id += 1
            self._live_search_ticket=None
            self._live_search_finalized=True
            self.stop_button.setEnabled(False)
            self.engine_status.setText("Búsqueda detenida; sesión preparada para la siguiente posición")
            self.message_label.setText("Se detuvo el análisis actual; la lectura del tablero continúa.")
            return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.engine_status.setText("Deteniendo análisis…")
            self.message_label.setText("Se solicitó detener la búsqueda actual.")

    def _set_running(self, running: bool) -> None:
        game_running = bool(self.game_worker and self.game_worker.isRunning())
        self.analyze_button.setEnabled(not running and not game_running)
        self.stop_button.setEnabled(running)
        self.browse_engine_button.setEnabled(not running and not game_running)
        self.game_analyze_button.setEnabled(
            not running and self._pgn_game is not None
            and not (self.game_worker and self.game_worker.isRunning())
        )

    def _show_progress(self, lines: tuple[AnalysisLine, ...], generation: int | None = None) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        self.engine_status.setText("Stockfish conectado · analizando")
        self._render_lines(lines)

    def _render_lines(self, lines: tuple[AnalysisLine, ...]) -> None:
        if not lines:
            return
        logger.debug("Representando variantes de la posición actual: %s",
                     [(line.index, line.best_move_uci, line.depth) for line in lines])
        best = lines[0]
        self.best_move_label.setText(f"Mejor jugada: {best.best_move_san}")
        self.eval_label.setText(f"Evaluación: {best.evaluation_text}")
        self.eval_bar.set_score(best.score_cp,best.mate)
        engine_time = f"{best.time_seconds:.2f} s" if best.time_seconds is not None else "—"
        self.depth_label.setText(
            f"Profundidad: {best.depth if best.depth is not None else '—'} · Búsqueda Stockfish: {engine_time}"
        )
        self.board_widget.analysis_turn = self._analysis_turn
        self.board_widget.set_best_moves(line.best_move_uci for line in lines[:5])
        rendered = []
        for line in lines:
            score = line.evaluation_text
            pv = " ".join(line.pv_san) or "sin variante"
            rendered.append(f"{line.index}. {score}\n   {pv}")
        self.variants_view.setPlainText("\n\n".join(rendered))
        if self.overlay_tab.timer.isActive():
            self.overlay_tab.sync()

    def _analysis_completed(self, result: AnalysisResult, cancelled: bool, generation: int | None = None) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if result.lines:
            self._render_lines(result.lines)
        else:
            self.best_move_label.setText("Mejor jugada: sin jugadas legales")
            self.eval_label.setText("Evaluación: posición terminal")
            self.eval_bar.setValue(50)
            self.depth_label.setText("Profundidad: —")
            self.variants_view.setPlainText("No hay variantes disponibles en esta posición.")
        if cancelled:
            self.engine_status.setText("Análisis detenido")
            self.message_label.setText("La búsqueda se detuvo correctamente.")
        else:
            self.engine_status.setText("Stockfish conectado · análisis finalizado")
            self.message_label.setText(f"Análisis terminado con {len(result.lines)} variante(s).")

    def _analysis_failed(self, reason: str, generation: int | None = None) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        self.engine_status.setText("Error del motor")
        self.engine_status.setStyleSheet("color: #FFAAA0; font-weight: 650")
        self.message_label.setText(f"No se pudo analizar la posición: {reason}")

    def _worker_finished(self) -> None:
        self._set_running(False)
        worker = self.worker
        self.worker = None
        if worker:
            if getattr(worker, "generation", self._analysis_generation) != self._analysis_generation:
                self.engine_status.setText("Análisis anterior detenido")
                self.message_label.setText("El resultado de la posición anterior se descartó.")
            worker.deleteLater()
        if self._selector_pending:
            self._selector_pending=False
            QTimer.singleShot(0,self._analyze_selected_turn)
        elif self._live_pending_analysis and self._live_monitoring and not self._live_paused:
            self._live_pending_analysis = False
            QTimer.singleShot(0, self._start_analysis)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        self.overlay_tab.shutdown()
        self.four_player_tab.shutdown()
        self._closed = True
        self._clear_premove()
        if self._vulnerability_worker is not None:
            self._vulnerability_worker.stop()
            self._vulnerability_worker.wait(3000)
        if self._pin_worker is not None:
            self._pin_worker.stop()
            self._pin_worker.wait(3000)
        self._live_monitoring=False
        self._live_generation+=1
        self._live_search_ticket=None
        self._live_pending_analysis=False
        self._live_timer.stop()
        if self._capture_selector:
            self._dispose_selector()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.game_worker and self.game_worker.isRunning():
            self.game_worker.stop()
            self.game_worker.wait(3000)
        if self.recognition_worker and self.recognition_worker.isRunning():
            self.recognition_worker.wait(3000)
        if self._live_region_worker and self._live_region_worker.isRunning():
            self._live_region_worker.requestInterruption()
            self._live_region_worker.wait(3000)
        if self.live_analysis_worker and self.live_analysis_worker.isRunning():
            self.live_analysis_worker.stop()
            self.live_analysis_worker.wait(3000)
        event.accept()
