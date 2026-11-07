"""Reusable Stockfish integration and UCI analysis helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable

import chess
import chess.engine
from .paths import resource_root


@dataclass(frozen=True)
class AnalysisLine:
    """One principal variation, with score always expressed from White's view."""

    index: int
    best_move_uci: str | None
    best_move_san: str
    score_cp: int | None
    mate: int | None
    depth: int | None
    time_seconds: float | None
    pv_san: tuple[str, ...]

    @property
    def evaluation_text(self) -> str:
        if self.mate is not None:
            if self.mate > 0:
                return f"Mate en {self.mate} para blancas (perspectiva de blancas)"
            if self.mate < 0:
                return f"Mate en {abs(self.mate)} para negras (perspectiva de blancas)"
            return "Mate en 0 (perspectiva de blancas)"
        if self.score_cp is None:
            return "Stockfish no proporcionó una evaluación"
        pawns = self.score_cp / 100
        if self.score_cp > 0:
            advantage = "ventaja blanca"
        elif self.score_cp < 0:
            advantage = "ventaja negra"
        else:
            advantage = "posición igualada"
        return f"{pawns:+.2f} peones · {advantage} (desde las blancas)"


@dataclass(frozen=True)
class AnalysisResult:
    fen: str
    engine_path: str
    lines: tuple[AnalysisLine, ...]


def find_engine(explicit: str | None = None) -> str:
    """Resolve Stockfish from a selected path, environment, project folder, or PATH."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No existe el ejecutable de Stockfish indicado: {path}")
        return str(path.resolve())

    root = resource_root()
    candidates = [os.environ.get("STOCKFISH_EXECUTABLE")]
    candidates.extend(str(path) for path in (root / "engines").rglob("stockfish*.exe"))
    candidates.append(shutil.which("stockfish"))
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser().resolve())
    raise FileNotFoundError(
        "No se encontró Stockfish. Selecciona el archivo .exe del motor oficial "
        "o define STOCKFISH_EXECUTABLE."
    )


def _convert_lines(board: chess.Board, infos: list[chess.engine.InfoDict]) -> tuple[AnalysisLine, ...]:
    lines: list[AnalysisLine] = []
    for index, info in enumerate(infos, 1):
        pv = info.get("pv", [])
        score = info.get("score")
        score_cp = score.white().score() if score is not None else None
        mate = score.white().mate() if score is not None else None
        line_board = board.copy()
        san_moves: list[str] = []
        for move in pv:
            if move not in line_board.legal_moves:
                break
            san_moves.append(line_board.san(move))
            line_board.push(move)
        lines.append(AnalysisLine(
            index=index,
            best_move_uci=pv[0].uci() if pv else None,
            best_move_san=san_moves[0] if san_moves else "sin jugada",
            score_cp=score_cp,
            mate=mate,
            depth=info.get("depth"),
            time_seconds=info.get("time"),
            pv_san=tuple(san_moves),
        ))
    return tuple(lines)


def analyze_position(
    fen: str,
    *,
    engine_path: str | None = None,
    seconds: float = 1.5,
    depth: int | None = None,
    multipv: int = 3,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[tuple[AnalysisLine, ...]], None] | None = None,
    analysis_handle_callback: Callable[[chess.engine.SimpleAnalysisResult], None] | None = None,
) -> AnalysisResult:
    """Analyze a FEN; optional callbacks support GUI progress and cooperative stop."""
    if seconds <= 0 or (depth is not None and depth < 1) or not 1 <= multipv <= 5:
        raise ValueError("El tiempo y la profundidad deben ser positivos; las variantes deben estar entre 1 y 5.")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("La posición FEN no representa una posición válida de ajedrez.")
    resolved_engine = find_engine(engine_path)
    return _analyze_with_uci_process(
        board, resolved_engine, seconds, depth, multipv, cancel_event,
        progress_callback, analysis_handle_callback,
    )


class _UCIStopHandle:
    """Thread-safe stop handle passed back to the GUI worker."""

    def __init__(self, process: subprocess.Popen, write_lock: threading.Lock):
        self.process = process
        self.write_lock = write_lock

    def stop(self) -> None:
        if self.process.poll() is None and self.process.stdin:
            try:
                with self.write_lock:
                    self.process.stdin.write("stop\n")
                    self.process.stdin.flush()
            except (OSError, ValueError):
                pass


class UCIEngineSession:
    """One persistent Stockfish process that can analyze many FEN positions."""

    def __init__(self, engine_path: str, *, multipv: int = 1, threads: int = 1, hash_mb: int = 16):
        if not 1 <= multipv <= 5 or threads < 1 or hash_mb < 1:
            raise ValueError("MultiPV, hilos y memoria deben tener valores positivos válidos.")
        self.engine_path = str(Path(engine_path).resolve())
        self.multipv = multipv
        self.threads = threads
        self.hash_mb = hash_mb
        self.process: subprocess.Popen | None = None
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.write_lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                [self.engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=creationflags,
            )
        except OSError as exc:
            raise RuntimeError(f"No se pudo iniciar Stockfish: {exc}") from exc

        def read_output() -> None:
            assert self.process is not None and self.process.stdout is not None
            try:
                for output_line in self.process.stdout:
                    self.messages.put(output_line.rstrip("\r\n"))
            finally:
                self.messages.put(None)

        self.reader = threading.Thread(target=read_output, name="stockfish-uci-reader", daemon=True)
        self.reader.start()
        try:
            self._send("uci")
            self._wait_token("uciok", 15)
            # Official Stockfish supports these standard UCI options. Keeping
            # the session settings fixed makes scores comparable across FENs.
            self._send(f"setoption name Threads value {self.threads}")
            self._send(f"setoption name Hash value {self.hash_mb}")
            self._send(f"setoption name MultiPV value {self.multipv}")
            self._send("isready")
            self._wait_token("readyok", 15)
        except BaseException:
            self.close()
            raise

    def _send(self, command: str) -> None:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("Stockfish se cerró durante la comunicación UCI.")
        with self.write_lock:
            process.stdin.write(command + "\n")
            process.stdin.flush()

    def _wait_token(self, expected: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = self.messages.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("Stockfish se cerró durante el inicio UCI.")
                continue
            if response is None:
                raise RuntimeError("Stockfish cerró su salida antes de responder UCI.")
            if response.strip() == expected:
                return
        raise TimeoutError(f"Stockfish no respondió con {expected} dentro del tiempo esperado.")

    def analyze(
        self,
        fen: str,
        *,
        seconds: float = 1.5,
        depth: int | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[tuple[AnalysisLine, ...]], None] | None = None,
        analysis_handle_callback: Callable[[object], None] | None = None,
    ) -> AnalysisResult:
        if self._closed:
            raise RuntimeError("La sesión de Stockfish ya está cerrada.")
        if seconds <= 0 or (depth is not None and depth < 1):
            raise ValueError("El tiempo y la profundidad deben ser positivos.")
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("La posición FEN no representa una posición válida de ajedrez.")
        self._send("position fen " + board.fen())
        go_command = f"go depth {depth}" if depth is not None else f"go movetime {max(1, round(seconds * 1000))}"
        self._send(go_command)
        process = self.process
        assert process is not None
        stop_handle = _UCIStopHandle(process, self.write_lock)
        if analysis_handle_callback:
            analysis_handle_callback(stop_handle)

        lines_by_index: dict[int, AnalysisLine] = {}
        latest_lines: tuple[AnalysisLine, ...] = ()
        complete_lines: tuple[AnalysisLine, ...] = ()
        expected_lines = min(self.multipv, board.legal_moves.count())
        stopped = False
        deadline = time.monotonic() + (max(60, seconds + 15) if depth is None else 300)
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set() and not stopped:
                stop_handle.stop()
                stopped = True
            try:
                response = self.messages.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError("Stockfish terminó inesperadamente durante el análisis.")
                continue
            if response is None:
                raise RuntimeError("Stockfish cerró su salida antes de terminar el análisis.")
            parsed = _parse_uci_info(board, response)
            if parsed:
                index, info_line = parsed
                # Stockfish also emits bookkeeping `info` records without a
                # completed PV. Do not let those erase a useful MultiPV line.
                if (info_line.depth is None or info_line.best_move_uci is None
                        or (info_line.score_cp is None and info_line.mate is None)):
                    continue
                # A final UCI flush can repeat the same depth with new rankings.
                # PV 1 starts a fresh batch, even at an unchanged depth; never
                # combine its new move with old PV 2..5 of that same depth.
                if index == 1:
                    if 1 in lines_by_index and info_line.depth < lines_by_index[1].depth:
                        continue
                    lines_by_index = {}
                elif (1 not in lines_by_index
                      or lines_by_index[1].depth != info_line.depth):
                    continue
                if not 1 <= index <= expected_lines:
                    continue
                if info_line.best_move_uci in {line.best_move_uci for line in lines_by_index.values()}:
                    continue
                lines_by_index[index] = info_line
                # Out-of-order PV 3 may arrive before PV 2. Keep it in this
                # batch, but publish only the contiguous prefix until PV 2 arrives.
                prefix=[]
                for key in range(1,expected_lines+1):
                    if key not in lines_by_index:
                        break
                    prefix.append(lines_by_index[key])
                latest_lines = tuple(prefix)
                if len(latest_lines) == expected_lines:
                    complete_lines = latest_lines
                if progress_callback and latest_lines:
                    progress_callback(latest_lines)
            if response.startswith("bestmove "):
                # Prefer the most recent complete iteration: at the time limit
                # Stockfish may have sent a newer PV 1/2 but not PV 3 yet.
                if complete_lines:
                    latest_lines = complete_lines
                elif not latest_lines and lines_by_index:
                    current_depth = max((line.depth or 0) for line in lines_by_index.values())
                    latest_lines = tuple(
                        lines_by_index[key] for key in sorted(lines_by_index)
                        if (lines_by_index[key].depth or 0) == current_depth
                    )
                break
        else:
            stop_handle.stop()
            raise TimeoutError("Stockfish excedió el tiempo máximo de análisis.")
        if progress_callback and latest_lines:
            progress_callback(latest_lines)
        return AnalysisResult(fen=board.fen(), engine_path=self.engine_path, lines=latest_lines)

    def new_game(self) -> None:
        """Reset game-specific engine state between searches, reusing this UCI process."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_token("readyok",15)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is not None and process.poll() is None:
            try:
                self._send("quit")
                process.wait(timeout=2)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        if self.reader:
            self.reader.join(timeout=2)
        if process is not None:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()

    def __enter__(self) -> "UCIEngineSession":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _parse_uci_info(board: chess.Board, line: str) -> tuple[int, AnalysisLine] | None:
    """Convert one UCI info line into a White-perspective SAN principal variation."""
    tokens = line.split()
    if not tokens or tokens[0] != "info":
        return None
    try:
        depth = int(tokens[tokens.index("depth") + 1]) if "depth" in tokens else None
        multipv = int(tokens[tokens.index("multipv") + 1]) if "multipv" in tokens else 1
        time_ms = int(tokens[tokens.index("time") + 1]) if "time" in tokens else None
        score_cp = score_mate = None
        if "score" in tokens:
            score_index = tokens.index("score")
            score_type, value = tokens[score_index + 1], int(tokens[score_index + 2])
            white_sign = 1 if board.turn == chess.WHITE else -1
            if score_type == "cp":
                score_cp = value * white_sign
            elif score_type == "mate":
                score_mate = value * white_sign
        pv_tokens = tokens[tokens.index("pv") + 1:] if "pv" in tokens else []
    except (ValueError, IndexError):
        return None

    line_board = board.copy(stack=False)
    san: list[str] = []
    pv_uci: list[str] = []
    for token in pv_tokens:
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            break
        if move not in line_board.legal_moves:
            break
        pv_uci.append(move.uci())
        san.append(line_board.san(move))
        line_board.push(move)
    best_uci = pv_uci[0] if pv_uci else None
    best_san = san[0] if san else "sin jugada"
    return multipv, AnalysisLine(
        index=multipv, best_move_uci=best_uci, best_move_san=best_san,
        score_cp=score_cp, mate=score_mate, depth=depth,
        time_seconds=time_ms / 1000 if time_ms is not None else None,
        pv_san=tuple(san),
    )


def _analyze_with_uci_process(
    board: chess.Board,
    engine_path: str,
    seconds: float,
    depth: int | None,
    multipv: int,
    cancel_event: threading.Event | None,
    progress_callback: Callable[[tuple[AnalysisLine, ...]], None] | None,
    analysis_handle_callback: Callable[[object], None] | None,
) -> AnalysisResult:
    """Analyze one FEN with a short-lived session (the single-position path)."""
    with UCIEngineSession(engine_path, multipv=multipv) as session:
        return session.analyze(
            board.fen(), seconds=seconds, depth=depth, cancel_event=cancel_event,
            progress_callback=progress_callback,
            analysis_handle_callback=analysis_handle_callback,
        )


def format_result(result: AnalysisResult) -> str:
    """Format structured results for the command-line prototype."""
    output: list[str] = []
    for line in result.lines:
        pv = " ".join(line.pv_san)
        output.append(
            f"{line.index}. Mejor jugada: {line.best_move_san} | Evaluación: {line.evaluation_text} | "
            f"Profundidad: {line.depth if line.depth is not None else '?'} | Variante: {pv}"
        )
    return "\n".join(output)
