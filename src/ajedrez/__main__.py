"""Command-line entry point; chess and engine logic lives in ajedrez.engine."""
from __future__ import annotations

import argparse
import sys

import chess
import chess.engine

from .engine import analyze_position, find_engine, format_result

START_FEN = chess.STARTING_FEN


def analyze(fen: str, engine_path: str, seconds: float, depth: int | None, multipv: int) -> str:
    """Keep the original CLI helper while delegating all UCI work to engine.py."""
    return format_result(analyze_position(
        fen, engine_path=engine_path, seconds=seconds, depth=depth, multipv=multipv
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analiza una posición local con Stockfish.")
    parser.add_argument("--fen", default=START_FEN, help="Posición FEN (por defecto, posición inicial)")
    parser.add_argument("--engine", help="Ruta al ejecutable Stockfish")
    parser.add_argument("--time", type=float, default=1.5, help="Segundos de análisis, salvo que se indique --depth")
    parser.add_argument("--depth", type=int, help="Profundidad fija de búsqueda")
    parser.add_argument("--multipv", type=int, default=3, help="Número de variantes principales (1-5)")
    args = parser.parse_args(argv)
    if args.time <= 0 or (args.depth is not None and args.depth < 1) or not 1 <= args.multipv <= 5:
        parser.error("--time y --depth deben ser positivos; --multipv debe estar entre 1 y 5")
    try:
        engine_path = find_engine(args.engine)
        print(f"Stockfish: {engine_path}")
        print(f"Posición: {args.fen}")
        print(analyze(args.fen, engine_path, args.time, args.depth, args.multipv))
        return 0
    except (FileNotFoundError, ValueError, chess.InvalidMoveError, chess.engine.EngineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())



