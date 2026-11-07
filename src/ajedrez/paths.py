"""Read-only bundled resources and per-user writable application data."""
from pathlib import Path
import os
import sys


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def user_data_dir() -> Path:
    directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "ChessAnalyzerPublic"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
