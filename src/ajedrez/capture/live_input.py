"""Read-only helpers for transient input during selected-screen observation."""
from __future__ import annotations

import sys


def left_mouse_button_down() -> bool:
    """Return whether Windows currently reports the left mouse button held."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
    except (AttributeError, OSError):
        return False
