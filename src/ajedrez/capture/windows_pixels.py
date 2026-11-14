"""GDI desktop snapshots without CAPTUREBLT or cursor manipulation."""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
from functools import lru_cache

from PySide6.QtGui import QImage


SRCCOPY = 0x00CC0020


class BitmapInfoHeader(ct.Structure):
    _fields_ = [
        ("size", wt.DWORD), ("width", wt.LONG), ("height", wt.LONG),
        ("planes", wt.WORD), ("bits", wt.WORD), ("compression", wt.DWORD),
        ("image_size", wt.DWORD), ("x_resolution", wt.LONG), ("y_resolution", wt.LONG),
        ("colors_used", wt.DWORD), ("important_colors", wt.DWORD),
    ]


@lru_cache(maxsize=1)
def _native_api():
    user, gdi = (ct.WinDLL(name, use_last_error=True) for name in ("user32", "gdi32"))
    h = wt.HANDLE
    i = ct.c_int
    ptr = ct.c_void_p
    for dll, name, result, args in (
        (user, "GetDC", h, (h,)),
        (user, "ReleaseDC", i, (h, h)),
        (gdi, "CreateCompatibleDC", h, (h,)),
        (gdi, "DeleteDC", wt.BOOL, (h,)),
        (gdi, "CreateDIBSection", h, (h, ct.POINTER(BitmapInfoHeader), wt.UINT, ct.POINTER(ptr), h, wt.DWORD)),
        (gdi, "SelectObject", h, (h, h)),
        (gdi, "DeleteObject", wt.BOOL, (h,)),
        (gdi, "BitBlt", wt.BOOL, (h, i, i, i, i, h, i, i, wt.DWORD)),
        (gdi, "GdiFlush", wt.BOOL, ()),
    ):
        function = getattr(dll, name)
        function.restype = result
        function.argtypes = args
    return user, gdi


def grab_desktop_pixels(rect) -> QImage:
    """Return an owned, top-down QImage for the requested physical rectangle."""
    width, height = rect.width(), rect.height()
    if width <= 0 or height <= 0:
        raise OSError("El monitor no tiene una región de captura válida.")
    user, gdi = _native_api()
    desktop = user.GetDC(None)
    if not desktop:
        raise ct.WinError(ct.get_last_error())
    memory_dc = bitmap = previous = None
    try:
        memory_dc = gdi.CreateCompatibleDC(desktop)
        if not memory_dc:
            raise ct.WinError(ct.get_last_error())
        header = BitmapInfoHeader(ct.sizeof(BitmapInfoHeader), width, -height, 1, 32)
        pixels = ct.c_void_p()
        bitmap = gdi.CreateDIBSection(desktop, ct.byref(header), 0, ct.byref(pixels), None, 0)
        if not bitmap or not pixels.value:
            raise ct.WinError(ct.get_last_error())
        previous = gdi.SelectObject(memory_dc, bitmap)
        if not previous:
            raise ct.WinError(ct.get_last_error())
        if not gdi.BitBlt(memory_dc, 0, 0, width, height, desktop, rect.x(), rect.y(), SRCCOPY):
            raise ct.WinError(ct.get_last_error())
        if not gdi.GdiFlush():
            raise ct.WinError(ct.get_last_error())
        # BI_RGB supplies BGRX on Windows. Copy while the DIB still owns its bytes.
        buffer = (ct.c_ubyte * (width * height * 4)).from_address(pixels.value)
        return QImage(memoryview(buffer), width, height, width * 4, QImage.Format.Format_RGB32).copy()
    finally:
        if previous:
            gdi.SelectObject(memory_dc, previous)
        if memory_dc:
            gdi.DeleteDC(memory_dc)
        if bitmap:
            gdi.DeleteObject(bitmap)
        user.ReleaseDC(None, desktop)
