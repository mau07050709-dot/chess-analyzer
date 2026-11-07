"""Read-only Windows window anchoring in physical pixels, with Qt DPI mapping."""
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
import sys
from PySide6.QtCore import QRect


def physical_bounds(screen):
    g=screen.geometry(); ratio=screen.devicePixelRatio()
    # Qt scales screen sizes, but Windows screen origins stay in native pixels.
    return QRect(g.x(),g.y(),round(g.width()*ratio),round(g.height()*ratio))


def to_physical(rect,screen):
    g=screen.geometry();r=screen.devicePixelRatio()
    return QRect(g.x()+round((rect.x()-g.x())*r),g.y()+round((rect.y()-g.y())*r),
                 round(rect.width()*r),round(rect.height()*r))


def to_logical(rect,screen):
    g=screen.geometry();r=screen.devicePixelRatio()
    return QRect(g.x()+round((rect.x()-g.x())/r),g.y()+round((rect.y()-g.y())/r),
                 round(rect.width()/r),round(rect.height()/r))


@dataclass(frozen=True)
class WindowGeometry:
    client: QRect
    dpi: int
    pid: int


class WindowAnchor:
    def __init__(self,handle,geometry,region):
        self.handle=handle;self.initial=geometry;self.region=QRect(region)

    @staticmethod
    def _api():
        api=ctypes.WinDLL('user32',use_last_error=True)
        api.GetClientRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
        api.ClientToScreen.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.POINT)]
        api.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
        api.IsWindow.argtypes=api.IsWindowVisible.argtypes=api.IsIconic.argtypes=[wintypes.HWND]
        api.GetDpiForWindow.argtypes=[wintypes.HWND]
        api.WindowFromPoint.argtypes=[wintypes.POINT];api.WindowFromPoint.restype=wintypes.HWND
        api.GetAncestor.argtypes=[wintypes.HWND,wintypes.UINT];api.GetAncestor.restype=wintypes.HWND
        api.GetWindowLongPtrW.argtypes=[wintypes.HWND,ctypes.c_int];api.GetWindowLongPtrW.restype=ctypes.c_ssize_t
        return api

    @classmethod
    def geometry(cls,handle):
        if sys.platform!='win32':return None
        api=cls._api()
        if not api.IsWindow(handle) or not api.IsWindowVisible(handle) or api.IsIconic(handle):return None
        rect=wintypes.RECT();origin=wintypes.POINT(0,0);pid=wintypes.DWORD()
        if not api.GetClientRect(handle,ctypes.byref(rect)) or not api.ClientToScreen(handle,ctypes.byref(origin)):return None
        api.GetWindowThreadProcessId(handle,ctypes.byref(pid))
        return WindowGeometry(QRect(origin.x,origin.y,rect.right,rect.bottom),api.GetDpiForWindow(handle) or 96,pid.value)

    @classmethod
    def bind(cls,region,screen):
        if sys.platform!='win32':return None
        target=to_physical(region,screen);matches=[];api=cls._api()
        # Prefer Windows' actual hit-test. Visible transparent overlays may span
        # the entire desktop without being the source the user selected.
        center=target.center()
        hit=api.WindowFromPoint(wintypes.POINT(center.x(),center.y()))
        root=api.GetAncestor(hit,2) if hit else None
        state=cls.geometry(root) if root else None
        if state and state.pid!=os.getpid() and state.client.contains(target):
            return cls(root,state,target)
        callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
        @callback_type
        def visit(handle,_):
            if api.GetWindowLongPtrW(handle,-20) & 0x20:
                return True  # WS_EX_TRANSPARENT: not the selected visible source.
            state=cls.geometry(handle)
            if state and state.pid!=os.getpid() and state.client.contains(target):
                matches.append(cls(handle,state,target));return False
            return True
        api.EnumWindows.argtypes=[callback_type,wintypes.LPARAM]
        api.EnumWindows(visit,0)
        return matches[0] if matches else None

    def current(self):
        state=self.geometry(self.handle)
        return state if state and state.pid==self.initial.pid else None

    def predicted(self,state):
        scale=state.dpi/self.initial.dpi
        return QRect(state.client.x()+round((self.region.x()-self.initial.client.x())*scale),
                     state.client.y()+round((self.region.y()-self.initial.client.y())*scale),
                     round(self.region.width()*scale),round(self.region.height()*scale))

    def update(self,state,region):
        self.initial=state;self.region=QRect(region)
