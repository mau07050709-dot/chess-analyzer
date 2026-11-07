"""Screen snapshot and region-cropping helpers backed by Qt's QScreen API."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, QRectF
from PySide6.QtGui import QImage


MIN_REGION_SIZE = 32


@dataclass(frozen=True)
class ScreenFrame:
    """One local monitor snapshot and its logical desktop geometry."""

    screen_name: str
    geometry: QRect
    image: QImage
    device_pixel_ratio: float


def capture_screen(screen) -> ScreenFrame:
    """Grab one monitor. The caller should hide/minimize its own window first."""
    geometry = QRect(screen.geometry())
    pixmap = screen.grabWindow(0)
    if pixmap.isNull():
        raise RuntimeError("Windows no devolvió una imagen de esa pantalla.")
    ratio = float(pixmap.devicePixelRatio() or 1.0)
    image = pixmap.toImage().copy()
    if image.isNull():
        raise RuntimeError("No se pudo leer la imagen capturada de la pantalla.")
    return ScreenFrame(screen.name() or "Pantalla", geometry, image, ratio)


def clamp_region(region: QRect, bounds: QRect, minimum_size: int = MIN_REGION_SIZE) -> QRect:
    """Clip a global logical-pixel rectangle to a screen and validate its size."""
    clipped = QRect(region).normalized().intersected(QRect(bounds))
    if clipped.isEmpty() or clipped.width() < minimum_size or clipped.height() < minimum_size:
        raise ValueError(f"Selecciona una región de al menos {minimum_size} × {minimum_size} píxeles.")
    return clipped


def crop_region(
    frame: ScreenFrame,
    region: QRect,
    minimum_size: int = MIN_REGION_SIZE,
) -> QImage:
    """Crop a global logical-pixel selection from a monitor's physical-pixel image."""
    clipped = clamp_region(region, frame.geometry, minimum_size)
    local_x = clipped.x() - frame.geometry.x()
    local_y = clipped.y() - frame.geometry.y()
    ratio = frame.device_pixel_ratio
    pixel_rect = QRectF(
        local_x * ratio,
        local_y * ratio,
        clipped.width() * ratio,
        clipped.height() * ratio,
    ).toAlignedRect().intersected(frame.image.rect())
    if pixel_rect.isEmpty():
        raise ValueError("La región seleccionada quedó fuera de la imagen de pantalla.")
    cropped = frame.image.copy(pixel_rect)
    cropped.setDevicePixelRatio(1.0)
    if cropped.isNull():
        raise RuntimeError("No se pudo recortar la región seleccionada.")
    return cropped
