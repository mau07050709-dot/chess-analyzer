"""Background search for a relocated board in bounded window screenshots."""
import logging
import time
from PySide6.QtCore import QThread, Signal, QRect
from ..capture.board_region import locate_board, discover_board, background_colors, grid_score, LocatedRegion

logger=logging.getLogger(__name__)

class RegionSearchWorker(QThread):
    completed=Signal(object)

    def __init__(self, searches, colors, parent=None):
        super().__init__(parent)
        self.searches=searches; self.colors=colors

    def run(self):
        started=time.perf_counter();matches=[]
        try:
            for index, origin, image, excluded in self.searches:
                if self.isInterruptionRequested():return
                local=locate_board(image,self.colors,excluded)
                if local is None:local=discover_board(image,excluded)
                if local is not None:
                    crop=image.copy(local);colors=background_colors(crop)
                    matches.append(LocatedRegion(index,local.translated(origin),grid_score(crop,colors),colors))
            # Multiple source monitors with boards are ambiguous: retry rather
            # than silently switching the user's chosen board.
            self.completed.emit(matches[0] if len(matches)==1 else None)
        except Exception:
            logger.exception("No se pudo reubicar la región; se conserva el tablero confirmado")
            self.completed.emit(None)
        finally:
            logger.info("Búsqueda de región: %.1f ms; candidatos=%s",(time.perf_counter()-started)*1000,len(matches))
