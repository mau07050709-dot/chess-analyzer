"""Non-interactive artwork confined to unused space around the chessboard."""
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout
from ..paths import resource_root


class DecorativeImage(QWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName('Imagen decorativa')
        self.original=QPixmap(str(resource_root()/'assets'/'decorations'/filename))
        self.scaled=QPixmap()

    def sizeHint(self):
        return QSize(0,0)

    def resizeEvent(self,event):
        if not self.original.isNull() and self.width()>0 and self.height()>0:
            self.scaled=self.original.scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
        super().resizeEvent(event)

    def paintEvent(self,event):
        if not self.scaled.isNull():
            painter=QPainter(self)
            painter.drawPixmap((self.width()-self.scaled.width())//2,
                               (self.height()-self.scaled.height())//2,self.scaled)


class RightDecorations(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0);layout.setSpacing(16)
        self.winry=DecorativeImage('winry.png',self)
        self.orihime=DecorativeImage('orihime.png',self)
        layout.addWidget(self.winry)
        layout.addWidget(self.orihime,1)

    def resizeEvent(self,event):
        self.winry.setFixedHeight(min(136,max(64,int(self.height()*.24))))
        super().resizeEvent(event)
