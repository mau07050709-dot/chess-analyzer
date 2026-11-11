"""Presentation-only analysis position and board-side evaluation controls."""
import math
import chess
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QFrame,QLabel,QVBoxLayout,QBoxLayout


POSITION_PROFILES = (
    ("Límite de tiempo", "time", None),
    ("Rápido · profundidad 8", "fast", 8),
    ("Equilibrado · profundidad 12", "balanced", 12),
    ("Profundo · profundidad 16", "deep", 16),
    ("Profundidad personalizada", "custom", None),
)
PROFILE_DEPTHS = {key: depth for _, key, depth in POSITION_PROFILES}


def analysis_position(board: chess.Board, turn: bool | None) -> chess.Board:
    result=board.copy(stack=False)
    if turn is not None and turn != result.turn:
        result.turn=turn
        # An en-passant target belongs only to the real next player. The
        # confirmed board keeps it; a hypothetical opposite turn cannot use it.
        result.ep_square=None
    return result


class EvaluationBar(QWidget):
    BLACK_COLOR='#20252a'
    WHITE_COLOR='#f7f7f4'
    BORDER_COLOR='#89938d'

    def __init__(self,parent=None):
        super().__init__(parent)
        self._value=50;self.orientation='white';self._score_text='—'
        self.setFixedWidth(34)
        self._layout=QVBoxLayout(self)
        self._layout.setContentsMargins(1,1,1,1);self._layout.setSpacing(0)
        self.black_segment=QFrame(self);self.white_segment=QFrame(self)
        self.black_segment.setStyleSheet(f'background-color: {self.BLACK_COLOR};')
        self.white_segment.setStyleSheet(f'background-color: {self.WHITE_COLOR};')
        self._layout.addWidget(self.black_segment);self._layout.addWidget(self.white_segment)
        self.setStyleSheet(f'EvaluationBar {{ border: 1px solid {self.BORDER_COLOR}; border-radius: 4px; }}')
        self.score_label=QLabel(self);self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAccessibleName('Barra vertical de evaluación desde las blancas')
        self.setValue(50)

    def setRange(self,*_args):pass

    def setValue(self,value):
        self._set_segments(value)
        self._score_text='—';self._update_label()

    def _set_segments(self,value):
        self._value=max(0,min(100,int(value)))
        black,white=self.segment_fractions()
        self._layout.setStretch(0,round(black*1000));self._layout.setStretch(1,round(white*1000))
    def value(self):return self._value

    def segment_fractions(self):
        white=self._value/100
        return 1-white,white

    def set_score(self,cp=None,mate=None):
        if mate is not None:
            self._set_segments(100 if mate>0 else 0 if mate<0 else 50)
            self._score_text=f'{"+" if mate>0 else "−" if mate<0 else ""}M{abs(mate)}'
        elif cp is not None:
            self._set_segments(round(50+50*math.tanh(cp/400)))
            self._score_text=f'{cp/100:+.1f}' if cp else '0.0'
        else:self.setValue(50)
        self._update_label()

    def set_terminal(self,winner):
        self._set_segments(100 if winner else 0)
        self._score_text='+M0' if winner else '−M0';self._update_label()

    def set_orientation(self,orientation):
        self.orientation=orientation
        self._layout.setDirection(QBoxLayout.Direction.TopToBottom if orientation=='white'
                                  else QBoxLayout.Direction.BottomToTop)
        self.setToolTip('Blanco: ventaja blanca · negro: ventaja negra · evaluación desde las blancas')
        self._update_label()

    def _update_label(self):
        white=self._value>=50
        top=white==(self.orientation=='black')
        self.score_label.setGeometry(1,3 if top else max(3,self.height()-28),max(1,self.width()-2),24)
        self.score_label.setStyleSheet(f'color: {self.BLACK_COLOR if white else self.WHITE_COLOR}; background: transparent; font-size: 10px; font-weight: 650;')
        self.score_label.setText(self._score_text);self.score_label.raise_()

    def resizeEvent(self,event):
        super().resizeEvent(event);self._update_label()


class BoardEvaluationPanel(QWidget):
    """Keep bar and actual board square adjacent, equal in height, and centered."""
    def __init__(self,board,bar,parent=None,*,reserve_artwork=True):
        super().__init__(parent)
        self.reserve_artwork=reserve_artwork
        self.board=board;self.bar=bar
        board.setParent(self);bar.setParent(self)
        self.setMinimumSize(board.minimumWidth()+44,board.minimumHeight())
        from .decorations import DecorativeImage,RightDecorations
        self.ino=DecorativeImage('ino.png',self)
        self.right_decorations=RightDecorations(self)
        board.orientation_changed.connect(bar.set_orientation)
        bar.set_orientation(board.orientation)

    def resizeEvent(self,event):
        # Reserve proportional artwork columns even on a compact display.
        art = min(240, max(32, int(self.width()*.13)))
        side=(min(self.height(),max(128,self.width()-44-2*(art+12)))
              if self.reserve_artwork else min(self.height(),max(1,self.width()-44)))
        left=(self.width()-side-44)//2;top=(self.height()-side)//2
        self.bar.setGeometry(left,top,34,side)
        self.board.setGeometry(left+44,top,side,side)
        # Artwork uses spare margins only. Board/bar geometry is unchanged;
        # the decorative columns disappear before they could overlap controls.
        margin=max(0,left-(12 if self.reserve_artwork else 24))
        visible=margin>=(24 if self.reserve_artwork else 72)
        self.ino.setVisible(visible)
        self.right_decorations.setVisible(visible)
        if visible:
            left_width=min(240,margin)
            right_width=min(210,margin)
            self.ino.setGeometry(left-12-left_width,top+int(side*.10),left_width,int(side*.80))
            self.right_decorations.setGeometry(left+44+side+12,top+8,right_width,side-16)
        super().resizeEvent(event)
