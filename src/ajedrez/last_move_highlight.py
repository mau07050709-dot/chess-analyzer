"""Complementary turn evidence from a stable Chess.com last-move highlight.

This module never recognizes pieces, stores invented moves, or changes FEN
metadata. Reverse positions below are disposable legality probes only.
"""
from dataclasses import dataclass
import chess
import numpy as np
from PySide6.QtCore import Qt
from .board_coordinates import grid_to_square
from .capture.board_region import rgb


@dataclass(frozen=True)
class HighlightedCells:
    cells: tuple[tuple[int, int], ...]  # Physical image (row, column), not chess squares.

    def squares(self, orientation):
        return tuple(grid_to_square(row,col,orientation) for row,col in self.cells)


def _yellow(samples):
    values=samples.astype(np.float32)
    maximum=values.max(axis=-1);minimum=values.min(axis=-1)
    delta=np.maximum(maximum-minimum,1.)
    red,green,blue=values[...,0],values[...,1],values[...,2]
    hue=np.where(maximum==red,60*((green-blue)/delta)%360,
                 np.where(maximum==green,120+60*(blue-red)/delta,240+60*(red-green)/delta))
    saturation=(maximum-minimum)/np.maximum(maximum,1.)
    return (hue>=45)&(hue<=78)&(saturation>=.28)&(maximum>=110)


def _highlight_candidates(image, inset=1):
    """Measure full-square yellow backgrounds, tolerating glyphs/JPEG.

    Four small corner patches avoid piece centers and legal-move dots. At
    least three corners must agree. Original light/dark backgrounds are learned
    separately; a yellow board theme itself is not last-move evidence.
    """
    if image.isNull() or min(image.width(),image.height())<128:
        return None
    if abs(image.width()/image.height()-1)>.12:
        return None
    small=image.scaled(160,160,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
    pixels=rgb(small)
    rows,cols=np.indices((8,8))
    far=18-inset
    offsets=[(y+dy,x+dx) for y,x in ((inset,inset),(inset,far),(far,inset),(far,far))
             for dy in (0,1) for dx in (0,1)]
    oy,ox=np.asarray(offsets).T
    patches=pixels[rows.reshape(64,1)*20+oy,cols.reshape(64,1)*20+ox].reshape(64,4,4,3)
    flags=_yellow(patches)
    strengths=(np.count_nonzero(flags.mean(axis=2)>=.75,axis=1)>=3)&(flags.mean(axis=(1,2))>=.70)
    medians=np.median(patches.reshape(64,16,3),axis=1)
    parity=(rows+cols).reshape(-1)%2
    baseline=np.asarray([np.median(medians[parity==color],axis=0) for color in (0,1)])
    different=np.linalg.norm(medians-baseline[parity],axis=1)>=35
    indices=np.flatnonzero(np.asarray(strengths)&different)
    return indices,medians


def detect_highlights(image):
    candidates=_highlight_candidates(image)
    if candidates is None:return None
    indices,_=candidates
    if len(indices)!=2:return None
    return HighlightedCells(tuple(divmod(int(index),8) for index in indices))


def detect_turn_highlights(image, placement, orientation, ordinary_checked=False):
    """Additional evidence only for an unknown turn, using ORIGINAL pixels.

    Video overlays may leave an isolated selected square beside a last-move
    pair. Accept three marks only when two backgrounds closely match, the
    third is clearly different AND empty, and exactly one pair has a legal
    reverse explanation. The caller still requires temporal confirmation.
    Never choose a pair merely because a result has higher chess evaluation.
    """
    if not ordinary_checked:
        ordinary=detect_highlights(image)
        if ordinary is not None:return ordinary
    from .recognition import board_from_placement
    board=board_from_placement(placement)
    from .premove import detect_premove_cells
    if detect_premove_cells(image) is not None:return None
    for inset in (1,3):
        candidates=_highlight_candidates(image,inset)
        if candidates is None:continue
        indices,medians=candidates
        if len(indices)==2 and inset==3:
            pair=HighlightedCells(tuple(divmod(int(i),8) for i in indices))
            if resolve_last_move(board,pair,orientation) is not None:return pair
        if len(indices)!=3:continue
        interpretations=[]
        for a,b,c in ((0,1,2),(0,2,1),(1,2,0)):
            ia,ib,ic=(int(indices[i]) for i in (a,b,c))
            pair=HighlightedCells((divmod(ia,8),divmod(ib,8)))
            if resolve_last_move(board,pair,orientation) is not None:
                interpretations.append((pair,ia,ib,ic))
        if len(interpretations)!=1:continue
        pair,a,b,other=interpretations[0]
        extra=HighlightedCells((divmod(other,8),)).squares(orientation)[0]
        if board.piece_at(extra) is not None:continue
        if (np.linalg.norm(medians[a]-medians[b])<=24
                and min(np.linalg.norm(medians[other]-medians[a]),
                        np.linalg.norm(medians[other]-medians[b]))>=40):return pair
    return None


@dataclass(frozen=True)
class TurnEvidence:
    origin: int
    destination: int
    mover: bool

    @property
    def next_turn(self):return not self.mover


def _reaches(current, previous, move):
    if not previous.is_valid() or move not in previous.legal_moves:return False
    previous.push(move)
    return previous.board_fen()==current.board_fen()


def resolve_last_move(board, highlights, orientation, anchor=None):
    """Require empty origin, occupied destination and a legal reverse explanation."""
    if highlights is None or len(highlights.cells)!=2:return None
    squares=highlights.squares(orientation)
    occupied=[sq for sq in squares if board.piece_at(sq) is not None]
    empty=[sq for sq in squares if board.piece_at(sq) is None]
    if len(occupied)!=1 or len(empty)!=1:return None
    origin,destination=empty[0],occupied[0]
    piece=board.piece_at(destination)
    # Reject unchanged markers/premoves already present in the last confirmed
    # state. A highlight alone must not override known visual history.
    if (anchor is not None and anchor.piece_at(origin)==board.piece_at(origin)
            and anchor.piece_at(destination)==piece):return None
    current=board.copy(stack=False);current.turn=not piece.color
    if not current.is_valid():return None
    previous=current.copy(stack=False);previous.turn=piece.color;previous.ep_square=None
    previous.remove_piece_at(destination);previous.set_piece_at(origin,piece)
    move=chess.Move(origin,destination)
    if _reaches(current,previous.copy(stack=False),move):
        return TurnEvidence(origin,destination,piece.color)
    # A king's two-square transition must also reverse the rook, and must pass
    # the ordinary castling attack/path checks. The temporary right is not saved.
    if piece.piece_type==chess.KING:
        rank=0 if piece.color else 7
        if origin==chess.square(4,rank) and destination in (chess.square(6,rank),chess.square(2,rank)):
            king_side=chess.square_file(destination)==6
            rook_from=chess.square(7 if king_side else 0,rank)
            rook_to=chess.square(5 if king_side else 3,rank)
            if (current.piece_at(rook_to)==chess.Piece(chess.ROOK,piece.color)
                    and current.piece_at(rook_from) is None):
                probe=previous.copy(stack=False);probe.remove_piece_at(rook_to)
                probe.set_piece_at(rook_from,chess.Piece(chess.ROOK,piece.color))
                probe.castling_rights|=chess.BB_SQUARES[rook_from]
                if _reaches(current,probe,move):return TurnEvidence(origin,destination,piece.color)
    # A captured piece can have blocked a different attack in the predecessor.
    # Trying only an empty destination would reject that legitimate capture.
    for victim in (chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN):
        probe=previous.copy(stack=False)
        probe.set_piece_at(destination,chess.Piece(victim,not piece.color))
        if _reaches(current,probe,move):return TurnEvidence(origin,destination,piece.color)
    # Captures/promotions: only disposable predecessor candidates are tried.
    # No captured piece or promotion history is added to the real board.
    promotion=(piece.piece_type in (chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT)
               and chess.square_rank(destination)==(7 if piece.color else 0)
               and chess.square_rank(origin)==(6 if piece.color else 1))
    if piece.piece_type==chess.PAWN or promotion:
        probe=previous.copy(stack=False)
        probe.set_piece_at(origin,chess.Piece(chess.PAWN,piece.color))
        pawn_move=chess.Move(origin,destination,promotion=piece.piece_type if promotion else None)
        for victim in (None,chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN):
            candidate=probe.copy(stack=False)
            if victim is not None:candidate.set_piece_at(destination,chess.Piece(victim,not piece.color))
            if _reaches(current,candidate,pawn_move):return TurnEvidence(origin,destination,piece.color)
        if not promotion and chess.square_rank(origin)==(4 if piece.color else 3):
            captured=chess.square(chess.square_file(destination),chess.square_rank(origin))
            if current.piece_at(captured) is None:
                probe.set_piece_at(captured,chess.Piece(chess.PAWN,not piece.color));probe.ep_square=destination
                if _reaches(current,probe,pawn_move):return TurnEvidence(origin,destination,piece.color)
    return None


class HighlightTurnRecovery:
    """Two stable matching observations, tied to position/session/geometry."""
    def __init__(self):self.reset()

    def reset(self):
        self.anchor=None;self.clear_pending()

    def clear_pending(self):
        self._key=None;self._count=0

    def remember_known(self,board):
        self.anchor=board.copy(stack=False);self.clear_pending()

    @property
    def pending(self):return self._key is not None

    def observe(self,board,highlights,orientation,token,*,visual_stable):
        evidence=resolve_last_move(board,highlights,orientation,self.anchor)
        if evidence is None:
            self.clear_pending();return None
        key=(token,orientation,board.fen(),evidence.origin,evidence.destination,evidence.next_turn)
        if key!=self._key:
            self._key=key;self._count=1
        elif visual_stable:self._count+=1
        if visual_stable and self._count>=2:return evidence
        return None
