"""Separate screen-only premove intent from legally confirmed game state.

The expensive matching runs inside the existing recognition worker. A red
background alone never proves a move. This module never mutates a tracker,
turn, FEN/history or an engine; it only interprets a captured placement.
"""
from dataclasses import dataclass, replace
import time
import chess
import numpy as np
from PySide6.QtCore import Qt
from .capture.board_region import rgb
from .last_move_highlight import HighlightedCells
from .recognition import RecognitionResult, SquarePrediction, board_from_placement, remap_recognition


def detect_premove_cells(image):
    """Exactly two red/rust full-square backgrounds; sample around glyphs."""
    if image.isNull() or min(image.width(),image.height())<128 or abs(image.width()/image.height()-1)>.12:
        return None
    small=image.scaled(160,160,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
    pixels=rgb(small);rows,cols=np.indices((8,8))
    # In the real h2/g3 capture the h2 border consumes half a 1px-inset
    # patch. Sample inside that border, still outside the central glyph.
    offsets=[(y+dy,x+dx) for y,x in ((3,3),(3,15),(15,3),(15,15)) for dy in (0,1) for dx in (0,1)]
    oy,ox=np.asarray(offsets).T
    patches=pixels[rows.reshape(64,1)*20+oy,cols.reshape(64,1)*20+ox].reshape(64,4,4,3)
    values=patches.astype(np.float32);r,g,b=values[...,0],values[...,1],values[...,2]
    flags=(r>=105)&(r>=g*1.22)&(r>=b*1.20)&((r-np.minimum(g,b))>=40)
    strengths=(np.count_nonzero(flags.mean(axis=2)>=.75,axis=1)>=3)&(flags.mean(axis=(1,2))>=.70)
    medians=np.median(patches.reshape(64,16,3),axis=1);parity=(rows+cols).reshape(-1)%2
    base=np.asarray([np.median(medians[parity==color],axis=0) for color in (0,1)])
    indices=np.flatnonzero(strengths&(np.linalg.norm(medians-base[parity],axis=1)>=35))
    return HighlightedCells(tuple(divmod(int(index),8) for index in indices)) if len(indices)==2 else None


@dataclass(frozen=True)
class PremoveIntent:
    move: chess.Move
    color: bool
    piece_symbol: str


@dataclass(frozen=True)
class PendingPremove:
    intent: PremoveIntent
    anchor_fen: str
    anchor_history: tuple[chess.Move,...]
    observed_at: float
    scope: tuple
    observations: int=1


@dataclass(frozen=True)
class PremoveEvidence:
    root_fen: str
    epoch: int
    scope: tuple
    original: RecognitionResult
    interpreted: RecognitionResult | None
    intent: PremoveIntent | None
    red_squares: tuple[int,...]
    hold: bool=False
    reason: str=''


def plausible_intent(board, origin, destination, observed):
    piece=board.piece_at(origin)
    if piece is None or origin==destination:return None
    if board.piece_type_at(destination)==chess.KING:return None
    dx=abs(chess.square_file(destination)-chess.square_file(origin))
    dy=chess.square_rank(destination)-chess.square_rank(origin)
    kind=piece.piece_type;promotion=None
    if kind==chess.PAWN:
        forward=1 if piece.color else -1
        valid=(dy==forward and dx<=1) or (dx==0 and dy==2*forward
              and chess.square_rank(origin)==(1 if piece.color else 6)
              and board.piece_at(origin+8*forward) is None)
        if not valid:return None
        if chess.square_rank(destination) in (0,7):
            shown=observed.piece_at(destination)
            if shown and shown.color==piece.color and shown.piece_type in (chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT):
                promotion=shown.piece_type
    elif kind==chess.KNIGHT:
        if (dx,abs(dy)) not in ((1,2),(2,1)):return None
    elif kind==chess.KING:
        castle=(origin==chess.square(4,0 if piece.color else 7) and dy==0 and dx==2
                and chess.square_file(destination) in (2,6))
        if not castle and max(dx,abs(dy))!=1:return None
        if castle:
            rook=chess.square(7 if destination>origin else 0,chess.square_rank(origin))
            if board.piece_at(rook)!=chess.Piece(chess.ROOK,piece.color):return None
    else:
        diagonal=dx==abs(dy) and dx>0;straight=(dx==0)!=(dy==0)
        if not ((kind in (chess.BISHOP,chess.QUEEN) and diagonal)
                or (kind in (chess.ROOK,chess.QUEEN) and straight)):return None
        if chess.between(origin,destination)&board.occupied:return None
    return PremoveIntent(chess.Move(origin,destination,promotion=promotion),piece.color,piece.symbol())


def preview_placements(board,intent):
    """Possible SCREEN layouts, not legal positions/FENs sent to the engine."""
    yield board.board_fen()  # Some platforms highlight without relocating.
    move=intent.move;piece=board.piece_at(move.from_square)
    if piece is None or piece.symbol()!=intent.piece_symbol:return
    ghost=board.copy(stack=False);ghost.remove_piece_at(move.from_square)
    ghost.set_piece_at(move.to_square,chess.Piece(move.promotion or piece.piece_type,piece.color))
    yield ghost.board_fen()
    if piece.piece_type==chess.PAWN and board.ep_square==move.to_square and board.is_en_passant(move):
        ghost.remove_piece_at(move.to_square+(-8 if piece.color else 8))
        yield ghost.board_fen()
    if piece.piece_type==chess.KING and abs(move.to_square-move.from_square)==2:
        rank=chess.square_rank(move.from_square);short=move.to_square>move.from_square
        rook_from=chess.square(7 if short else 0,rank);rook_to=chess.square(5 if short else 3,rank)
        if board.piece_at(rook_from)==chess.Piece(chess.ROOK,piece.color):
            ghost.remove_piece_at(rook_from);ghost.set_piece_at(rook_to,chess.Piece(chess.ROOK,piece.color))
            yield ghost.board_fen()


def _interpreted(result,board):
    if result.piece_placement==board.board_fen():return result
    predictions=tuple(SquarePrediction(p.square,board.piece_at(p.square).symbol() if board.piece_at(p.square) else None,
                      'premove_mask' if p.symbol!=(board.piece_at(p.square).symbol() if board.piece_at(p.square) else None)
                      else p.template_class,p.confidence) for p in result.predictions)
    return RecognitionResult(board.board_fen(),result.orientation,predictions,result.confidence_threshold,result.timings)


def interpret_premove(result,red,confirmed,pending=None,*,turn_known=True,orientation='white',epoch=0,scope=()):
    """Constrain visual ghosts against zero or ONE real legal opponent move.

    Once red disappears, ordinary tracking still reconciles shortest legal
    paths (one/two plies before its existing longer-gap fallback). No hinted
    move is pushed into confirmed history by this interpreter.
    """
    root_fen=confirmed.fen()
    def evidence(normalized,intent=None,squares=(),hold=False,reason='',original=result):
        return PremoveEvidence(root_fen,epoch,scope,original,normalized,intent,tuple(squares),hold,reason)
    if red is None:
        if pending is not None:
            intent=pending.intent
            # A missing red animation frame cannot manufacture three plies
            # (opponent out-and-back) just to explain one white ghost move.
            for view in (orientation,'black' if orientation=='white' else 'white'):
                current=remap_recognition(result,view)
                if (current.piece_placement!=confirmed.board_fen()
                        and current.piece_placement in preview_placements(confirmed,intent)
                        and (not turn_known or confirmed.turn!=intent.color)):
                    return evidence(None,intent,hold=True,reason='ghost_without_opponents_move')
        return evidence(result,pending.intent if pending else None)
    if pending is not None and not result.uncertain_squares and all(
            sum(board_from_placement(remap_recognition(result,view).piece_placement).piece_at(s)
                !=confirmed.piece_at(s) for s in chess.SQUARES)>8
            for view in (orientation,'black' if orientation=='white' else 'white')):
        # A stale intent from another video/position cannot hold tracking forever.
        # This releases the observation to the normal stable/legal tracker; it
        # does not confirm a move or manufacture a turn.
        return evidence(result,reason='stale_premove_context')
    roots=[]
    for color in ((confirmed.turn,) if turn_known else chess.COLORS):
        root=confirmed.copy(stack=False);root.turn=color
        if not turn_known:root.ep_square=None
        if root.is_valid():roots.append(root)
    real_boards=[confirmed.copy(stack=False)]
    for root in roots:
        for move in root.legal_moves:
            child=root.copy(stack=False);child.push(move);real_boards.append(child)
    any_plausible=False
    for view in (orientation,'black' if orientation=='white' else 'white'):
        current=remap_recognition(result,view);observed=board_from_placement(current.piece_placement)
        if sum(observed.piece_at(s)!=confirmed.piece_at(s) for s in chess.SQUARES)>8:continue
        squares=red.squares(view);intents=[]
        if pending and {pending.intent.move.from_square,pending.intent.move.to_square}==set(squares):
            intents.append(pending.intent)
        for origin,destination in (squares,squares[::-1]):
            piece=confirmed.piece_at(origin)
            allowed=(pending.intent.color if pending else not confirmed.turn) if turn_known else None
            if piece is None or (allowed is not None and piece.color!=allowed):continue
            intent=plausible_intent(confirmed,origin,destination,observed)
            if intent and intent not in intents:intents.append(intent)
        any_plausible |= bool(intents)
        matches={}
        for intent in intents:
            for board in real_boards:
                if current.piece_placement in preview_placements(board,intent):
                    matches[(board.board_fen(),intent)] = board
        placements={key[0] for key in matches};matched_intents={key[1] for key in matches}
        if len(placements)==1 and len(matched_intents)==1:
            intent=next(iter(matched_intents));real=next(iter(matches.values()))
            if current.uncertain_squares:
                return evidence(None,intent,squares,True,'uncertain_premove',current)
            # If the real opponent captured the source, cancel its intent; the
            # matched direct layout remains eligible for ordinary confirmation.
            if real.piece_at(intent.move.from_square)!=chess.Piece.from_symbol(intent.piece_symbol):
                return evidence(_interpreted(current,real),None,(),False,'source_captured',current)
            return evidence(_interpreted(current,real),intent,squares,False,'premove_overlay',current)
        if matches:
            return evidence(None,pending.intent if pending else None,squares,True,'ambiguous_red',current)
    # A projected piece vacates its source. Two occupied marked squares on a
    # complete board can be annotations over a game that has advanced several
    # plies. Let the normal stable/legal tracker reconcile the actual pieces.
    if pending is None and not result.uncertain_squares and all(
            board_from_placement(result.piece_placement).piece_at(sq) is not None
            for sq in red.squares(result.orientation)):
        return evidence(result)
    if pending or any_plausible:
        return evidence(None,pending.intent if pending else None,red.squares(orientation),True,'unresolved_red')
    return evidence(result)  # Red marks without premove/history evidence are not a blanket veto.


class PremoveMonitor:
    """Tiny UI-side intent lifecycle; the tracker remains the sole game owner."""
    def __init__(self):
        self.revision=0;self.reset()

    def reset(self):
        self.revision+=1;self.pending=None;self.state='NONE';self.last_outcome=None
        self.squares=();self._key=None;self._count=0;self._last=None

    def consume(self,evidence,board,*,visual_stable):
        if evidence.epoch!=self.revision or evidence.root_fen!=board.fen():return False
        if evidence.reason=='stale_premove_context':self.pending=None
        self.revision+=1;self._last=evidence
        key=(evidence.original.piece_placement,evidence.red_squares,evidence.intent,evidence.reason)
        if key==self._key:
            if visual_stable:self._count+=1
        else:self._key=key;self._count=1
        self.squares=evidence.red_squares
        if evidence.intent and evidence.red_squares:
            if self.pending is None or self.pending.intent!=evidence.intent:
                self.pending=PendingPremove(evidence.intent,board.fen(),tuple(board.move_stack),time.perf_counter(),evidence.scope)
            else:
                self.pending=replace(self.pending,scope=evidence.scope,observations=self.pending.observations+int(visual_stable))
            self.state='PENDING' if self._count>=2 else 'CANDIDATE'
        elif self.pending:self.state='RESOLVING'
        elif evidence.hold:self.state='CANDIDATE'
        else:self.state='NONE'
        return True

    @property
    def needs_retry(self):return self._count<2 and self.state!='NONE'

    def after_tracking(self,board,*,visual_stable):
        evidence=self._last
        if self.pending is None or evidence is None:return
        if evidence.interpreted is None or board.board_fen()!=evidence.interpreted.piece_placement:return
        if evidence.red_squares and evidence.intent:
            if evidence.reason!='source_captured':return
        if not evidence.red_squares and not evidence.hold and visual_stable:
            old=self.pending;history=tuple(board.move_stack);prefix=old.anchor_history
            suffix=history[len(prefix):] if history[:len(prefix)]==prefix else ()
            anchor_turn=chess.Board(old.anchor_fen).turn
            executed=any((anchor_turn if index%2==0 else not anchor_turn)==old.intent.color
                         and m.from_square==old.intent.move.from_square and m.to_square==old.intent.move.to_square
                         and (old.intent.move.promotion is None or m.promotion==old.intent.move.promotion)
                         for index,m in enumerate(suffix))
            self.last_outcome='COMPLETED' if executed else ('CANCELED' if board.fen()==old.anchor_fen or suffix else 'UNRESOLVED')
            self.pending=None;self.state='NONE';self.squares=();self.revision+=1
