"""Conservative absolute and material-relative line pins on confirmed layouts."""
from dataclasses import dataclass
import time
import chess
from .vulnerability import VALUES, legal_exchange_gain


@dataclass(frozen=True)
class PinnedPiece:
    square: chess.Square
    kind: str  # Stable IDs: absolute / relative.
    attacker: chess.Square | None
    target: chess.Square
    gain_cp: int | None = None


@dataclass(frozen=True)
class PinResult:
    pins: tuple[PinnedPiece, ...]
    complete: bool
    nodes: int  # Number of geometric relative candidates checked.
    elapsed_ms: float

    @property
    def squares(self):
        return frozenset(pin.square for pin in self.pins)


def find_pins(board: chess.Board, *, turn_known=True, budget_ms=50,
              cancelled=lambda: False) -> PinResult:
    """No moves or historical fields are changed on the caller's board.

    Relative pins use a counterfactual empty blocker, NOT an invented move.
    A legal winning capture by the aligned slider must exist there. A legal
    non-losing countercapture of that slider by the exposed target rejects the
    relative mark. This deliberately omits some conventional geometric pins.
    """
    started=time.perf_counter();deadline=started+budget_ms/1000
    original=board.copy(stack=False)
    probe=original.copy(stack=False);probe.ep_square=None
    valid=False
    for color in chess.COLORS:
        probe.turn=color;valid |= probe.is_valid()
    if not valid or cancelled():
        return PinResult((),False,0,(time.perf_counter()-started)*1000)
    found={square:PinnedPiece(square,'absolute',None,original.king(piece.color))
           for square,piece in original.piece_map().items()
           if piece.piece_type!=chess.KING and original.is_pinned(piece.color,square)}
    candidates=0;complete=True
    def expired():return cancelled() or time.perf_counter()>=deadline
    def result():return PinResult(tuple(found[s] for s in sorted(found)),complete,candidates,
                                 (time.perf_counter()-started)*1000)
    for attacker,piece in original.piece_map().items():
        if piece.piece_type not in (chess.ROOK,chess.BISHOP,chess.QUEEN):continue
        directions=[]
        if piece.piece_type in (chess.ROOK,chess.QUEEN):directions += [(1,0),(-1,0),(0,1),(0,-1)]
        if piece.piece_type in (chess.BISHOP,chess.QUEEN):directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx,dy in directions:
            if expired():complete=False;return result()
            x,y=chess.square_file(attacker)+dx,chess.square_rank(attacker)+dy
            occupied=[]
            while 0<=x<8 and 0<=y<8:
                square=chess.square(x,y)
                if original.piece_at(square):
                    occupied.append(square)
                    if len(occupied)==2:break
                x+=dx;y+=dy
            if len(occupied)!=2:continue
            blocker,target=occupied
            middle,behind=original.piece_at(blocker),original.piece_at(target)
            if (middle.color==piece.color or behind.color!=middle.color
                    or middle.piece_type==chess.KING or behind.piece_type==chess.KING
                    or blocker in found or VALUES[behind.piece_type]<=VALUES[middle.piece_type]):continue
            candidates+=1
            exposed=original.copy(stack=False)
            exposed.remove_piece_at(blocker)
            exposed.ep_square=None  # Removing a blocker proves no special historical move.
            exposed.turn=piece.color
            if chess.Move(attacker,target) not in exposed.legal_moves:continue
            gain=legal_exchange_gain(exposed,chess.Move(attacker,target),
                                     budget_ms=max(0,(deadline-time.perf_counter())*1000),cancelled=expired)
            if gain is None:
                complete=False
                if expired():complete=False;return result()
                continue
            if gain<=0:continue
            exposed.turn=behind.color
            reverse=chess.Move(target,attacker)
            if reverse in exposed.legal_moves:
                counter=legal_exchange_gain(exposed,reverse,
                                             budget_ms=max(0,(deadline-time.perf_counter())*1000),cancelled=expired)
                if counter is None:
                    complete=False
                    if expired():return result()
                    continue
                if counter>=0:continue
            found[blocker]=PinnedPiece(blocker,'relative',attacker,target,gain)
    return result()
