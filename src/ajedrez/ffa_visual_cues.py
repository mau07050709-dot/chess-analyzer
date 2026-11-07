"""Additional FFA evidence: orientation, coloured last moves and running clocks.

No OCR, additional captures or model sessions. All samples come from the
already-selected 14x14 rectangle, including its inactive corner areas.
"""
from dataclasses import dataclass
import time
import numpy as np
from PySide6.QtCore import Qt
from .four_player import PLAYERS,display_cell
from .capture.board_region import rgb


def hue_labels(values,minimum_chroma=18):
    a=values.astype(np.float32);hi=a.max(-1);lo=a.min(-1);d=np.maximum(hi-lo,.001)
    h=np.zeros_like(hi);red=hi==a[...,0];green=(hi==a[...,1])&~red;blue=~red&~green
    h[red]=((a[...,1]-a[...,2])/d)[red]%6
    h[green]=((a[...,2]-a[...,0])/d+2)[green]
    h[blue]=((a[...,0]-a[...,1])/d+4)[blue];h/=6
    labels=np.full(hi.shape,-1,dtype=np.int8)
    for i,mask in enumerate(((h<.065)|(h>.94),(h>.50)&(h<.72),(h>.08)&(h<.20),(h>.23)&(h<.48))):
        labels[mask&(hi-lo>=minimum_chroma)]=i
    return labels


def orient(pieces,bottom,canonical):
    visual={display_cell(c,r,bottom):p for (c,r),p in pieces.items()}
    candidates=[]
    for orientation in PLAYERS:
        mapped={canonical(c,r,orientation):p for (c,r),p in visual.items()}
        # Surviving home armies establish view even after some moves. Never
        # infer turn from these positions. Require evidence from 3 armies.
        home=[0]*4
        for (c,r),piece in mapped.items():
            if piece.player not in PLAYERS:continue
            seat=PLAYERS.index(piece.player)
            inside=(r>=11 if seat==0 else c<=2 if seat==1 else r<=2 if seat==2 else c>=11)
            if inside:home[seat]+=1
        candidates.append((sum(home),sum(n>=2 for n in home),orientation,mapped))
    candidates.sort(key=lambda v:v[0],reverse=True)
    required=1 if sum(p.player=='dead' for p in pieces.values())>=3 else 3
    if candidates[0][1]>=required and candidates[0][0]-candidates[1][0]>=8:
        return candidates[0][3],candidates[0][2]
    return pieces,bottom


def position_from_pieces(pieces):
    from . import ffa_engine
    b,*_=ffa_engine.modules();position=b.Board(b.MODE_FFA)
    for (c,r),piece in pieces.items():
        seat=PLAYERS.index(piece.player) if piece.player in PLAYERS else b.DEAD_UNKNOWN
        position.sq[b.sq_of(c,13-r)]=b.make_piece(seat,b.TYPE_CHARS.index(piece.kind))
    position.find_kings();position.alive=[king>=0 for king in position.kings];position.recompute_key()
    return position


def reverse_is_legal(pieces,origin,destination,player):
    from . import ffa_engine
    b,gen,*_=ffa_engine.modules()
    if origin in pieces or destination not in pieces or pieces[destination].player!=player:return False
    current=position_from_pieces(pieces)
    if not current.alive[PLAYERS.index(player)]:return False
    seat=PLAYERS.index(player);a=b.sq_of(origin[0],13-origin[1]);z=b.sq_of(destination[0],13-destination[1])
    piece=current.sq[z]
    kinds=[b.PC_TYPE[piece]]
    if b.PC_TYPE[piece]==b.QUEEN:kinds.append(b.PAWN)
    for kind in kinds:
        before=current.copy();before.turn=seat;before.sq[a]=b.make_piece(seat,kind);before.sq[z]=0
        # Enroque: reconstruct king + rook only in this disposable proof.
        if kind==b.KING:
            for _,geometry in (b.castle_geometry(seat,a) or {}).items():
                rook_from,king_to,rook_to=geometry[:3]
                if z==king_to and before.sq[rook_to]==b.make_piece(seat,b.ROOK) and not before.sq[rook_from]:
                    before.sq[rook_from]=before.sq[rook_to];before.sq[rook_to]=0
                    before.ck[seat]=before.cq[seat]=True
        # A pawn diagonal needs a victim in the predecessor. This is a proof
        # of movement shape, not reconstructed capture history or points.
        if kind==b.PAWN and z-a in b.PAWN_TAKES[seat]:
            before.sq[z]=b.make_piece((seat+1)%4,b.PAWN)
        before.find_kings();before.recompute_key()
        for move in gen.gen_legal(before):
            if b.mv_from(move)==a and b.mv_to(move)==z:return True
    return False


@dataclass(frozen=True)
class LastMove:
    player:str
    origin:tuple
    destination:tuple
    reverse_legal:bool=True


def movement_shape(pieces,origin,destination,player):
    """Older marks may predate a later check; use geometry only for filtering
    forward legal recovery, never as standalone proof of the current turn."""
    if origin in pieces or destination not in pieces or pieces[destination].player!=player:return False
    from . import ffa_engine
    b,*_=ffa_engine.modules();seat=PLAYERS.index(player);piece=pieces[destination]
    a=b.sq_of(origin[0],13-origin[1]);z=b.sq_of(destination[0],13-destination[1]);delta=z-a
    dx=destination[0]-origin[0];dy=destination[1]-origin[1]
    if piece.kind=='P':
        return (delta in b.PAWN_TAKES[seat] or delta==b.PAWN_PUSH[seat] or
                (delta==2*b.PAWN_PUSH[seat] and b.pawn_coord(seat,a)==1))
    if piece.kind=='N':return sorted((abs(dx),abs(dy)))==[1,2]
    if piece.kind=='K':return max(abs(dx),abs(dy))==1
    if not (piece.kind in 'BQ' and abs(dx)==abs(dy) or piece.kind in 'RQ' and (dx==0 or dy==0)):return False
    sx=(dx>0)-(dx<0);sy=(dy>0)-(dy<0);c,r=origin[0]+sx,origin[1]+sy
    while (c,r)!=destination:
        if (c,r) in pieces:return False
        c+=sx;r+=sy
    return True


def last_moves(cells,borders,pieces,bottom,canonical):
    labels=hue_labels(borders)
    result=[]
    for seat,player in enumerate(PLAYERS):
        marked=[canonical(*cells[i],bottom) for i in np.flatnonzero(labels==seat)]
        if not 2<=len(marked)<=3:continue
        candidates=[]
        for a in marked:
            for z in marked:
                if a!=z:
                    legal=reverse_is_legal(pieces,a,z,player)
                    if legal or movement_shape(pieces,a,z,player):candidates.append(LastMove(player,a,z,legal))
        if len(candidates)==1:result.append(candidates[0])
    return tuple(result)


@dataclass(frozen=True)
class ClockSample:
    player:str
    box:tuple
    digits:bytes


def clocks(image):
    pixels=rgb(image.scaled(280,280,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation))
    samples=[]
    for x0,y0 in ((0,40),(220,40),(0,220),(220,220)):
        region=pixels[y0:y0+20,x0:x0+60];labels=hue_labels(region,35)
        for seat,player in enumerate(PLAYERS):
            mask=labels==seat
            rows=np.flatnonzero(mask.sum(1)>=20)
            if len(rows)<5 or mask.sum()<100:continue
            cols=np.flatnonzero(mask[rows].sum(0)>=len(rows)*.35)
            if len(cols)<25:continue
            left,right=int(cols[0]),int(cols[-1]+1);top,bottom=int(rows[0]),int(rows[-1]+1)
            # Locate on a small image, retain native pixels for the digits.
            # Reducing a 40-px clock to 12 px erased strokes such as 9→8.
            sx=image.width()/280;sy=image.height()/280
            ax=round((x0+left+4)*sx);ay=round((y0+top+1)*sy)
            bx=round((x0+right-1)*sx);by=round((y0+bottom-1)*sy)
            patch=rgb(image.copy(ax,ay,bx-ax,by-ay)).astype(float)
            if not patch.size:continue
            background=np.median(region[mask].astype(float),axis=0)
            ink=(patch.mean(2)>background.mean()+55)&(patch.max(2)>160)
            if ink.sum()<6:continue
            samples.append(ClockSample(player,(x0+left,y0+top,right-left,bottom-top),
                                       bytes(ink.astype('uint8').reshape(-1))))
    return tuple(samples)


class ClockTurn:
    """Two independent digit changes on an otherwise stable board identify a clock.

    A single update may be increment credit after a move, so it is insufficient.
    Geometry, position, another clock changing or missing samples reset evidence.
    """
    def __init__(self):self.reset()
    def reset(self):
        self.key=None;self.samples={};self.player=None;self.hits=0;self.last_at=0
    def observe(self,observation,now=None):
        now=time.monotonic() if now is None else now
        samples={s.player:s for s in observation.clock_samples}
        key=(observation.signature,observation.bottom)
        if observation.uncertain or observation.transient or key!=self.key:
            self.reset();self.key=key;self.samples=samples;return None
        if not samples:
            self.samples={};self.player=None;self.hits=0;return None
        if self.player is not None and self.player not in samples:
            self.player=None;self.hits=0
        changed=[]
        for player,sample in samples.items():
            old=self.samples.get(player)
            if old is None or old.box!=sample.box or len(old.digits)!=len(sample.digits):
                self.player=None;self.hits=0;continue
            delta=sum(a!=b for a,b in zip(old.digits,sample.digits))
            if 4<=delta<=len(sample.digits)*.55:changed.append(player)
        self.samples=samples
        if len(changed)>1:self.player=None;self.hits=0;return None
        if len(changed)==1:
            player=changed[0]
            if player==self.player and .3<=now-self.last_at<=3.5:self.hits+=1
            else:self.player=player;self.hits=1
            self.last_at=now
            if self.hits>=2:return player
        return self.player if self.hits>=2 and now-self.last_at<=3.5 else None
