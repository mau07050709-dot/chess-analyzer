"""Stable FFA observations and legal transitions, isolated from classic mode."""
import time
from . import ffa_engine as engine
from .four_player import PLAYERS,initial_pieces


def _signature(pieces):
    return tuple(sorted((c,r,p.player,p.kind) for (c,r),p in pieces.items()))


class FFATracker:
    def __init__(self,position=None):
        self.position=position.copy() if position is not None else None
        self.turn_known=position is not None
        self.history_known=position is not None
        self.pending=None
        self.observations=0
        self.previous=None

    def reset_pending(self):
        self.pending=None;self.observations=0

    def confirm_turn(self,player):
        if self.position is None:return None
        position=self.position.copy();seat=PLAYERS.index(player)
        if not position.alive[seat]:return None
        if not self.turn_known or seat!=position.turn:position.ep=[None]*4
        position.turn=seat;position.recompute_key()
        self.position=position;self.turn_known=True
        return position.copy()

    def observe(self,observation,dragging=False):
        if dragging or observation.uncertain or observation.transient:
            self.reset_pending();return 'retry',None
        key=observation.signature
        if key!=self.pending:self.pending=key;self.observations=1
        else:self.observations+=1
        if self.observations<2:return 'waiting',None
        # A stale mark or explicit hint may name an eliminated seat. It is
        # not evidence of resurrection and must never return a None position.
        hinted=observation.turn_hint
        if hinted and not any(p.kind=='K' and p.player==hinted for p in observation.pieces.values()):
            from dataclasses import replace
            observation=replace(observation,turn_hint=None,turn_source=None)
        if self.position is not None and key==_signature(engine.pieces_of(self.position)):
            if observation.turn_hint and (not self.turn_known or
                    (not self.history_known and observation.turn_source in ('clock','manual'))):
                if not self.turn_known or self.position.turn!=PLAYERS.index(observation.turn_hint):
                    return 'turn_updated',self.confirm_turn(observation.turn_hint)
            return 'unchanged',None
        # A resignation changes army colour without moving the pieces. Keep
        # unrelated historical rights/EP/points, and do not award fake points
        # by calling Board.eliminate for a visually grey army.
        if self.position is not None:
            old=engine.pieces_of(self.position)
            if (old.keys()==observation.pieces.keys() and
                    all(old[cell].kind==p.kind and (old[cell].player==p.player or p.player=='dead')
                        for cell,p in observation.pieces.items())):
                position=self.position.copy();retired=set();b,*_=engine.modules()
                for (c,r),piece in observation.pieces.items():
                    if piece.player=='dead' and old[c,r].player in PLAYERS:
                        retired.add(PLAYERS.index(old[c,r].player))
                        position.sq[b.sq_of(c,13-r)]=b.make_piece(b.DEAD_UNKNOWN,b.TYPE_CHARS.index(piece.kind))
                position.find_kings();position.alive=[k>=0 for k in position.kings]
                if not any(position.alive):return 'retry',None
                for seat in retired:position.ck[seat]=position.cq[seat]=False
                if not position.alive[position.turn]:
                    position.turn=position.next_turn();position.ep=[None]*4
                position.recompute_key();self.position=position
                if observation.turn_hint:self.confirm_turn(observation.turn_hint)
                return 'updated',self.position.copy()
        if observation.pieces==initial_pieces():
            self.position=engine.new_position()
            # The screenshot cannot tell which seat starts a custom session.
            # Rights come from the standard layout, turn needs visual evidence.
            self.turn_known=False;self.history_known=False
            if observation.turn_hint:
                return 'updated',self.confirm_turn(observation.turn_hint)
            return 'unlinked',self.position.copy()
        b,gen,_,_,_,game=engine.modules()
        # Try the real known turn; if unknown, test all living roots. The next
        # move must identify one unique state, not just a legal-looking layout.
        found={};nodes=0;deadline=time.perf_counter()+.18
        if self.position is not None:
            starts=[self.position.copy()]
            if not self.turn_known:
                starts=[]
                for seat in range(4):
                    if self.position.alive[seat]:
                        root=self.position.copy();root.turn=seat;root.recompute_key();starts.append(root)
            exhausted=False
            hints={m.player:m for m in observation.last_moves}
            maximum=4 if len(hints)>=3 else 2
            for depth in range(1,maximum+1):
                def visit(board,left):
                    nonlocal nodes,exhausted
                    if nodes>=1800 or time.perf_counter()>deadline:
                        exhausted=True;return
                    hint=hints.get(PLAYERS[board.turn])
                    for move in gen.gen_legal(board):
                        if len(hints)>=3 and hint:
                            a=b.sq_of(hint.origin[0],13-hint.origin[1])
                            z=b.sq_of(hint.destination[0],13-hint.destination[1])
                            if b.mv_from(move)!=a or b.mv_to(move)!=z:continue
                        nodes+=1
                        child=board.copy();child.make(move)
                        game.resolve(child)
                        if left==1:
                            if _signature(engine.pieces_of(child))==key:
                                if observation.turn_hint and child.turn!=PLAYERS.index(observation.turn_hint):continue
                                # Include all metadata: identical pieces with
                                # different rights/EP/points are ambiguous.
                                found[child.to_fen4()]=child
                        elif sum(child.alive)>1:visit(child,left-1)
                        if exhausted:return
                for root in starts:
                    visit(root,depth)
                    if exhausted:break
                if found or exhausted:break
            if len(found)==1 and not exhausted:
                self.position=next(iter(found.values()));self.turn_known=True
                return 'updated',self.position.copy()
            if found or (exhausted and self.observations<4):return 'retry',None
            old=engine.pieces_of(self.position)
            changed={cell for cell in set(old)|set(observation.pieces) if old.get(cell)!=observation.pieces.get(cell)}
            if changed and all(cell not in observation.pieces for cell in changed):
                # A lifted/temporarily hidden piece without a destination is
                # not a complete board, even when every king remains visible.
                return 'retry',None
            changed_armies={p.player for cell in changed for p in (old.get(cell),observation.pieces.get(cell)) if p}
            if (self.turn_known and len(changed_armies)==1 and PLAYERS[self.position.turn] not in changed_armies
                    and not (observation.turn_source=='clock' and not self.history_known)):
                # A projection of another player's pending move is not a
                # completed legal turn. Do not bypass history via resync.
                return 'retry',None
        # Unlinked stable layout: replicate it without inventing historical
        # rights, EP, points earned during missed moves, or a certain turn.
        from .ffa_visual_cues import position_from_pieces
        position=position_from_pieces(observation.pieces)
        if self.position is not None:position.points=list(self.position.points)
        # Missing kings can be eliminated players OR a bad frame. An image
        # alone cannot settle that; retain the last confirmed game.
        neutral_kings=sum(p.kind=='K' and p.player=='dead' for p in observation.pieces.values())
        known_missing=0
        if self.position is not None:
            previous=engine.pieces_of(self.position)
            # A previously dead king can be captured like another obstacle.
            # A newly missing live king still indicates an incomplete frame.
            known_missing=sum(p.kind=='K' and p.player=='dead' and cell not in observation.pieces
                              for cell,p in previous.items())
            known_missing=max(known_missing,4-sum(p.kind=='K' for p in previous.values()))
        if not any(position.alive) or sum(position.alive)+neutral_kings+known_missing!=4:return 'retry',None
        for player in PLAYERS:
            if sum(p.player==player and p.kind=='K' for p in observation.pieces.values())>1:
                return 'retry',None
        position.turn=next(i for i,alive in enumerate(position.alive) if alive)
        position.recompute_key()
        self.previous=self.position
        self.position=position;self.turn_known=False;self.history_known=False
        if observation.turn_hint:return 'updated',self.confirm_turn(observation.turn_hint)
        return 'unlinked',position.copy()
