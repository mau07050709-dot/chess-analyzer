"""Detect game boundaries and view changes before applying chess transitions."""
from dataclasses import dataclass
import chess
from .recognition import (RecognitionResult,remap_recognition,
                          choose_recognition_orientation,board_from_placement)
from .live_board import legal_turn_candidates


@dataclass(frozen=True)
class SessionDecision:
    event: str
    result: RecognitionResult
    observations: int=0
    message: str=""


class LiveSessionWatch:
    """Require stable evidence; never mutate the tracker during a view probe."""
    def __init__(self):
        self.clear()

    def clear(self):
        self._candidate=None
        self._count=0

    def _pending(self,event,result,visual_stable,needed,message):
        key=(event,result.orientation,result.piece_placement)
        if self._candidate==key:
            if visual_stable:self._count+=1
        else:self._candidate=key;self._count=1
        if not visual_stable or self._count<needed:
            return SessionDecision("waiting",result,self._count,"Comprobando una transición estable de partida/orientación…")
        return SessionDecision(event,result,self._count,message)

    def inspect(self,result,tracker,orientation,*,visual_stable):
        if result.uncertain_squares:
            self.clear()
            return SessionDecision("normal",result)
        current=remap_recognition(result,orientation)
        opposite=remap_recognition(current,"black" if orientation=="white" else "white")
        old=tracker.board.board_fen()
        start=chess.Board().board_fen()
        # A standard initial position is stronger evidence than a layout prior.
        for candidate in (current,opposite):
            if candidate.piece_placement==start and old!=start:
                # Knights can return to the opening layout in the SAME game.
                # A legal explanation takes precedence over a board-only reset.
                if tracker.explains_placement(start):
                    if candidate.orientation==orientation:
                        self.clear()
                        return SessionDecision("normal",current)
                    return self._pending("orientation",candidate,visual_stable,2,
                                         "Giro con regreso legal a la colocación inicial; se conserva la historia.")
                return self._pending("new_game",candidate,visual_stable,2,
                                     "Nueva partida: posición inicial estándar confirmada.")
        if current.piece_placement==old:
            self.clear()
            return SessionDecision("normal",current)
        if opposite.piece_placement==old:
            return self._pending("orientation",opposite,visual_stable,2,
                                 "Orientación invertida; se conserva la partida y su turno.")
        target=board_from_placement(current.piece_placement)
        reverse=board_from_placement(opposite.piece_placement)
        distance=sum(target.piece_at(sq)!=tracker.board.piece_at(sq) for sq in chess.SQUARES)
        reverse_distance=sum(reverse.piece_at(sq)!=tracker.board.piece_at(sq) for sq in chess.SQUARES)
        # Normal moves/castling keep the fast path. Check both mappings when the
        # board delta or a much closer opposite view warrants a session probe.
        suspect=(distance>4 or reverse_distance<distance
                 or len(target.piece_map())>len(tracker.board.piece_map()))
        if not suspect or tracker.explains_placement(current.piece_placement):
            self.clear()
            return SessionDecision("normal",current)
        if tracker.explains_placement(opposite.piece_placement):
            return self._pending("orientation",opposite,visual_stable,2,
                                 "Giro con jugadas enlazadas por el historial legal.")
        if distance<=4:
            self.clear()
            return SessionDecision("normal",current)
        # Unlinked captures may be another game or a long missed interval. Only
        # stable, structurally legal images qualify; game history stays unknown.
        white=remap_recognition(current,"white")
        black=remap_recognition(current,"black")
        legal=[candidate for candidate in (white,black)
               if legal_turn_candidates(candidate.piece_placement)]
        if not legal:
            self.clear()
            return SessionDecision("normal",current)
        chosen=legal[0] if len(legal)==1 else choose_recognition_orientation(white,black)
        return self._pending("unlinked_game",chosen,visual_stable,3,
                             "Posición estable sin enlace con la partida anterior; historial desconocido.")
