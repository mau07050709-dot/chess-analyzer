"""Stable live-board synchronization with short-history and visual recovery."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import chess

from .recognition import RecognitionResult, board_from_placement


@dataclass(frozen=True)
class LiveBoardUpdate:
    status: str
    board: chess.Board
    move: chess.Move | None = None
    message: str = ""
    moves: tuple[chess.Move, ...] = ()


class LiveBoardTracker:
    """Accept stable legal transitions and conservatively recover short gaps."""

    def __init__(self, board: chess.Board, stable_observations: int = 2,
                 max_recovery_plies: int = 3, review_after_uncertain: int = 4,
                 turn_known: bool = True) -> None:
        if not board.is_valid():
            raise ValueError("La posición inicial del seguimiento no es legal.")
        if stable_observations < 2:
            raise ValueError("Se requieren al menos dos capturas estables.")
        self.board = board.copy(stack=True)
        self._turn_candidates = ({board.turn} if turn_known else
                                 {candidate.turn for candidate in legal_turn_candidates(board.board_fen())})
        if not self._turn_candidates:
            raise ValueError("No hay un turno legal compatible con la posición inicial.")
        # A visual midgame anchor lacks historical castling evidence. This
        # remains distinct from a trusted FEN explicitly forbidding castling.
        self._castling_history_unknown = not turn_known
        self.stable_observations = stable_observations
        self.max_recovery_plies = max(1, min(3, max_recovery_plies))
        self.review_after_uncertain = max(2, review_after_uncertain)
        self._candidate: str | None = None
        self._candidate_count = 0
        self._uncertain_count = 0
        self._recent_confirmed = deque(maxlen=6)
        self._visual_reanchor = False
        self._saw_castling_partial = False
        self._pending_transition_roots = []
        self._pending_anchor_fen = None

    @property
    def turn_known(self) -> bool:
        return len(self._turn_candidates) == 1

    def confirm_visual_turn(self, turn: bool, expected_fen: str) -> bool:
        """Confirm only an unknown turn for the exact current visual position.

        Legality/color evidence and temporal agreement are checked by the
        highlight recovery caller. No move/history or special metadata is added.
        """
        if self.turn_known or expected_fen != self.board.fen() or turn not in self._turn_candidates:
            return False
        candidate=self.board.copy(stack=True);candidate.turn=turn
        if not candidate.is_valid():return False
        self.board=candidate
        self._turn_candidates={turn}
        self._reset_candidate()
        return True

    def discard_pending_candidate(self) -> None:
        """Forget visual intermediates while retaining the confirmed chess state."""
        self._reset_candidate()

    def observe(self, result: RecognitionResult, *, visual_stable: bool = True) -> LiveBoardUpdate:
        """Wait out animation frames, then accept a uniquely matching legal move."""
        if result.uncertain_squares:
            self._reset_candidate()
            self._uncertain_count += 1
            squares = ", ".join(result.uncertain_squares[:8])
            more = "…" if len(result.uncertain_squares) > 8 else ""
            status = "review" if self._uncertain_count >= self.review_after_uncertain else "retry"
            message = f"Casillas dudosas: {squares}{more}. Se conserva la última posición confirmada."
            if status == "retry":
                message += " Se volverá a capturar automáticamente."
            else:
                message += " Revisa la posición o reanuda para volver a intentar."
            return LiveBoardUpdate(status, self.board.copy(stack=True), message=message)
        self._uncertain_count = 0
        try:
            observed = board_from_placement(result.piece_placement)
        except ValueError as exc:
            self._reset_candidate()
            return self._retry_update(str(exc))
        if (len(observed.pieces(chess.KING, chess.WHITE)) != 1
                or len(observed.pieces(chess.KING, chess.BLACK)) != 1):
            self._reset_candidate()
            return self._retry_update("La captura no muestra exactamente un rey de cada color.")
        placement = observed.board_fen()
        if placement == self.board.board_fen():
            self._reset_candidate()
            self._pending_transition_roots=[]
            return LiveBoardUpdate("unchanged", self.board.copy(stack=True), message="La posición no cambió.")
        if sum(observed.piece_at(sq) != self.board.piece_at(sq) for sq in chess.SQUARES) == 1:
            self._reset_candidate()
            return self._retry_update("Un cambio aislado de una casilla no demuestra una jugada completa.")
        # Castling animations may settle the king and rook on separate frames.
        # Neither half is a new game position: keep the full confirmed history.
        for move in self.board.legal_moves:
            if not self.board.is_castling(move):
                continue
            final = self.board.copy(stack=False)
            final.push(move)
            changed = [sq for sq in chess.SQUARES
                       if self.board.piece_at(sq) != final.piece_at(sq)]
            if (placement != final.board_fen()
                    and all(observed.piece_at(sq) in (self.board.piece_at(sq), final.piece_at(sq))
                            for sq in changed)
                    and all(observed.piece_at(sq) == self.board.piece_at(sq)
                            for sq in chess.SQUARES if sq not in changed)):
                self._note_castling_partial(self.board,observed,())
                self._reset_candidate()
                return LiveBoardUpdate("waiting", self.board.copy(stack=True), message=(
                    "La captura puede ser un enroque incompleto; se conserva el estado hasta "
                    "el enroque completo o una continuación legal que descarte esa animación."
                ))
        if placement == self._candidate:
            if visual_stable:
                self._candidate_count += 1
        else:
            self._candidate = placement
            self._candidate_count = 1
        needed = self.stable_observations
        if not visual_stable or self._candidate_count < needed:
            return LiveBoardUpdate("waiting", self.board.copy(stack=True),
                                   message="Esperando que termine la transición visual.")

        paths = self._find_paths(placement)
        if self._saw_castling_partial:
            self._reset_candidate()
            return LiveBoardUpdate("waiting", self.board.copy(stack=True), message=(
                "Una secuencia legal reciente explica un enroque todavía parcial; se conserva la posición."
            ))
        if len(paths) != 1:
            merged = self._merge_legal_paths(paths, placement)
            if merged is not None:
                self._remember_confirmed()
                self.board = merged
                self._visual_reanchor = False
                self._turn_candidates = {merged.turn}
                self._reset_candidate()
                return LiveBoardUpdate("resynced", merged.copy(stack=True), message=(
                    "Varias secuencias legales explican la captura: historial desconocido en su orden, "
                    "pero turno y derechos de enroque conservados según las secuencias compatibles; "
                    "al paso solo si coinciden y contador de medio movimiento conservador."
                ))
            if self._pending_transition_roots and self._pending_anchor_fen==self.board.fen():
                roots=[board for board,prefix in self._pending_transition_roots]
                pending_paths=self._find_paths(placement,roots=roots)
                continued=self._merge_legal_paths(pending_paths,placement,roots=roots)
                if continued is not None:
                    self._remember_confirmed()
                    self.board=continued
                    self._turn_candidates={continued.turn}
                    self._visual_reanchor=False
                    self._pending_transition_roots=[]
                    self._reset_candidate()
                    return LiveBoardUpdate("resynced",continued.copy(stack=True),message=(
                        "Continuación legal de una transición pendiente confirmada; "
                        "se conservan turno y derechos derivados de las hipótesis legales."
                    ))
            if self._visual_reanchor or not self.turn_known:
                for index in reversed(range(len(self._recent_confirmed))):
                    anchor = self._recent_confirmed[index]
                    recovered_paths = self._find_paths(placement, roots=[anchor])
                    recovered = self._merge_legal_paths(recovered_paths, placement, roots=[anchor])
                    if recovered is not None:
                        self.board = recovered
                        self._turn_candidates = {recovered.turn}
                        self._recent_confirmed = deque(list(self._recent_confirmed)[:index+1],maxlen=6)
                        self._visual_reanchor = False
                        self._reset_candidate()
                        return LiveBoardUpdate("resynced",recovered.copy(stack=True),message=(
                            "Posición recuperada desde un estado confirmado reciente; "
                            "turno y metadatos derivados de secuencias legales conocidas."
                        ),moves=tuple(recovered_paths[0]) if len(recovered_paths)==1 else ())
            # A midgame anchor may have discarded rights before an actual
            # castle. Its complete four-square transition still proves the next
            # turn; do not leave the ordinary analysis flow waiting for it.
            observed_castle = self._recover_observed_castle(placement) if not paths else None
            if observed_castle is not None:
                self._remember_confirmed()
                self.board = observed_castle
                self._turn_candidates = {observed_castle.turn}
                self._visual_reanchor = True
                self._reset_candidate()
                return LiveBoardUpdate("resynced", observed_castle.copy(stack=True), message=(
                    "Enroque completo observado; turno siguiente confirmado. "
                    "No se restauran derechos históricos desconocidos."
                ))
            # The screen shows the current placement, not necessarily every move
            # that led to it. If legal history cannot be recovered, re-anchor on
            # a stable, high-confidence legal position while discarding metadata
            # that cannot be justified from pixels (castling, EP, clocks).
            self._remember_confirmed()
            recovered, turn_known = self._reanchor(placement)
            if recovered is None:
                self._reset_candidate()
                reason = ("La posición observada no es legal; se conserva la anterior."
                          if not paths else "La posición admite varias historias y no pasa la validación legal.")
                return LiveBoardUpdate("review", self.board.copy(stack=True), message=reason)
            self.board = recovered
            self._visual_reanchor = True
            self._reset_candidate()
            detail = "turno deducido por legalidad" if turn_known else "turno provisional; verifica si fuera necesario"
            return LiveBoardUpdate(
                "resynced", recovered.copy(stack=True), message=(
                    "Sincronización recuperada desde la posición estable observada; "
                    "historial desconocido, enroque y al paso reiniciados, contadores conservadores; "
                    f"{detail}."
                ),
            )
        path = paths[0]
        roots = self._root_positions()
        accepted_roots: list[chess.Board] = []
        for root in roots:
            candidate = root.copy(stack=True)
            try:
                for move in path:
                    if move not in candidate.legal_moves:
                        raise ValueError
                    candidate.push(move)
            except ValueError:
                continue
            if candidate.board_fen() == placement:
                accepted_roots.append(candidate)
        if len(accepted_roots) != 1:
            self._remember_confirmed()
            recovered, _turn_known = self._reanchor(placement)
            if recovered is None:
                self._reset_candidate()
                return LiveBoardUpdate("review", self.board.copy(stack=True),
                                       message="La transición no permite determinar un turno legal único.")
            self.board = recovered
            self._visual_reanchor = True
            self._reset_candidate()
            return LiveBoardUpdate("resynced", recovered.copy(stack=True), message=(
                "Posición recuperada, pero el historial no determina el turno; "
                "se conserva como provisional y Stockfish espera nueva evidencia."
            ))
        accepted = accepted_roots[0]
        san_moves: list[str] = []
        # Reconstruct SAN from the uniquely matching root without losing the
        # metadata and history carried by a confirmed position.
        root = next(root.copy(stack=True) for root in roots
                    if self._path_reaches(root, path, placement))
        for move in path:
            san_moves.append(root.san(move))
            root.push(move)
        if not accepted.is_valid():
            self._reset_candidate()
            return LiveBoardUpdate("review", self.board.copy(stack=True),
                                   message="La transición detectada produciría una posición ilegal.")
        self._remember_confirmed()
        self.board = accepted
        self._visual_reanchor = False
        self._turn_candidates = {accepted.turn}
        self._reset_candidate()
        return LiveBoardUpdate(
            "updated", self.board.copy(stack=True), path[-1],
            f"Movimiento(s) legal(es) detectado(s): {' '.join(san_moves)}.", tuple(path),
        )

    def _recover_observed_castle(self, placement: str) -> chess.Board | None:
        """Recover only a complete castle after rights were lost visually.

        The temporary right is a geometry/attack-path probe, never a restored
        historical right. Known legal paths have already taken precedence.
        """
        if not self.turn_known or not self._castling_history_unknown:
            return None
        observed = board_from_placement(placement)
        if sum(self.board.piece_at(sq) != observed.piece_at(sq) for sq in chess.SQUARES) != 4:
            return None
        color = self.board.turn
        rank = 0 if color == chess.WHITE else 7
        king_from = chess.square(4, rank)
        if self.board.king(color) != king_from:
            return None
        root = self.board.root()
        for rook_file, king_file in ((7, 6), (0, 2)):
            rook_from = chess.square(rook_file, rank)
            if self.board.castling_rights & chess.BB_SQUARES[rook_from]:
                continue  # Already handled through the ordinary legal history.
            if (root.king(color) != king_from
                    or root.piece_at(rook_from) != chess.Piece(chess.ROOK, color)
                    or any(move.from_square in (king_from, rook_from)
                           or move.to_square == rook_from for move in self.board.move_stack)):
                continue  # Do not contradict moves known since the visual anchor.
            probe = self.board.copy(stack=False)
            probe.castling_rights |= chess.BB_SQUARES[rook_from]
            move = chess.Move(king_from, chess.square(king_file, rank))
            if not probe.is_valid() or move not in probe.legal_moves or not probe.is_castling(move):
                continue
            probe.push(move)
            if probe.board_fen() == placement:
                probe.castling_rights &= self.board.castling_rights
                probe.clear_stack()  # Do not invent a legally proven history.
                return probe
        return None

    def _reanchor(self, placement: str) -> tuple[chess.Board | None, bool]:
        candidates = list(legal_turn_candidates(placement))
        if not candidates:
            return None, False
        self._castling_history_unknown = True
        self._turn_candidates = {candidate.turn for candidate in candidates}
        if len(candidates) == 1:
            return candidates[0], True
        # A lost-history re-anchor cannot prove the turn. Prefer the opponent of
        # the last confirmed turn (the common one-missed-ply case), and expose
        # that this is provisional in the returned status message.
        preferred_turn = not self.board.turn
        chosen = next((candidate for candidate in candidates if candidate.turn == preferred_turn), candidates[0])
        return chosen, False

    def _find_paths(self, placement: str, roots=None, *, record_partials=True, first_match_only=False) -> list[tuple[chess.Move, ...]]:
        """Find all shortest paths within the budget, so metadata is justified.

        An exhausted budget returns no proof; a partial list must never be used
        to claim that all possible histories agree on castling or side to move.
        """
        target = board_from_placement(placement)
        if record_partials:
            self._saw_castling_partial = False
        frontier: list[tuple[chess.Board, tuple[chess.Move, ...]]] = [
            (root, ()) for root in (self._root_positions() if roots is None else roots)
        ]
        for _depth in range(1, self.max_recovery_plies + 1):
            following: list[tuple[chess.Board, tuple[chess.Move, ...]]] = []
            matches: list[tuple[chess.Move, ...]] = []
            expanded = 0
            for position, path in frontier:
                # A single move changes at most four squares (castling). If the
                # remaining board delta is larger than that bound, this branch
                # cannot reach the observed placement within the ply budget.
                remaining_plies = self.max_recovery_plies - len(path)
                distance = sum(position.piece_at(square) != target.piece_at(square)
                               for square in chess.SQUARES)
                if distance > 4 * remaining_plies:
                    continue
                if record_partials and distance <= 4 and self._is_castling_partial(position, target):
                    self._note_castling_partial(position,target,path)
                    self._saw_castling_partial = True
                    return []
                for move in position.legal_moves:
                    if remaining_plies == 1 and (
                            position.piece_at(move.from_square)==target.piece_at(move.from_square)
                            or position.piece_at(move.to_square)==target.piece_at(move.to_square)):
                        continue
                    expanded += 1
                    if expanded > 15000:
                        return []
                    next_position = position.copy(stack=True)
                    next_position.push(move)
                    next_path = (*path, move)
                    if next_position.board_fen() == placement:
                        matches.append(next_path)
                        if first_match_only:
                            return matches
                    following.append((next_position, next_path))
            if matches:
                return matches
            frontier = following
        return []

    def explains_placement(self, placement: str) -> bool:
        """Read-only history probe used before treating a view as another game."""
        if placement==self.board.board_fen() or self._find_paths(placement,record_partials=False,first_match_only=True):
            return True
        if self._pending_transition_roots and self._pending_anchor_fen==self.board.fen():
            if self._find_paths(placement,roots=[board for board,_ in self._pending_transition_roots],
                                record_partials=False,first_match_only=True):
                return True
        if self._visual_reanchor or not self.turn_known:
            return any(self._find_paths(placement,roots=[anchor],record_partials=False,first_match_only=True)
                       for anchor in reversed(self._recent_confirmed))
        return False

    def observe_stable(self, result: RecognitionResult, observations: int) -> LiveBoardUpdate:
        """Transfer stable evidence already counted by the session/view detector."""
        if observations<self.stable_observations:
            raise ValueError("Faltan observaciones estables de la nueva orientación.")
        self._candidate=result.piece_placement
        self._candidate_count=observations-1
        return self.observe(result,visual_stable=True)

    def _merge_legal_paths(self, paths, placement: str, roots=None) -> chess.Board | None:
        if not paths:
            return None
        states = []
        for root in (self._root_positions() if roots is None else roots):
            for path in paths:
                if self._path_reaches(root, path, placement):
                    state = root.copy(stack=True)
                    for move in path:
                        state.push(move)
                    states.append(state)
        if not states or len({state.turn for state in states}) != 1:
            return None
        merged = states[0].copy(stack=False)
        for state in states[1:]:
            merged.castling_rights &= state.castling_rights
        if len({state.ep_square for state in states}) != 1:
            merged.ep_square = None
        merged.halfmove_clock = min(state.halfmove_clock for state in states)
        merged.fullmove_number = min(state.fullmove_number for state in states)
        return merged if merged.is_valid() else None

    @staticmethod
    def _is_castling_partial(board, observed):
        for move in board.generate_castling_moves():
            final=board.copy(stack=False);final.push(move)
            if observed.board_fen() in (board.board_fen(),final.board_fen()):
                continue
            if all(observed.piece_at(sq) in (board.piece_at(sq),final.piece_at(sq))
                   for sq in chess.SQUARES):
                return True
        return False

    def _remember_confirmed(self):
        if self.turn_known and not self._visual_reanchor:
            if not self._recent_confirmed or self._recent_confirmed[-1].fen()!=self.board.fen():
                self._recent_confirmed.append(self.board.copy(stack=True))
        self._pending_transition_roots=[]

    def _note_castling_partial(self,position,observed,prefix):
        roots=[]
        for move in position.legal_moves:
            candidate=position.copy(stack=True);candidate.push(move)
            if candidate.board_fen()==observed.board_fen() or (
                    position.is_castling(move)
                    and observed.board_fen() not in (position.board_fen(),candidate.board_fen())
                    and all(observed.piece_at(sq) in (position.piece_at(sq),candidate.piece_at(sq))
                            for sq in chess.SQUARES)):
                roots.append((candidate,(*prefix,move)))
        self._pending_transition_roots=roots[:4]
        self._pending_anchor_fen=self.board.fen()

    def reset(self, board: chess.Board) -> None:
        if not board.is_valid():
            raise ValueError("La posición inicial del seguimiento no es legal.")
        self.board = board.copy(stack=True)
        self._turn_candidates = {board.turn}
        self._castling_history_unknown = False
        self._recent_confirmed.clear()
        self._visual_reanchor = False
        self._pending_transition_roots=[]
        self._reset_candidate()

    def _root_positions(self) -> list[chess.Board]:
        if self.turn_known:
            return [self.board.copy(stack=True)]
        roots = []
        for turn in sorted(self._turn_candidates, reverse=True):
            root = chess.Board(f"{self.board.board_fen()} {'w' if turn else 'b'} - - 0 1")
            if root.is_valid():
                roots.append(root)
        return roots

    @staticmethod
    def _path_reaches(root: chess.Board, path: tuple[chess.Move, ...], placement: str) -> bool:
        candidate = root.copy(stack=True)
        for move in path:
            if move not in candidate.legal_moves:
                return False
            candidate.push(move)
        return candidate.board_fen() == placement

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._candidate_count = 0

    def _retry_update(self, message: str) -> LiveBoardUpdate:
        self._uncertain_count += 1
        status = "review" if self._uncertain_count >= self.review_after_uncertain else "retry"
        if status == "retry":
            message += " Se conserva la última posición confirmada y se volverá a capturar."
        return LiveBoardUpdate(status, self.board.copy(stack=True), message=message)


def legal_turn_candidates(placement: str) -> tuple[chess.Board, ...]:
    """Return only side-to-move candidates consistent with python-chess legality."""
    candidates = []
    for turn in (chess.WHITE, chess.BLACK):
        board = chess.Board(f"{placement} {'w' if turn else 'b'} - - 0 1")
        if board.is_valid():
            candidates.append(board)
    return tuple(candidates)
