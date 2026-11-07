"""Local Tetrarch FFA adapter. Never passes a four-player board to Stockfish."""
from dataclasses import dataclass
from functools import lru_cache
import sys
import threading
from .paths import resource_root
from .four_player import FourPiece, PLAYERS

_lock = threading.Lock()
_ready = False


@lru_cache(maxsize=1)
def modules():
    folder = resource_root()/'engines'/'tetrarch'
    if not (folder/'tetrarch'/'board.py').is_file():
        raise RuntimeError('Faltan los recursos locales de Tetrarch.')
    sys.path.insert(0,str(folder))
    from tetrarch import board, movegen, core, search, nnue, game
    return board, movegen, core, search, nnue, game


def new_position():
    b,*_ = modules()
    return b.start_board('modern',b.MODE_FFA)


def parse_position(text):
    b,*_ = modules()
    position = b.Board.from_fen4(text,b.MODE_FFA)
    for seat in range(4):
        if position.alive[seat] and position.kings[seat] < 0:
            raise ValueError('Falta el rey de un jugador activo en la FEN4.')
    if not position.alive[position.turn]:
        raise ValueError('La FEN4 asigna el turno a un jugador eliminado.')
    return position


def pieces_of(position):
    b,*_ = modules()
    return {(b.file_of(sq),13-b.rank_of(sq)):
            FourPiece(PLAYERS[b.PC_COLOR[p]] if b.PC_COLOR[p]<4 and position.alive[b.PC_COLOR[p]] else 'dead',
                      b.TYPE_CHARS[b.PC_TYPE[p]])
            for sq,p in enumerate(position.sq) if p and b.VALID[sq]}


def play_move(position,text):
    b,gen,_,_,_,game = modules()
    moves = {b.move_str(m):m for m in gen.gen_legal(position)}
    if text.strip() not in moves:
        raise ValueError('Jugada no legal para el turno actual. Ejemplo de formato: h2h3.')
    result=position.copy()
    result.make(moves[text.strip()])
    game.resolve(result)
    return result


@dataclass(frozen=True)
class FFAResult:
    player: str
    move: str
    origin: tuple[int,int]
    destination: tuple[int,int]
    score: int
    depth: int
    seconds: float
    nodes: int


class AnalysisCancelled(Exception):
    pass


def analyze(position, milliseconds=1000, cancel=None, progress=None):
    """Serialize the native singleton; bounded iterations allow safe cancellation.

    The score is the root seat's paranoid utility, NOT white centipawns or a
    probability of winning. Upstream FFA only supports a single real PV.
    """
    global _ready
    cancel=cancel or threading.Event()
    while not _lock.acquire(timeout=.05):
        if cancel.is_set():raise AnalysisCancelled()
    try:
        if cancel.is_set():raise AnalysisCancelled()
        b,gen,core,search,nnue,_ = modules()
        if not _ready:
            core.load()
            core.set_hash(64)
            core.load_net(nnue.Net.load(str(resource_root()/'engines'/'tetrarch'/'nets'/'net-ffa1.nnue')))
            _ready=True
        if position.mode != b.MODE_FFA:
            raise ValueError('Esta pestaña solo analiza todos contra todos.')
        legal=set(gen.gen_legal(position))
        if not legal:
            raise ValueError('No hay jugadas legales para el jugador seleccionado.')
        snapshot=position.copy()
        latest=None
        def publish(r):
            nonlocal latest
            if cancel.is_set():raise AnalysisCancelled()
            if r.best not in legal:
                raise RuntimeError('Tetrarch devolvió una jugada que no es legal en esta posición.')
            a,z=b.mv_from(r.best),b.mv_to(r.best)
            latest=FFAResult(PLAYERS[snapshot.turn],b.move_str(r.best),
                             (b.file_of(a),13-b.rank_of(a)),(b.file_of(z),13-b.rank_of(z)),
                             r.score,r.depth,r.elapsed,r.nodes)
            if progress:progress(latest)
        search.search(snapshot,search.Limits(movetime=milliseconds,max_depth=8),publish)
        if cancel.is_set():raise AnalysisCancelled()
        if latest is None:raise RuntimeError('Tetrarch no completó ninguna profundidad de búsqueda.')
        return latest
    finally:
        _lock.release()
