"""160-cell ONNX shape recognition plus conservative four-colour evidence.

The existing model/session is reused; its white/black probabilities are summed
by piece type. Army colour is measured independently from foreground hue.
"""
from dataclasses import dataclass
import time
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from .capture.board_region import rgb
from .four_player import FourPiece, PLAYERS, playable, initial_pieces, display_cell
from .recognition import _load_classifier, MODEL_CLASSES
from .paths import resource_root


def canonical_cell(column,row,bottom):
    for _ in range(PLAYERS.index(bottom)):
        column,row=13-row,column
    return column,row


@dataclass(frozen=True)
class FFAObservation:
    pieces: dict
    uncertain: tuple
    bottom: str
    milliseconds: float
    transient: bool = False
    last_moves: tuple = ()
    clock_samples: tuple = ()
    turn_hint: str | None = None
    turn_source: str | None = None
    board_rect: tuple | None = None

    @property
    def signature(self):
        return tuple(sorted((c,r,p.player,p.kind) for (c,r),p in self.pieces.items()))


def _army(tile):
    # Border samples estimate THIS square's background, including highlights.
    border=np.concatenate((tile[:3].reshape(-1,3),tile[-3:].reshape(-1,3),
                           tile[:,:3].reshape(-1,3),tile[:,-3:].reshape(-1,3)))
    background=np.median(border,axis=0)
    a=tile[4:28,4:28].astype(np.float32)/255
    hi=a.max(2);lo=a.min(2);delta=hi-lo
    mask=(delta/np.maximum(hi,.001)>.28)&(np.linalg.norm(a-background/255,axis=2)>.12)
    if mask.sum()<12:return None
    # HSV hue, with ties irrelevant because grayscale pixels are masked out.
    h=np.zeros_like(hi)
    red=hi==a[:,:,0];green=(hi==a[:,:,1])&~red;blue=~red&~green
    d=np.maximum(delta,.001)
    h[red]=((a[:,:,1]-a[:,:,2])/d)[red]%6
    h[green]=((a[:,:,2]-a[:,:,0])/d+2)[green]
    h[blue]=((a[:,:,0]-a[:,:,1])/d+4)[blue]
    h/=6
    counts=np.array([np.sum(mask&((h<.065)|(h>.94))),
                     np.sum(mask&(h>.50)&(h<.72)),
                     np.sum(mask&(h>.08)&(h<.20)),
                     np.sum(mask&(h>.23)&(h<.48))])
    winner=int(counts.argmax())
    if counts[winner]/mask.sum()>=.75:return PLAYERS[winner]
    # A check halo occupies the OUTER part of a square. Require the same
    # plurality in the central half of the tile; never switch colours because
    # the center disagrees. This removes halo contamination, not piece pixels.
    center=mask.copy();center[:4]=False;center[-4:]=False;center[:,:4]=False;center[:,-4:]=False
    if center.sum()>=12:
        central=np.array([np.sum(center&((h<.065)|(h>.94))),
                          np.sum(center&(h>.50)&(h<.72)),np.sum(center&(h>.08)&(h<.20)),
                          np.sum(center&(h>.23)&(h<.48))])
        if central.argmax()==winner and central[winner]/center.sum()>=.75:return PLAYERS[winner]
    return None


def _gray_piece(tile):
    """A neutral silhouette, not an empty gray square or lost colour label."""
    background=np.median(np.concatenate((tile[:3].reshape(-1,3),tile[-3:].reshape(-1,3))),axis=0)
    center=tile[4:28,4:28].astype(float)
    foreground=center[np.linalg.norm(center-background,axis=2)>35]
    return (len(foreground)>=16 and
            np.mean(foreground.max(1)-foreground.min(1)<24)>=.85)


def _shape_scores(probabilities):
    return np.stack([probabilities[:,0]]+[
        probabilities[:,MODEL_CLASSES.index(k)]+probabilities[:,MODEL_CLASSES.index(k.lower())]
        for k in 'KQRBNP'],axis=1)


def _rescue_shapes(session,pixels,scores):
    """Retry doubtful shapes with the same two exposure views as classic mode.

    Both views must independently exceed .70 and agree with the original
    shape. This cannot relabel a piece, infer from its square, or change the
    colour/background evidence. Confident cells incur no additional inference.
    """
    labels=scores.argmax(1);doubts=np.flatnonzero(scores.max(1)<.70)
    if not len(doubts):return scores
    values=pixels[doubts].astype(np.float32)/255
    views=np.concatenate((values*values,np.clip(values*1.275,0,1)))
    tensor=np.ascontiguousarray((views@np.array([.299,.587,.114],np.float32)).reshape(-1,1024))
    probabilities=np.asarray(session.run(None,{session.get_inputs()[0].name:tensor})[0])
    if probabilities.shape!=(2*len(doubts),13) or not np.isfinite(probabilities).all():return scores
    first,second=np.split(_shape_scores(probabilities),2)
    result=scores.copy()
    for j,i in enumerate(doubts):
        label=labels[i]
        if first[j].argmax()==second[j].argmax()==label and min(first[j,label],second[j,label])>=.70:
            result[i]=first[j] if first[j,label]<second[j,label] else second[j]
    return result


def recognize_ffa(image:QImage,bottom='red',board_rect=None,previous_pieces=None):
    started=time.perf_counter()
    if image.isNull() or min(image.width(),image.height())<280:
        raise ValueError('Selecciona el tablero completo de 14 × 14, de al menos 280 píxeles.')
    from .ffa_region import locate
    rect=locate(image,board_rect)
    if rect is None:
        raise ValueError(f'No se encontró la cruz completa en la selección de {image.width()} × {image.height()} px. '
                         'Selecciona las 14 filas y columnas, incluidas las esquinas vacías; puede haber margen alrededor.')
    geometry=(rect.x(),rect.y(),rect.width(),rect.height())
    # Intersect explicitly: QImage.copy outside the image would silently add
    # black pixels. Keep the measured grid spacing for the existing pixels.
    image=image.copy(rect.intersected(image.rect()))
    cells=[(c,r) for r in range(14) for c in range(14) if playable(c,r)]
    tiles=[]
    for c,r in cells:
        x0,x1=round(c*rect.width()/14),min(image.width(),round((c+1)*rect.width()/14))
        y0,y1=round(r*rect.height()/14),min(image.height(),round((r+1)*rect.height()/14))
        view=image.copy(x0,y0,x1-x0,y1-y0).scaled(32,32,Qt.AspectRatioMode.IgnoreAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
        tiles.append(rgb(view))
    pixels=np.stack(tiles)
    borders=np.median(np.stack([pixels[:,:3].reshape(160,-1,3),pixels[:,-3:].reshape(160,-1,3)],axis=1).reshape(160,-1,3),axis=1)
    # Two red outlier backgrounds can indicate a premove/selection overlay.
    # Red PIECES are central foreground and do not satisfy this border test.
    reds=(borders[:,0]>borders[:,1]*1.35)&(borders[:,0]>borders[:,2]*1.35)
    outliers=np.zeros(160,dtype=bool)
    for parity in (0,1):
        indexes=np.array([(c+r)%2==parity for c,r in cells])
        baseline=np.median(borders[indexes],axis=0)
        outliers[indexes]=np.linalg.norm(borders[indexes]-baseline,axis=1)>40
    transient=np.count_nonzero(reds&outliers)>=2
    tensor=np.ascontiguousarray((pixels.astype(np.float32)@np.array([.299,.587,.114],dtype=np.float32)
                                /255).reshape(160,1024))
    session=_load_classifier(str((resource_root()/'models'/'chess-tiles-v2.onnx').resolve()))
    probabilities=np.asarray(session.run(None,{session.get_inputs()[0].name:tensor})[0])
    if probabilities.shape!=(160,13) or not np.isfinite(probabilities).all():
        raise ValueError('El modelo no devolvió las 160 clasificaciones completas.')
    kinds='KQRBNP'
    scores=_rescue_shapes(session,pixels,_shape_scores(probabilities))
    pieces={};uncertain=[];gray=[]
    for i,(c,r) in enumerate(cells):
        label=int(scores[i].argmax());confidence=float(scores[i,label])
        cell=canonical_cell(c,r,bottom)
        if label==0:
            if confidence<.70:uncertain.append(cell)
            continue
        kind=label-1
        player=_army(pixels[i])
        if player is None and confidence>=.70 and _gray_piece(pixels[i]):
            gray.append(cell);player='dead'
        if confidence<.70 or player is None:uncertain.append(cell)
        if player:pieces[cell]=FourPiece(player,kinds[kind])
    # A single desaturated piece can be an animation. Multiple consistent
    # neutral silhouettes are the visible inactive army, not missing pieces.
    from .ffa_visual_cues import orient,last_moves,clocks
    supplied=bottom
    pieces,bottom=orient(pieces,bottom,canonical_cell)
    if len(gray)<3:
        for c,r in gray:
            cell=canonical_cell(*display_cell(c,r,supplied),bottom)
            if not previous_pieces or previous_pieces.get(cell)!=pieces.get(cell):
                uncertain.append((c,r))
    uncertain=tuple(canonical_cell(*display_cell(c,r,supplied),bottom) for c,r in uncertain)
    moves=last_moves(cells,borders,pieces,bottom,canonical_cell)
    if transient:
        red_move=next((m for m in moves if m.player=='red'),None)
        if red_move:
            marked=[canonical_cell(*cells[i],bottom) for i in np.flatnonzero(reds&outliers)]
            if all(cell in (red_move.origin,red_move.destination) or
                   (pieces.get(cell) is not None and pieces[cell].kind=='K') for cell in marked):
                transient=False
    hint=None
    if len(moves)==1 and moves[0].reverse_legal:
        alive={p.player for p in pieces.values() if p.kind=='K' and p.player in PLAYERS}
        seat=PLAYERS.index(moves[0].player)
        hint=next((PLAYERS[(seat+n)%4] for n in range(1,5) if PLAYERS[(seat+n)%4] in alive),None)
    return FFAObservation(pieces,uncertain,bottom,(time.perf_counter()-started)*1000,
                          transient,moves,clocks(image),hint,'last_move' if hint else None,geometry)
