"""Find the 14x14 cross INSIDE the user's selection; never pad missing pixels."""
import itertools
import numpy as np
from PySide6.QtCore import QRect, Qt
from .capture.board_region import rgb
from .four_player import playable


CELLS=np.array([(c,r) for r in range(14) for c in range(14) if playable(c,r)])


def grid_score(image,rect):
    if rect.width()<280 or rect.height()<280 or _visible(image,rect) is None:return 0.
    pixels=rgb(image.copy(rect.intersected(image.rect()))).astype(float)
    samples=[];valid=[]
    for dy,dx in ((.07,.07),(.07,.93),(.93,.07),(.93,.93)):
        ys=((CELLS[:,1]+dy)*rect.height()/14).astype(int)
        xs=((CELLS[:,0]+dx)*rect.width()/14).astype(int)
        mask=(ys<pixels.shape[0])&(xs<pixels.shape[1]);values=np.zeros((160,3))
        values[mask]=pixels[ys[mask],xs[mask]];samples.append(values);valid.append(mask)
    samples=np.stack(samples);valid=np.stack(valid)
    parity=CELLS.sum(1)%2
    colors=np.array([np.median(samples[:,parity==p].reshape(-1,3),axis=0) for p in (0,1)])
    if np.linalg.norm(colors[0]-colors[1])<35:return 0.
    return float(np.mean((np.linalg.norm(samples-colors[parity],axis=2)<28)&valid))


def _visible(image,rect):
    # A few clipped background pixels at the outside edge need not discard
    # all 160 cells. Never synthesize them. Larger missing rows fail safe.
    visible=rect.intersected(image.rect())
    # One pixel accounts for integer rounding of the fitted grid at a new DPI.
    tolerance=min(rect.width(),rect.height())/14*.12+1
    if (rect.x()<0 or rect.y()<0 or rect.width()-visible.width()>tolerance or
            rect.height()-visible.height()>tolerance):return None
    return visible


def _labels(pixels,colors):
    distance=np.linalg.norm(pixels[...,None,:].astype(float)-colors,axis=-1)
    result=distance.argmin(-1).astype(np.int8)
    result[distance.min(-1)>24]=-1
    return result


def _runs(line):
    # Resizing introduces one or two blended pixels at colour boundaries.
    # Bridge only those tiny gaps, never piece bodies or missing squares.
    line=line.copy()
    gaps=np.r_[0,np.flatnonzero(line[1:]!=line[:-1])+1,len(line)]
    for i in np.flatnonzero((line[gaps[:-1]]<0)&(np.diff(gaps)<=2)):
        start,end=gaps[i:i+2]
        if start>0 and end<len(line) and line[start-1]>=0 and line[end]>=0:line[start:end]=line[start-1]
    edges=np.r_[0,np.flatnonzero(line[1:]!=line[:-1])+1,len(line)]
    values=line[edges[:-1]]
    if len(values)<14:return
    windows=np.lib.stride_tricks.sliding_window_view(values,14)
    valid=np.all(windows>=0,axis=1)&np.all(windows[:,1:]!=windows[:,:-1],axis=1)
    for i in np.flatnonzero(valid):
        boundaries=edges[i:i+15];lengths=np.diff(boundaries);cell=float(np.median(lengths[1:-1]))
        if cell<7 or np.max(np.abs(lengths[1:-1]-cell))>max(2,cell*.10):continue
        if min(lengths[0],lengths[-1])<cell*.85:continue
        step=float(np.dot(np.arange(-6,7),boundaries[1:14])/182)
        yield round(float(np.mean(boundaries[1:14]))-7*step),round(14*step)


def locate(image,hint=None):
    """Return only a complete, verified grid. Ambiguous separate boards fail safe.

    Most cycles verify the cached rectangle or exact square without scanning.
    Relocation scans background runs; it does not invoke ONNX or capture again.
    """
    if image.isNull():return None
    if hint is not None:
        rect=QRect(*hint)
        if grid_score(image,rect)>=.80:return rect
    if abs(image.width()-image.height())<=max(2,min(image.width(),image.height())*.015):
        if grid_score(image,image.rect())>=.80:return image.rect()
    scale=min(1.,900/max(image.width(),image.height()))
    small=image.scaled(round(image.width()*scale),round(image.height()*scale),
                       Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
    pixels=rgb(small)
    # Quantized histogram supplies actual theme colours, not fixed gray RGBs.
    quant=(pixels//8).astype(np.int32)
    keys=quant[...,0]*1024+quant[...,1]*32+quant[...,2]
    counts=np.bincount(keys.ravel(),minlength=32768)
    keys=np.argsort(counts)[-8:][::-1]
    colors=np.array([(k//1024,(k//32)%32,k%32) for k in keys],dtype=float)*8+3.5
    candidates=[];native=None
    for a,b in itertools.combinations(colors,2):
        if not 35<=np.linalg.norm(a-b)<=340:continue
        labels=_labels(pixels,np.array([a,b]))
        seen=set();native_labels=None
        for y in range(0,small.height(),3):
            for left,side in _runs(labels[y]):
                key=(left//2,side//2)
                if key in seen or left<0 or left+side>small.width():continue
                seen.add(key)
                # All central files cross 14 rows. Sample near square edges
                # so piece bodies and coordinate glyphs do not set geometry.
                proposals=set()
                for c in range(3,11):
                    for fraction in (.07,.93):
                        x=round(left+(c+fraction)*side/14)
                        for top,height in _runs(labels[:,x]):
                            if abs(height-side)<=max(3,side*.015) and top>=0:
                                proposals.add((top,height))
                for top,height in proposals:
                    if top+height-small.height()>height/14*.16:continue
                    rect=QRect(round(left/scale),round(top/scale),round(side/scale),round(height/scale))
                    # Refine coarse rounding using full-resolution grid edges.
                    if native_labels is None:
                        if native is None:native=rgb(image)
                        native_labels=_labels(native,np.array([a,b]))
                    hfits=[];vfits=[]
                    for c in range(3,11):
                        for fraction in (.07,.93):
                            yy=round(rect.y()+(c+fraction)*rect.height()/14)
                            xx=round(rect.x()+(c+fraction)*rect.width()/14)
                            hfits.extend((l,s) for l,s in _runs(native_labels[yy]) if abs(l-rect.x())<8/scale and abs(s-rect.width())<8/scale)
                            vfits.extend((t,s) for t,s in _runs(native_labels[:,xx]) if abs(t-rect.y())<8/scale and abs(s-rect.height())<8/scale)
                    if hfits and vfits:
                        fitted_left,width=np.median(hfits,axis=0).round().astype(int)
                        fitted_top,height=np.median(vfits,axis=0).round().astype(int)
                        rect=QRect(int(fitted_left),int(fitted_top),int(width),int(height))
                    if _visible(image,rect) is None:continue
                    score=grid_score(image,rect)
                    if score>=.80:candidates.append((score,rect))
    if not candidates:return None
    candidates.sort(key=lambda item:item[0],reverse=True);best=candidates[0][1]
    if any(abs(r.center().x()-best.center().x())>best.width()*.2 or
           abs(r.center().y()-best.center().y())>best.height()*.2 for _,r in candidates):return None
    return best
