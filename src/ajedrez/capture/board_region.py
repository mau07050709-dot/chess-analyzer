"""Lightweight checkerboard geometry checks; no piece recognition or chess state."""
from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage


def rgb(image):
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    return np.frombuffer(converted.constBits(), np.uint8).reshape(
        converted.height(), converted.bytesPerLine())[:, :converted.width()*3].reshape(
        converted.height(), converted.width(), 3).copy()


def background_colors(image):
    pixels = rgb(image)
    colors = [[], []]
    for row in range(8):
        for col in range(8):
            for dy, dx in ((.06,.08),(.06,.92),(.94,.08),(.94,.92)):
                colors[(row+col)%2].append(pixels[min(image.height()-1,int((row+dy)*image.height()/8)),
                                                       min(image.width()-1,int((col+dx)*image.width()/8))])
    pair = np.asarray([np.median(samples, axis=0) for samples in colors], dtype=np.float32)
    return pair if np.linalg.norm(pair[0]-pair[1]) >= 35 else None


def grid_score(image, colors):
    if image.isNull() or min(image.width(),image.height()) < 96 or colors is None:
        return 0.
    if abs(image.width()/image.height()-1) > .035:
        return 0.
    pixels = rgb(image.scaled(128,128,Qt.AspectRatioMode.IgnoreAspectRatio,
                              Qt.TransformationMode.FastTransformation)).astype(np.float32)
    scores = []
    for dy, dx in ((1,1),(1,14),(14,1),(14,14)):
        samples = pixels[dy::16,dx::16]
        parity = np.indices((8,8)).sum(axis=0)%2
        scores.append(np.linalg.norm(samples-colors[parity],axis=2) < 42)
    # Coordinate glyphs/highlights may cover a minority of corners.
    return float(np.mean(scores))


def _eight_runs(labels, cell_hint=None):
    # Only bridge isolated antialias pixels, never large piece silhouettes.
    labels = labels.copy()
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    for i in range(1,len(boundaries)-2):
        start,end = boundaries[i:i+2]
        if labels[start] < 0 and end-start <= (max(2,round(cell_hint*.08)) if cell_hint else 2) and labels[start-1] >= 0 and labels[end] >= 0:
            labels[start:end] = labels[start-1]
    changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    values = labels[changes[:-1]]
    for start in range(len(values)-7):
        seq = values[start:start+8]
        if np.any(seq < 0) or np.any(seq[1:] == seq[:-1]):
            continue
        edges = changes[start:start+9]
        lengths = np.diff(edges)
        cell = float(np.median(lengths[1:-1]))
        # A 128px video board becomes ~60px in the 900px screen scan.
        # About seven samples/cell suffice for a coarse proposal; the native
        # grid fit and >=.80 corner validation below remain mandatory.
        if cell < 7 or np.max(np.abs(lengths[1:-1]-cell)) > max(2.,cell*.09):
            continue
        # Outer runs can merge with a surrounding page of the same color.
        if lengths[0] < cell*.85 or lengths[-1] < cell*.85:
            continue
        fitted_cell = float(np.dot(np.arange(-3,4), edges[1:8])/28)
        yield round(float(np.mean(edges[1:8]))-4*fitted_cell), round(fitted_cell*8)


def _refine_rect(pixels, rect, colors):
    """Fit the full-resolution grid after coarse scan rounding (no inference)."""
    rows,cols=np.indices((8,8));parity=(rows+cols)%2
    best=None;best_score=0.;best_quality=-1.
    rr,cc=np.indices((8,7));cc=cc+1
    for dx in range(-3,4):
        for dy in range(-3,4):
            for ds in range(-3,4):
                candidate=QRect(rect.x()+dx,rect.y()+dy,rect.width()+ds,rect.height()+ds)
                if (candidate.left()<0 or candidate.top()<0 or candidate.right()>=pixels.shape[1]
                        or candidate.bottom()>=pixels.shape[0]):continue
                scores=[]
                for cy,cx in ((.09375,.09375),(.09375,.90625),(.90625,.09375),(.90625,.90625)):
                    ys=(candidate.y()+(rows+cy)*candidate.height()/8).astype(int)
                    xs=(candidate.x()+(cols+cx)*candidate.width()/8).astype(int)
                    scores.append(np.linalg.norm(pixels[ys,xs]-colors[parity],axis=2)<42)
                score=float(np.mean(scores))
                # Match both sides of all seven internal boundaries. Corner
                # scores alone have a flat plateau that can shift a crop.
                boundary=[]
                for offset in (-1.,1.):
                    ys=(candidate.y()+(rr+.5)*candidate.height()/8).astype(int)
                    xs=(candidate.x()+cc*candidate.width()/8+offset).astype(int)
                    expected=colors[(rr+cc-(offset<0))%2]
                    boundary.append(np.linalg.norm(pixels[ys,xs]-expected,axis=2)<42)
                    xs=(candidate.x()+(rr+.5)*candidate.width()/8).astype(int)
                    ys=(candidate.y()+cc*candidate.height()/8+offset).astype(int)
                    boundary.append(np.linalg.norm(pixels[ys,xs]-expected,axis=2)<42)
                quality=score+.25*float(np.mean(boundary))
                if quality>best_quality:
                    best,best_score,best_quality=candidate,score,quality
    return best,best_score


def locate_board(image, colors, excluded=()):
    """Return a local physical-pixel square, or None. Search only on relocation."""
    if colors is None or image.isNull():
        return None
    scale = min(1., 900/max(image.width(),image.height()))
    small = image.scaled(round(image.width()*scale),round(image.height()*scale),
                         Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
    pixels = rgb(small).astype(np.float32)
    distance = np.linalg.norm(pixels[:,:,None,:]-colors[None,None,:,:],axis=3)
    labels = np.argmin(distance,axis=2).astype(np.int8)
    labels[np.min(distance,axis=2) > 42] = -1
    for rect in excluded:
        r = QRect(round(rect.x()*scale),round(rect.y()*scale),round(rect.width()*scale),round(rect.height()*scale))
        labels[max(0,r.top()):r.bottom()+1,max(0,r.left()):r.right()+1] = -1
    original_pixels = rgb(image).astype(np.float32)
    seen = set(); processed = []
    best = None; best_score = .79; candidates = []
    for y in range(0,small.height(),3):
        for left,side in _eight_runs(labels[y]):
            if left < 0 or left+side > small.width() or side > small.height():
                continue
            key = (left//3,side//3)
            if key in seen:
                continue
            seen.add(key)
            # A vertical line can cross glyphs/coordinate digits in one file.
            # Combine both background edges of all files, normalized by parity,
            # and require a consistent majority before proposing row boundaries.
            columns=[]
            for file in range(8):
                for offset in (.03,.97):
                    x=min(small.width()-1,round(left+side*(file+offset)/8))
                    values=labels[:,x]
                    columns.append(np.where(values>=0,values^(file%2),-1))
            samples=np.asarray(columns)
            zeros=np.sum(samples==0,axis=0);ones=np.sum(samples==1,axis=0)
            trace=np.full(small.height(),-1,dtype=np.int8)
            trace[(zeros>=10)&(zeros>ones)]=0;trace[(ones>=10)&(ones>zeros)]=1
            traces=[trace]
            for fraction in (.03/8,.97/8,1.06/8,6.94/8):
                traces.append(labels[:,min(small.width()-1,round(left+side*fraction))])
            for trace in traces:
                for top,height in _eight_runs(trace,side/8):
                    if abs(height-side)>max(3,side*.025) or top<0 or top+side>small.height():
                        continue
                    rect = QRect(round(left/scale),round(top/scale),round(side/scale),round(side/scale))
                    if any(rect.intersects(r) for r in excluded):
                        continue
                    if any(abs(rect.center().x()-old.center().x())<rect.width()*.05
                           and abs(rect.center().y()-old.center().y())<rect.height()*.05
                           and abs(rect.width()-old.width())<rect.width()*.05 for old in processed):continue
                    processed.append(rect)
                    rect,score = _refine_rect(original_pixels,rect,colors)
                    if rect is None or any(rect.intersects(r) for r in excluded):continue
                    score = min(score,grid_score(image.copy(rect),colors))
                    if score >= .80:
                        candidates.append(rect)
                    if score > best_score:
                        best,best_score=rect,score
    if best is not None and any(
            abs(rect.center().x()-best.center().x())>best.width()*.25
            or abs(rect.center().y()-best.center().y())>best.height()*.25
            or not .75 < rect.width()/best.width() < 1.33 for rect in candidates):
        return None
    return best


@dataclass(frozen=True)
class LocatedRegion:
    screen_index: int
    region: QRect
    score: float

