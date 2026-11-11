"""Optional labeled RGB exemplars for a visually verified colored piece theme.

No chess-position priors or square labels at runtime. Similarities are NOT
neural probabilities. All 64 cells must pass an absolute and runner-up margin
gate before this fallback may replace an uncertain CNN result.
"""
from functools import lru_cache
import numpy as np
from PySide6.QtCore import Qt
from .capture.board_region import rgb


def feature(cell):
    from PySide6.QtGui import QImage
    cell=cell.scaled(48,48,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
    a=rgb(cell).astype(np.float32)
    # Separate local background using four edge patches; neutralize only the
    # background, keeping actual foreground color and its antialiased outline.
    corners=np.concatenate([a[2:7,2:7].reshape(-1,3),a[2:7,41:46].reshape(-1,3),
                            a[41:46,2:7].reshape(-1,3),a[41:46,41:46].reshape(-1,3)])
    bg=np.median(corners,axis=0)
    alpha=np.clip((np.linalg.norm(a-bg,axis=2)-35)/30,0,1)
    alpha[:2]=0;alpha[-2:]=0;alpha[:,:2]=0;alpha[:,-2:]=0
    return (a*alpha[:,:,None]+128*(1-alpha[:,:,None]))/255,alpha


@lru_cache(maxsize=1)
def load_bank():
    from .paths import resource_root
    path=resource_root()/'models'/'colored_exemplars.npz'
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as data:
        return data['features'],data['labels']


def classify(image):
    bank=load_bank()
    if bank is None:return None
    features,labels=bank;result=[];occupied=0
    for row in range(8):
        for col in range(8):
            x0,x1=round(col*image.width()/8),round((col+1)*image.width()/8)
            y0,y1=round(row*image.height()/8),round((row+1)*image.height()/8)
            value,mask=feature(image.copy(x0,y0,x1-x0,y1-y0))
            if mask[7:41,7:41].mean()<.025:
                result.append((None,1.));continue
            distances=np.mean(np.abs(features-value),axis=(1,2,3))
            ranked=sorted((float(distances[labels==label].min()),str(label)) for label in np.unique(labels))
            best,label=ranked[0];margin=ranked[1][0]-best
            if best>.055 or margin<.012:return None
            occupied+=1;result.append((label,1.-best))
    return result if occupied>=8 else None
