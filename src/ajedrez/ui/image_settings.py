"""Local decoration import: alpha preservation and conservative border removal."""
from collections import deque
import numpy as np
from PySide6.QtCore import Qt,QThread,Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QCheckBox
from ..paths import user_data_dir

def prepare_image(image,remove_background=True):
    if image.isNull():raise ValueError('No se pudo abrir la imagen.')
    image=image.scaled(1000,1000,Qt.KeepAspectRatio,Qt.SmoothTransformation) if max(image.width(),image.height())>1000 else image
    rgba=image.convertToFormat(QImage.Format_RGBA8888)
    a=np.frombuffer(rgba.constBits(),np.uint8).reshape(rgba.height(),rgba.bytesPerLine())[:,:rgba.width()*4].reshape(rgba.height(),rgba.width(),4).copy()
    if remove_background and np.all(a[:,:,3]==255):
        border=np.concatenate((a[0,:,:3],a[-1,:,:3],a[:,0,:3],a[:,-1,:3])).astype(float)
        background=np.median(border,axis=0)
        if np.mean(np.linalg.norm(border-background,axis=1)<35)<.85:
            raise ValueError('El fondo no es uniforme. Usa un PNG ya recortado o desmarca quitar fondo.')
        mask=np.linalg.norm(a[:,:,:3].astype(float)-background,axis=2)<35
        seen=np.zeros(mask.shape,bool);h,w=mask.shape
        queue=deque((y,x) for y in range(h) for x in (0,w-1) if mask[y,x])
        queue.extend((y,x) for y in (0,h-1) for x in range(w) if mask[y,x])
        while queue:
            y,x=queue.popleft()
            if seen[y,x]:continue
            seen[y,x]=True
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<h and 0<=xx<w and mask[yy,xx] and not seen[yy,xx]:queue.append((yy,xx))
        a[seen,3]=0
    return QImage(a.data,rgba.width(),rgba.height(),a.strides[0],QImage.Format_RGBA8888).copy()

class ImagePreparationWorker(QThread):
    ready=Signal(object)
    failed=Signal(str)
    def __init__(self,path,remove,parent):
        super().__init__(parent);self.path=path;self.remove=remove
    def run(self):
        try:self.ready.emit(prepare_image(QImage(self.path),self.remove))
        except Exception as exc:self.failed.emit(str(exc))

class ImageSettingsDialog(QDialog):
    def __init__(self,panel,parent=None):
        super().__init__(parent);self.worker=None;self.setWindowTitle('Imágenes del programa');layout=QVBoxLayout(self)
        layout.addWidget(QLabel('Elige una imagen para cada espacio. Se guarda una copia PNG local.'))
        self.remove=QCheckBox('Quitar fondo uniforme (los PNG transparentes se conservan)');self.remove.setChecked(True)
        layout.addWidget(self.remove)
        self.status=QLabel();self.status.setWordWrap(True)
        for title,widget in (('Izquierda',panel.ino),('Derecha superior',panel.right_decorations.winry),('Derecha inferior',panel.right_decorations.orihime)):
            row=QHBoxLayout();row.addWidget(QLabel(title));choose=QPushButton('Elegir…');reset=QPushButton('Predeterminada')
            choose.clicked.connect(lambda _,w=widget:self.choose(w));reset.clicked.connect(lambda _,w=widget:self.restore(w))
            row.addWidget(choose);row.addWidget(reset);layout.addLayout(row)
        layout.addWidget(self.status);close=QPushButton('Cerrar');close.clicked.connect(self.accept);layout.addWidget(close)
    def choose(self,widget):
        if self.worker is not None:return
        path,_=QFileDialog.getOpenFileName(self,'Elegir imagen','','Imágenes (*.png *.jpg *.jpeg *.webp *.bmp)')
        if not path:return
        self.status.setText('Preparando imagen local…')
        worker=ImagePreparationWorker(path,self.remove.isChecked(),self);self.worker=worker
        worker.ready.connect(lambda image:self.save_image(widget,image))
        worker.failed.connect(self.status.setText)
        worker.finished.connect(self.preparation_finished);worker.start()
    def preparation_finished(self):
        worker=self.worker;self.worker=None
        if worker:worker.deleteLater()
    def reject(self):
        if self.worker is None:super().reject()
    def accept(self):
        if self.worker is None:super().accept()
    def save_image(self,widget,image):
        try:
            folder=user_data_dir()/'decorations';folder.mkdir(exist_ok=True)
            target=folder/widget.filename
            if not image.save(str(target),'PNG'):raise ValueError('No se pudo guardar la imagen.')
            widget.reload_image();self.status.setText('Imagen actualizada. El archivo original no se modifica.')
        except Exception as exc:self.status.setText(str(exc))
    def restore(self,widget):
        if self.worker is not None:return
        path=user_data_dir()/'decorations'/widget.filename
        if path.exists():path.unlink()
        widget.reload_image();self.status.setText('Imagen predeterminada restaurada.')
