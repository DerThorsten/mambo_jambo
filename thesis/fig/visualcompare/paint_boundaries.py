import glob

import sys
import os

import numpy
import vigra
import qimage2ndarray
import h5py

from PyQt4.QtGui import QTransform, QImage, QColor, QGraphicsScene, QGraphicsPixmapItem, QApplication, \
                        QPixmap, QPainter
from PyQt4.QtCore import QRectF

import opengm

#Thorben's projects
import cgp
import segmentation
import segmentation.oversegmentation
from segmentation.labeling.boundariesLayer import BoundariesLayerPy, BoundariesLayer
from chatty import Logger, Color
logger = Logger.getInstance()

def paintBoundaries(tgslicesFile, zSlice, d, boundaryData, mask, filename, swapCoordinates=False, colortable=None, penWidth=4.0):
    print "* making boundary img '%s'" % filename
    
    #b = BoundariesLayerPy(tgslicesFile=tgslicesFile, normalAxis=2, data2scene=QTransform(), swapCoordinates=True)
    b = BoundariesLayer(None, 2, QTransform(), True)
    
    n  = b.normalAxis()
    axis = None
    if swapCoordinates:
        if   n == 0: axis = "z"
        elif n == 1: axis = "y"
        else:        axis = "x"
    else:
        if   n == 0: axis = "x"
        elif n == 1: axis = "y"
        else:        axis = "z"
    
    f = h5py.File(tgslicesFile, 'r')
    group = "%s/%d" % (axis, zSlice)
    serializedBoundaries = f[group].value
    f.close()
    
    assert d.ndim == 2
    
    scene = QGraphicsScene()
    
    b.setSliceNumber( zSlice )
    b.setBoundaries(serializedBoundaries)
    b.setColormap("tyr")
    assert boundaryData.dtype == numpy.float32
    assert mask.dtype == numpy.float32
    b.setBoundaryData(boundaryData, boundaryData.size, boundaryData)
    b.setBoundaryMask(mask, mask.size)
    if colortable is not None:
        b.setColormap(colortable)
    print "setting pen width to be %f" % penWidth
    b.setPenWidth(float(penWidth))
    print "...done"
   
    mag = 4
    shape = d.shape
    
    dBig = vigra.sampling.resizeImageNoInterpolation(d.astype(numpy.float32), (mag*shape[0], mag*shape[1]))
    #dBig = dBig.swapaxes(0,1)
    qimg = qimage2ndarray.gray2qimage(dBig)
    #qimg = qimg.mirrored(True, False)
    imgItm = QGraphicsPixmapItem(QPixmap(qimg))
    imgItm.setScale(1.0/mag)
    scene.addItem(imgItm)
    
    sourceRect = QRectF(0,0,shape[1], shape[0])
    targetRect = QRectF(0,0,mag*shape[1], mag*shape[0])
    scene.setSceneRect(sourceRect)
    scene.addItem(b)
    
    img = QImage(targetRect.width(), targetRect.height(), QImage.Format_ARGB32);
    painter = QPainter(img);
    painter.setRenderHint(QPainter.Antialiasing);
    scene.render(painter, targetRect, sourceRect );
    img.save(filename)
    painter.end()
    #print "img has size ", img.width(), img.height()
    img = None
    painter = None
    scene = None

if __name__ == "__main__":
    assert len(sys.argv) == 5

    tgslicesFile = sys.argv[1]
    assert os.path.exists(tgslicesFile)
    
    fname = sys.argv[2]
    
    assert os.path.exists(sys.argv[3])
    img   = vigra.impex.readImage(sys.argv[3])[:,:,0]
   
    assert os.path.exists(sys.argv[4])
    f = h5py.File(sys.argv[4], 'r')
    boundaryData = f["boundaryData"].value
    mask = f["mask"].value
    
    app = QApplication([])
    paintBoundaries(tgslicesFile, 0, img, boundaryData, mask, fname, colortable=None)
    os.system("convert %s -rotate 90 -flop %s" % (fname, fname))
    #os.system("optipng %s" % fname)

