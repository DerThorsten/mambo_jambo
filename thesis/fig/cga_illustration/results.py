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
from segmentation.labeling.boundariesLayer import BoundariesLayer
from chatty import Logger, Color
logger = Logger.getInstance()

def node_to_edge_labeling(labels, geometryReader):
    boundaryData = numpy.ones((geometryReader.maxLabel(2),), dtype=numpy.float32)
    for faceLabel in range(1,geometryReader.maxLabel(2)+1):
        b = geometryReader.bounds(2, faceLabel)
        assert len(b) == 2
        if(labels[b[0]-1] == labels[b[1]-1]):
            boundaryData[faceLabel-1] = 0
    return boundaryData

def makeBoundaryImage(img, boundaryData, mask, fname):
    #
    # because the painting code crashes after one image,
    # here is the workaround which uses os.system to call
    # the painting code for each image anew
    #
    f = h5py.File("/tmp/bd.h5", 'w')
    f.create_dataset("boundaryData", data=boundaryData)
    f.create_dataset("mask", data=mask)
    f.close()
    imgfile = "/tmp/img.png"
    vigra.impex.writeImage(img, imgfile)
    
    cmd = "python paint_boundaries.py /tmp/tgslices.h5 %s %s /tmp/bd.h5" % (fname, imgfile)
    
    logger.log(Color.yellow, cmd)
    
    os.system(cmd)

    '''
    labels = numpy.asarray(labels, dtype=numpy.uint32)
    result = labels[segA[:,:,0]-1]
    logger.log(Color.yellow, "making segmentation image '%s'" % fname)
    ctable = (255*numpy.random.random((labels.max()+1, 3))).astype(numpy.uint8)
    vigra.impex.writeImage(ctable[result], fname)
    '''

def optimalStates(num, geometryReader):
    alg = "MC_CCIFD"
    fname = "data/image-seg/image-seg-b-03/%s/%d.bmp.h5" % (alg, num)
    f = h5py.File(fname, 'r')
    arg = f["states"].value
    values = f["values"].value
    f.close()
    return node_to_edge_labeling(arg, geometryReader)
    
def resultFor(num, geometryReader, alg=None, arg=None, statesOpt=None):
    img = vigra.impex.readImage("data/image-seg/images/test/%d.jpg" % num).view(numpy.ndarray)
    print img.shape
    img = vigra.sampling.resize(img, (img.shape[0]*2, img.shape[1]*2, 3))
    if alg is not None:
        fname = "data/image-seg/image-seg-b-03/%s/%d.bmp.h5" % (alg, num)
        if not os.path.exists(fname):
            raise RuntimeError("does not exist '%s'" % fname)
        f = h5py.File(fname, 'r')
        arg = f["states"].value
        values = f["values"].value
        print "%s has value %f" % (alg, values[-1])
        f.close()
        fname = "%d_%s.png" % (num, alg)
    elif arg is None:
        raise RuntimeError("need arg")
        if fname is None:
            raise RuntimeError("need fname")
    
    boundaryData = node_to_edge_labeling(arg, geometryReader)
    mask = numpy.ones(len(boundaryData), dtype=numpy.float32)
    if statesOpt is not None:
        assert len(boundaryData) == len(statesOpt)
        for i, (opt, this) in enumerate(zip(statesOpt,boundaryData)):
            assert this in [0,1]
            assert opt in [0,1]
            if(opt == 1 and this==1):
                boundaryData[i] = 0.5
            elif(opt == 0 and this==0):
                mask[i] = 0 
            elif(opt == 1 and this == 0):
                boundaryData[i] = 0.0
            elif(opt == 0 and this == 1):
                boundaryData[i] = 1.0
            else:
                raise RuntimeError("programmer stupid")
            
    print "result for %s has 0.0=%d, 0.5=%d, 1.0=%d" % (alg, numpy.sum(boundaryData==0.0), numpy.sum(boundaryData==0.5), numpy.sum(boundaryData==1.0))
    
    makeBoundaryImage(img, boundaryData, mask, fname, penWidth=4.0)
    
def initImage(num):
    img = vigra.impex.readImage("data/image-seg/images/test/%d.jpg" % num).view(numpy.ndarray)
    print img.shape
    img = vigra.sampling.resize(img, (img.shape[0]*2, img.shape[1]*2, 3))
    f = h5py.File("data/image-seg/superpixel-segmentations/%d.bmp.h5" % num, 'r')
    segA = f["superpixel-segmentation"].value
    segA = segA.swapaxes(0,1)
    f.close()
    
    segA = segA[:,:,numpy.newaxis]
    segA = numpy.dstack([segA, segA])
    f = h5py.File("/tmp/seg.h5", 'w')
    f.create_dataset("seg", data=(segA).astype(numpy.uint32))
    f.close()

    segmentation.oversegmentation.cgpx("/tmp/seg.h5", "/tmp/tg.h5")
    segmentation.oversegmentation.cgpr("/tmp/tg.h5", "/tmp/geom.h5")
    segmentation.oversegmentation.tgslices("/tmp/tg.h5", "/tmp/tgslices.h5")
    
    r = cgp.GeometryReader("/tmp/geom.h5")
    return r

if __name__ == "__main__":
    num = 69015
    r = initImage(num)
    
    statesOpt = optimalStates(num, r)
    
    gm = opengm.loadGm(" ./gm.h5", "gm")
    r = cgp.GeometryReader("/tmp/geom.h5")
    param = opengm.InfParam(bookkeeping=True, planar=True)
    cgc = opengm.inference.Cgc(gm, parameter=param)
    cgc.infer()
    labels = cgc.arg().astype(numpy.uint32)
    resultFor(69015, r, arg=labels, fname = "%d_CGC.png" % num)
    
    resultFor(num, r, "MC_CCFDB", statesOpt=statesOpt)
    resultFor(num, r, "CCP", statesOpt=statesOpt)
    resultFor(num, r, "KL", statesOpt=statesOpt)
    resultFor(num, r, "planarcc_40", statesOpt=statesOpt)
    resultFor(num, r, "MC_CCIFD", statesOpt=statesOpt)
