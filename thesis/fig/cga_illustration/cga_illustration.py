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

def makeBoundaryImage(img, labels, segA, geometry, fname):
    f = h5py.File(segA, 'r')
    segA = f["seg"].value
    f.close()

    r = cgp.GeometryReader(geometry)

    boundaryData = numpy.ones((r.maxLabel(2),), dtype=numpy.float32)
    for faceLabel in range(1,r.maxLabel(2)+1):
        b = r.bounds(2, faceLabel)
        assert len(b) == 2
        if(labels[b[0]-1] == labels[b[1]-1]):
            boundaryData[faceLabel-1] = 0
    
    #
    # because the painting code crashes after one image,
    # here is the workaround which uses os.system to call
    # the painting code for each image anew
    #
    
    f = h5py.File("/tmp/bd.h5", 'w')
    f.create_dataset("boundaryData", data=boundaryData)
    f.close()
    imgfile = "/tmp/img.png"
    vigra.impex.writeImage(img, imgfile)
    os.system("python paint_boundaries.py /tmp/tgslices.h5 %s %s /tmp/bd.h5" % (fname, imgfile))

    '''
    labels = numpy.asarray(labels, dtype=numpy.uint32)
    result = labels[segA[:,:,0]-1]
    logger.log(Color.yellow, "making segmentation image '%s'" % fname)
    ctable = (255*numpy.random.random((labels.max()+1, 3))).astype(numpy.uint8)
    vigra.impex.writeImage(ctable[result], fname)
    '''

if __name__ == "__main__":
    img = vigra.impex.readImage("296059.jpg")

    gm = opengm.loadGm("./gm.h5", "gm")

    mc = opengm.inference.Multicut(gm)
    mc.infer()
    labels = mc.arg().astype(numpy.uint32)
    fname = "result_multicut.png"
    makeBoundaryImage(img, labels, "segA.h5", "geom.h5", fname)
    del mc
    del labels
    del fname

    pm = opengm.inference.PartitionMove(gm)
    pm.infer()
    labels = pm.arg().astype(numpy.uint32)
    fname = "result_partitionmove.png"
    makeBoundaryImage(img, labels, "segA.h5", "geom.h5", fname)
    del pm
    del labels
    del fname

    param = opengm.InfParam(threshold=0.0, nbfs=0, useBfs=False, initRegionSize=0, noInference=True)
    cgc0 = opengm.inference.Hmc(gm, parameter=param)
    cgc0.infer()
    labels = cgc0.arg().astype(numpy.uint32)
    fname = "result_cgc0.png"
    makeBoundaryImage(img, labels, "segA.h5", "geom.h5", fname)

'''
if True:
    img = vigra.impex.readImage("296059.jpg")

    f = h5py.File("cga_illustration.h5", 'w')
    f.create_dataset("img", data=img)

    imgLAB = vigra.colors.transform_RGB2Lab(img)

    g = vigra.filters.gaussianGradientMagnitude(imgLAB[:,:,0], 1.0)
    #g = vigra.filters.hessianOfGaussianEigenvalues(numpy.average(img, axis=2), 2.0)[:,:,0]
    vigra.impex.writeImage(g, "01_gradient.png")
    
    f.create_dataset("grad", data=g)

    m = vigra.analysis.extendedLocalMinima(vigra.filters.gaussianSmoothing(g, 3.0))
    m = vigra.analysis.labelImageWithBackground(m)
    print m.max()

    segA, numSeg = vigra.analysis.watersheds(g, seeds=m)

    ctable = (255*numpy.random.random((numSeg+1, 3))).astype(numpy.uint8)
    vigra.impex.writeImage(ctable[segA], "02_segA.png")

    f.create_dataset("segA", data=segA)
    segA = segA[:,:,numpy.newaxis]
    segA = numpy.dstack([segA, segA])

    segmentation.oversegmentation.cgpx("segA.h5", "tg.h5")
    segmentation.oversegmentation.cgpr("tg.h5", "geom.h5")
    segmentation.oversegmentation.tgslices("tg.h5", "tgslices.h5")

    r = cgp.GeometryReader("geom.h5")
    g = g[:,:,numpy.newaxis]; g = numpy.dstack([g, g])
    feat, names = segmentation.faceFeatures(r, g) 
    featMean = feat[:,0]
    w = (featMean-featMean.min())/(featMean.max()-featMean.min())
    w = 0.2-w

    print featMean.shape    

    outDir = "."
    numFac = r.maxLabel(2)
    numVar = r.maxLabel(3)
    gm = opengm.gm([numVar]*numVar)                                                                              
    gm.reserveFactors(numFac)                                                                                       
    gm.reserveFunctions(numFac,'potts')                                                                             
    fShape = [numVar, numVar] 
    for i in range(numFac):                                                                                         
        vis = r.bounds(2,i+1)
        assert vis[0] > 0 and vis[1] > 0 and vis[0] < vis[1]
        vis = [vis[0]-1, vis[1]-1]
        v00 = 0.0                                                                                                   
        v01 = float(w[i])                                                                                           
        fPotts  = opengm.pottsFunction(fShape,v00,v01)                                                              
        gm.addFactor(gm.addFunction(fPotts), numpy.asarray(vis, dtype=numpy.uint32))                                
    opengm.hdf5.saveGraphicalModel(gm,outDir+"/gm.h5", 'gm') 

    fname = "gm.h5"
    gm = opengm.adder.GraphicalModel()
    opengm.hdf5.loadGraphicalModel(gm, fname, "gm")
    print "  #var=%d, numFactors=%d" % (gm.numberOfVariables, gm.numberOfFactors)
    param = opengm.InfParam(dirtyHeuristic=True, debug=True, planar=True, initRegionSize=0, illustrationOut="cga_illustration_log.txt")
    cga = opengm.inference.Hmc(gm, parameter=param)                                                                                                                                                                                                                                                                                                 
    visitor = None
    if True:
        visitor = cga.verboseVisitor()                                                                                       
    cga.infer(visitor) 
    print gm.evaluate(cga.arg())

    l = cga.arg().astype(numpy.uint32)
    print l.shape, l.dtype
    segA = segA.view(numpy.ndarray)

    result = l[segA[:,:,0]-1]
    vigra.impex.writeImage(ctable[result], "03_result.png")
    
    f = open("cga_illustration_log.txt")
    for it, l in enumerate(f.readlines()):
        l = l.strip()
        l = l.split(" ")
        what = l[0]
        l = l[1:]
        l = [int(x) for x in l]
        assert len(l) == r.maxLabel(3) 
       
        boundaryData = numpy.ones((r.maxLabel(2),), dtype=numpy.float32)
        for faceLabel in range(1,r.maxLabel(2)+1):
            b = r.bounds(2, faceLabel)
            assert len(b) == 2
            if(l[b[0]-1] == l[b[1]-1]):
                boundaryData[faceLabel-1] = 0
        
        #
        # because the painting code crashes after one image,
        # here is the workaround which uses os.system to call
        # the painting code for each image anew
        #
        
        f = h5py.File("/tmp/bd.h5", 'w')
        f.create_dataset("boundaryData", data=boundaryData)
        f.close()
        fname = "04_%04d_boundaries_%s.png" % (it, what)
        imgfile = "/tmp/img.png"
        vigra.impex.writeImage(imgLAB, imgfile)
        os.system("python paint_boundaries.py tgslices.h5 %s %s /tmp/bd.h5" % (fname, imgfile))

        l = numpy.asarray(l, dtype=numpy.uint32)
        result = l[segA[:,:,0]-1]
        fname = "04_%04d_seg_%s.png" % (it, what)
        logger.log(Color.yellow, "making segmentation image '%s'" % fname)
        vigra.impex.writeImage(ctable[result], fname)
'''
