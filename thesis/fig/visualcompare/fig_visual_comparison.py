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
    
def resultFor(num, geometryReader, alg=None, arg=None, statesOpt=None, fname=None):
    #img = vigra.impex.readImage("data/image-seg/images/test/%d.jpg" % num).view(numpy.ndarray)
    img = vigra.impex.readImage("%d.jpg" % num).view(numpy.ndarray)
    img = vigra.colors.transform_RGB2Lab(img)
    img = img[:,:,0] #make image look a bit lighter
    
    print img.shape
    if img.ndim == 3:
        img = vigra.sampling.resize(img, (img.shape[0]*2, img.shape[1]*2, 3))
    else:
        img = vigra.sampling.resize(img, (img.shape[0]*2, img.shape[1]*2))
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
    
    makeBoundaryImage(img, boundaryData, mask, fname)
    
def initImage(num):
    #img = vigra.impex.readImage("data/image-seg/images/test/%d.jpg" % num).view(numpy.ndarray)
    img = vigra.impex.readImage("%d.jpg" % num).view(numpy.ndarray)
    print img.shape
    img = vigra.sampling.resize(img, (img.shape[0]*2, img.shape[1]*2, 3))
    #f = h5py.File("data/image-seg/superpixel-segmentations/%d.bmp.h5" % num, 'r')
    f = h5py.File("sp_%d.bmp.h5" % num, 'r')
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

def makeSuperpixelImage(num):
    img = vigra.impex.readImage("%d.jpg" % num).view(numpy.ndarray)
    imgLab = vigra.colors.transform_RGB2Lab(img)
    img = imgLab[:,:,0]
    img = vigra.sampling.resize(img, (img.shape[0]*2, img.shape[1]*2))
    img = numpy.dstack([img, img, img])
    f = h5py.File("sp_%d.bmp.h5" % num, 'r')
    segA = f["superpixel-segmentation"].value
    segA = segA.swapaxes(0,1)
    f.close()
    ctable = numpy.random.random((segA.max()+1, 3)).astype(numpy.float32)
    segA_img = ctable[segA]
    
    ret = 1.0*imgLab #+ 0.6*segA_img 
    return ret

if __name__ == "__main__":
    num = 69015
    
    r = initImage(num)
    
    gmFname = "/media/tkroeger/pinky/image-seg/image-seg/0.3/%d.bmp.h5" % num
    
    gm = opengm.loadGm(gmFname, "gm")
    r = cgp.GeometryReader("/tmp/geom.h5")
    
    #MC-I
    param_mci = opengm.InfParam(workFlow="CCIFD")
    mci = opengm.inference.Multicut(gm, parameter=param_mci)
    mci.infer()
    arg_mci = mci.arg().astype(numpy.uint32)
    statesOpt = node_to_edge_labeling(arg_mci, r)
    resultFor(69015, r, arg=arg_mci, fname = "%d_MCI.png" % num, statesOpt=statesOpt)
    
    #a-exp-ecc
    f=h5py.File("69015_sol_aexpcc.h5"); arg_aexpcc = f["states"].value.astype(numpy.uint32); f.close()
    resultFor(69015, r, arg=arg_aexpcc,  fname = "%d_aexpcc.png"  % num, statesOpt=statesOpt)
    sys.exit()
   
    #superpixel
    arg_sp = numpy.arange(len(arg_mci), dtype=statesOpt.dtype) 
    resultFor(69015, r, arg=arg_sp, fname = "%d_superpixel.png"  % (num), statesOpt=statesOpt)
    sys.exit()
    
    #CGC, only cut phase
    param_cgc_steps = opengm.InfParam(useBookkeeping=True, planar=True, maxIterations=1, illustrationOut="cgc_steps.log")
    cgc_steps = opengm.inference.Cgc(gm, parameter=param_cgc_steps)
    cgc_steps.infer()
    f = open("cgc_steps.log", 'r')
    cut_phase = []
    for l in f.readlines():
        l=l.strip()
        l = l.split(" ")
        if l[0] == "R2C":
            l = l[1:]
            cut_phase.append(numpy.asarray(l, dtype=numpy.uint32))
    resultFor(69015, r, arg=cut_phase[0], fname = "%d_CGC_cut_phase_%02d.png"  % (num, 0), statesOpt=statesOpt)
    resultFor(69015, r, arg=cut_phase[-1], fname = "%d_CGC_cut_phase_%02d.png" % (num, len(cut_phase)-1), statesOpt=statesOpt)
    
    sys.exit()
    
  
    #MC-R
    param_mcr = opengm.InfParam(workFlow="CCFDB")
    mcr = opengm.inference.Multicut(gm, parameter=param_mcr)
    mcr.infer()
    arg_mcr = mcr.arg().astype(numpy.uint32)
   
    #CGC
    param_cgc = opengm.InfParam(useBookkeeping=True, planar=True)
    cgc = opengm.inference.Cgc(gm, parameter=param_cgc)
    cgc.infer()
    arg_cgc = cgc.arg().astype(numpy.uint32)
    
    #KL
    kl = opengm.inference.PartitionMove(gm)
    kl.infer()
    arg_kl = kl.arg().astype(numpy.uint32)
   
    #PlanarCC
    f=h5py.File("69015_sol_planarcc.h5"); arg_planarcc = f["states"].value.astype(numpy.uint32); f.close()
    
    
    resultFor(69015, r, arg=arg_planarcc,  fname = "%d_planarcc.png"  % num, statesOpt=statesOpt)
    resultFor(69015, r, arg=arg_mcr, fname = "%d_MCR.png" % num, statesOpt=statesOpt)
    resultFor(69015, r, arg=arg_cgc, fname = "%d_CGC.png" % num, statesOpt=statesOpt)
    resultFor(69015, r, arg=arg_kl,  fname = "%d_KL.png"  % num, statesOpt=statesOpt)
    
    sys.exit()
    
    #statesOpt = optimalStates(num, r)
    #resultFor(num, r, "MC_CCFDB", statesOpt=statesOpt)
    #resultFor(num, r, "CCP", statesOpt=statesOpt)
    #resultFor(num, r, "KL", statesOpt=statesOpt)
    #resultFor(num, r, "planarcc_40", statesOpt=statesOpt)
    #resultFor(num, r, "MC_CCIFD", statesOpt=statesOpt)
