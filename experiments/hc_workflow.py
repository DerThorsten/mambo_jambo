import vigra
import vigra.graphs as vigraph
import numpy
import os
import pylab
print "get input"

baseName = "12074"
f = "/home/tbeier/datasets/BSR/BSDS500/data/images/train/%s.jpg" % baseName

outBaseDir = "/home/tbeier/src/masterthesis/thesis/fig/"
outDir = outBaseDir + baseName+"/"


if not os.path.exists(outDir):
    os.makedirs(outDir)


sigma = 4.0

img = vigra.impex.readImage(f)  # [0:100,0:100,:]
imgLab = vigra.colors.transform_RGB2Lab(img)
imgLabInterpolated = vigra.resize(
    imgLab, [imgLab.shape[0] * 2 - 1, imgLab.shape[1] * 2 - 1])
gradmag = numpy.squeeze(
    vigra.filters.gaussianGradientMagnitude(imgLabInterpolated, sigma))
labels, nseg = vigra.analysis.slicSuperpixels(imgLab, 10.0, 3)
labels = numpy.squeeze(vigra.analysis.labelImage(labels))


gridGraph = vigraph.gridGraph(img.shape[0:2])
gridGraphEdgeIndicator = vigraph.edgeFeaturesFromInterpolatedImage(
    gridGraph, gradmag)

rag = vigraph.regionAdjacencyGraph(gridGraph, labels)
edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)
nodeFeatures = rag.accumulateNodeFeatures(img)


counter=1
while rag.nodeNum >= 2:
    
    nodeNumStop = rag.nodeNum / 2
    print rag.nodeNum,nodeNumStop

    print "get the seg"
    labels = vigraph.agglomerativeClustering(
        rag, edgeWeights=edgeWeights, beta=0.1, wardness=1.0,
        nodeFeatures=nodeFeatures, nodeNumStop=nodeNumStop)

    #labels = vigraph.felzenszwalbSegmentation(rag, edgeWeights,nodeNumStop=nodeNumStop)

    print "make the rag"

    rag2 = vigraph.regionAdjacencyGraph(graph=rag, labels=labels)

    if rag2.nodeNum == 1:
        break


    edgeWeights = rag2.accumulateEdgeFeatures(edgeWeights)
    print "blabla"
    nodeFeatures = rag2.accumulateNodeFeatures(nodeFeatures)

    print "as image"

    asImg = rag2.projectNodeFeaturesToGridGraph(nodeFeatures)
    asImg = vigra.taggedView(asImg, "xyc")


    outFileName = outDir + str(counter)+".png"
    print outFileName
    
    rag = rag2
    counter+=1
    asSegImg=rag2.show(asImg,returnImg=True)
    vigra.impex.writeImage(asSegImg, outFileName)
    #vigra.show()

print "exit"
