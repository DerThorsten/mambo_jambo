import vigra
import vigra.graphs as vigraph
import numpy

print "get input"
f = "/home/tbeier/datasets/BSR/BSDS500/data/images/train/12074.jpg"


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

while rag.nodeNum >= 2:
    nodeNumStop = rag.nodeNum / 2
    labels = vigraph.agglomerativeClustering(
        rag, edgeWeights=edgeWeights, beta=0.1, wardness=1.0,
        nodeFeatures=nodeFeatures, nodeNumStop=nodeNumStop)
    rag2 = vigraph.regionAdjacencyGraph(graph=rag, labels=labels)

    edgeWeights = rag2.accumulateEdgeFeatures(edgeWeights)
    nodeFeatures = rag2.accumulateNodeFeatures(nodeFeatures)

    asImg = rag2.projectNodeFeaturesToGridGraph(nodeFeatures)
    asImg = vigra.taggedView(asImg, "xyc")

    rag2.show(asImg)
    vigra.show()
    rag = rag2
