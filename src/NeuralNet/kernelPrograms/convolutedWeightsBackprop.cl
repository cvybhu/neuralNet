
kernel void convolutedWeightsBackprop(
    global float* outVals,  //[outFeat][outX][outY]
    global float* weightsChanges, //[prevFeat][outFeat][kernelX][kernelY][outX][outY]
    int prevFeats,
    int outFeats,
    int width,
    int height,
    int kernelWidth,
    int kernelHeight
)
{
    //ok so im a pixel in out and want to share my opinion on each of the wieghts

}