
kernel void convolutedUpdateWeights(
    global float* weights,
    global float* weightsChanges,
    global float* biases,
    global float* biasesChanges,
    int prevFeats,
    int outFeats,
    int width,
    int height,
    int kernelWidth,
    int kernelHeight,
    float learningRate
)
{
    int gid = get_global_id(0);

    int prevFeat, outFeat;
    to2D(gid, &prevFeat, &outFeat,  prevFeats);

    float changesNumber = width * height; //To average change

    for(int kernelX = 0; kernelX < kernelWidth; kernelX++)
    for(int kernelY = 0; kernelY < kernelHeight; kernelY++)
    {
        //float curChange = weightsUpdates[prevFeat][outFeat][kernelX][kernelY][0][0];
        float curChange = weightsChanges[get6D(prevFeat, outFeat, kernelX, kernelY, 0, 0,   prevFeats, outFeats, kernelWidth, kernelHeight, width)];
        weights[get4D(prevFeat, outFeat, kernelX, kernelY,   prevFeats, outFeats, kernelWidth)]
         += learningRate * curChange / changesNumber;
    }

    if(prevFeat == 0) 
    { //update biases[outFeat]
        float biasChange = biasesChanges[get3D(outFeat, 0, 0,  outFeats, width)];
        biases[outFeat] += learningRate * biasChange;
    }
}