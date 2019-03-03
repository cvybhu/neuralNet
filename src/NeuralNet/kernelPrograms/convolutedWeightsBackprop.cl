
kernel void convolutedWeightsBackprop(
    global const float* prevVals,  //[prevFeat][prevX][prevY]
    global const float* outErrDerivatives,
    global float* weightsChanges, //[prevFeat][outFeat][kernelX][kernelY][outX][outY]
    global float* biasesChanges,  //[outFeat][outX][outY]
    int prevFeats,
    int outFeats,
    int width,
    int height,
    int kernelWidth,
    int kernelHeight
)
{
    //ok so im a pixel in out and want to share my opinion on each of the wieghts
    //which pixel am i?
    int gid = get_global_id(0);
    int outFeat, outX, outY;
    to3D(gid, &outFeat, &outX, &outY, outFeats, width);

    float outErrDerivative = outErrDerivatives[gid];

    float curWeightDerivative = 0;

    //Now gotta iterate over weights coming to me
    for(int prevFeat = 0; prevFeat < prevFeats; prevFeat++)
    {
        for(int kernelX = 0; kernelX < kernelWidth; kernelX++)
        {
            int prevX = outX - kernelWidth/2 + kernelX;
            if(prevX < 0 || prevX > width)
                continue;

            for(int kernelY = 0; kernelY < kernelHeight; kernelY++)
            {
                int prevY = outY - kernelHeight/2 + kernelY;
                if(prevY < 0 || prevY > height)
                    continue;

                float prevVal = prevVals[get3D(prevFeat, prevX, prevY,   prevFeats, width)];
                weightsChanges[get6D(prevFeat, outFeat, kernelX, kernelY, outX, outY,   prevFeats, outFeats, kernelWidth, kernelHeight, width)]
                 = prevVal * outErrDerivative;
            }
        }
    }

    biasesChanges[get3D(outFeat, outX, outY,  outFeats, width)] = outErrDerivative;
}