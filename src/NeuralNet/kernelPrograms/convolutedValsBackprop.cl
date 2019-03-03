
kernel void convolutedValsBackprop(
    global float* prevErrDerivatives,
    int prevFeats,
    int prevWidth,
    int prevHeight,
    global const float* weights,
    int kernelWidth,
    int kernelHeight,
    global const float* outErrDerivatives,
    int outFeats,
    int outWidth,
    int outHeight
)
{
    int gid = get_global_id(0);
    int prevFeat, x, y;
    to3D(gid, &prevFeat, &x, &y, prevFeats, prevWidth);

    float errDerivative = 0;

    //This value affects each of [outFeat][x +- kernelWidth/2][y +- kernelheight/2]
    for(int outFeat = 0; outFeat < outFeats; outFeat++)
    {
        for(int kernelX = 0; kernelX < kernelWidth; kernelX++)
        {
            int outX = x  + kernelWidth/2 - kernelX;
            if(outX < 0 || outX >= outWidth)
                continue;

            for(int kernelY = 0; kernelY < kernelHeight; kernelY++)
            {
                int outY = y + kernelHeight/2 - kernelY;
                if(outY < 0 || outY > outHeight)
                    continue;

                float curWeight = weights[get4D(prevFeat, outFeat, kernelX,   kernelY, prevFeats, outFeats, kernelWidth)];
                float curOutDerivative = outErrDerivatives[get3D(outFeat, outX, outY,  outFeats, outWidth)];
                errDerivative += curWeight * curOutDerivative;
            }
        }
    }

    prevErrDerivatives[gid] = errDerivative;
}