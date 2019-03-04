
kernel void maxPoolBackprop(
    global const float* outErrDerivatives,
    global const float* prevVals,
    global float* prevDerivatives,
    int feats,
    int prevWidth,
    int prevHeight,
    int outWidth,
    int outHeight,
    int poolWidth,
    int poolHeight
)
{
    int gid = get_global_id(0);
    int feat, outX, outY;
    to3D(gid, &feat, &outX, &outY,   feats, outWidth);

    int poolStartX = outX * poolWidth;
    int poolStartY = outY * poolHeight;

    int maxPrevX = poolStartX;
    int maxPrevY = poolStartY;
    float maxVal = prevVals[get3D(feat, poolStartX, poolStartY,   feats, prevWidth)];

    for(int prevX = poolStartX; prevX < min(poolStartX + poolWidth, prevWidth); prevX++)
    for(int prevY = poolStartY; prevY < min(poolStartY + poolHeight, prevHeight); prevY++)
    {
        prevDerivatives[get3D(feat, prevX, prevY, feats, prevWidth)] = 0;

        float curPrevVal = prevVals[get3D(feat, prevX, prevY,   feats, prevWidth)];
        if(curPrevVal > maxVal)
        {
            maxVal = curPrevVal;
            maxPrevX = prevX;
            maxPrevY = prevY;
        }
    }

    prevDerivatives[get3D(feat, maxPrevX, maxPrevY, feats, prevWidth)] = outErrDerivatives[get3D(feat, outX, outY, feats, outWidth)];
}