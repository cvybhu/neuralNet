
kernel void doMaxPool(
    global const float* prevVals,
    global float* outVals,
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

    float res = prevVals[get3D(feat, poolStartX, poolStartY,   feats, prevWidth)];

    for(int prevX = poolStartX; prevX < min(poolStartX + poolWidth, prevWidth); prevX++)
    for(int prevY = poolStartY; prevY < min(poolStartY + poolHeight, prevHeight); prevY++)
        res = max(res, prevVals[get3D(feat, prevX, prevY,   feats, prevWidth)]);

    outVals[get3D(feat, outX, outY,   feats, outWidth)] = res;
}