
kernel void doConvoluted(
    global const float* prevVals,
    int prevFeats,
    int prevWidth,
    int prevHeight,
    global const float* weights,
    int kernelWidth,
    int kernelHeight,
    global float* outVals,
    int outFeats,
    int width,
    int height,
    global const float* biases
)
{
//Get current coordinates
    int gid = get_global_id(0);
    int outFeat, x, y;
    to3D(gid, &outFeat, &x, &y, outFeats, width);

    float outVal = 0;

//Iterate over previous vals
    for(int prevFeat = 0; prevFeat < prevFeats; prevFeat++)
    {
        for(int dx = 0; dx < kernelWidth; dx++)
        {

            int prevX = x - kernelWidth/2 + dx;
            if(prevX < 0 || prevX >= prevWidth)
                continue;

            for(int dy = 0; dy < kernelHeight; dy++)
            {
                int prevY = y - kernelHeight/2 + dy;
                if(prevY < 0 || prevY >= prevHeight)
                    continue;

                float prevVal = prevVals[get3D(prevFeat, prevX, prevY,   prevFeats, prevWidth)];
                float curWeight = weights[get4D(prevFeat, outFeat, dx, dy,  prevFeats, outFeats, kernelWidth)];

                outVal += prevVal * curWeight;
            }
        }
    }

    outVal += biases[outFeat];
    
    outVals[gid] = outVal;
}