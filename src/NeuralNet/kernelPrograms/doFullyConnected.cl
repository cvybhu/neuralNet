
kernel void doFullyConnected(
    global const float* prevVals,
    global const float* weights,
    global float* outVals,
    int prevSize
)
{
    int out = get_global_id(0);
    float outVal = 0;

    for(int curPrev = 0; curPrev < prevSize-1; curPrev++)
        outVal += prevVals[curPrev] * weights[get2D(curPrev, out, prevSize)];

    outVal += 1.f * weights[get2D(prevSize-1, out, prevSize)]; //Bias node

    outVals[out] = outVal;
}