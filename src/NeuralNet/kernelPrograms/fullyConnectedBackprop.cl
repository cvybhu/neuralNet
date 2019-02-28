kernel void fullyConnectedBackprop(
    global float* outErrDerivatives,
    global float* weights,
    global float* weightsChanges,
    global float* prevVals,
    global float* prevDerivatives,
    int prevSize,
    int outSize
)
{
    int prev = get_global_id(0);
    float prevVal = (prev == prevSize-1 ? 1 : prevVals[prev]);
    float prevErrDerivative = 0;

    for(int next = 0; next < outSize; next++)
    {
        int curEdge = get2D(prev, next, prevSize);
        float outErrDerivative = outErrDerivatives[next];

        weightsChanges[curEdge] += prevVal * outErrDerivative;

        if(prev != prevSize-1)
            prevErrDerivative += weights[curEdge] * outErrDerivative;
    }

    prevDerivatives[prev] += prevErrDerivative;
}