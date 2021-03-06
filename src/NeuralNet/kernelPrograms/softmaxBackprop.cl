
kernel void softmaxBackprop(
    global float* prevDerivatives,
    global const float* outVals,
    global const float* outErrDerivatives,
    int prevSize
)
{
    int gid = get_global_id(0);
    
    prevDerivatives[gid] = 0.f;

    for(int next = 0; next < prevSize; next++)
    {
        float curDerivative;

        if(next == gid)
            curDerivative = outVals[next] * (1.f - outVals[next]);
        else
            curDerivative = -outVals[gid] * outVals[next];
        
        curDerivative *= outErrDerivatives[next];

        prevDerivatives[gid] += curDerivative;
    }

}