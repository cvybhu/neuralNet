kernel void sigmoidBackprop(
    global float* prevDerivatives,
    global const float* outVals,
    global const float* outErrDerivatives
)
{
    int gid = get_global_id(0);
    
    float funcDerivative = outVals[gid] * (1.f - outVals[gid]);
    prevDerivatives[gid] = funcDerivative * outErrDerivatives[gid];
}