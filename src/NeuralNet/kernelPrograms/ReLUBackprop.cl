
kernel void ReLUBackprop(
    global float* prevDerivatives,
    global const float* outVals,
    global const float* outErrDerivatives
)
{
    int gid = get_global_id(0);
    
    float funcDerivative = (outVals[gid] > 0.f ? 1.f : 0.f);
    prevDerivatives[gid] = funcDerivative * outErrDerivatives[gid];
}