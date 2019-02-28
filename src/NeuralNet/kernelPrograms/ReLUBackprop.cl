__kernel void ReLUBackprop(
    __global float* prevDerivatives,
    __global const float* outVals,
    __global const float* outErrDerivatives
)
{
    int gid = get_global_id(0);
    
    float funcDerivative = (outVals[gid] > 0.f ? 1.f : 0.f);
    prevDerivatives[gid] = funcDerivative * outErrDerivatives[gid];
}