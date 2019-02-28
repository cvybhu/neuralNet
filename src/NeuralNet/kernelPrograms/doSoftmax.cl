__kernel void doSoftmax(
    __global const float* in,
    __global float* out,
    float allExpSum
)
{
    int gid = get_global_id(0);
    out[gid] = exp(in[gid]) / allExpSum;
}