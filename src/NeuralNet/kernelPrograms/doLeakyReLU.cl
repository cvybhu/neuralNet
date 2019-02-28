
__kernel void doLeakyReLU(
    __global const float* in,
    __global float* out
)
{
    int gid = get_global_id(0);
    out[gid] = (in[gid] >= 0.f ? in[gid] : 0.1f * in[gid]);
}