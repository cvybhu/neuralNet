
kernel void doReLU(
    global const float* in,
    global float* out
)
{
    int gid = get_global_id(0);
    out[gid] = (in[gid] > 0.f ? in[gid] : 0.f);
}