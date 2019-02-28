
kernel void doSigmoid(
    global const float* in,
    global float* out
)
{
    int gid = get_global_id(0);
    out[gid] = 1.f / (1.f + exp(-in[gid]));
}