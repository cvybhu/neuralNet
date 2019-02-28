
kernel void doSoftmax(
    global const float* in,
    global float* out,
    float allExpSum
)
{
    int gid = get_global_id(0);
    out[gid] = exp(in[gid]) / allExpSum;
}