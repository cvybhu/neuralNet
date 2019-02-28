
kernel void zeroBuff(
    global float* buff
)
{
    int gid = get_global_id(0);
    buff[gid] = 0;
}