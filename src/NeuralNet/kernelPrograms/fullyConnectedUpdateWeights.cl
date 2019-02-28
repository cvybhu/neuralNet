
kernel void fullyConnectedUpdateWeights(
    global float* weights,
    global float* weightsChanges,
    float learnRate)
{
    int gid = get_global_id(0);
    weights[gid] += weightsChanges[gid] * learnRate;
    weightsChanges[gid] = 0;
}