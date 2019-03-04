#include "MaxPoolLayer.hpp"

MaxPoolLayer::MaxPoolLayer(ClData data, int features, int width, int height, std::pair<int,int> poolSizeIn)
    : Layer(Type::MaxPool, data, features, width, height),
      poolSize(poolSizeIn)
{
    forwardKernel = createKernelFromSources<cl::Buffer, cl::Buffer, int, int, int, int, int, int, int>(
        "doMaxPool",
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/doMaxPool.cl"}, 
        clData);

    backpropKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int, int>(
        "maxPoolBackprop",
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/maxPoolBackprop.cl"}, 
        clData);
    //maxPoolBackprop
}

void MaxPoolLayer::setPrevious(Layer* l)
{
    prev = l;

    assert(prev->vals.dimensions == 3);
    assert(prev->vals.size[0] == this->vals.size[0]);
    assert(ceil((float)prev->vals.size[1] / (float)poolSize.first) == this->vals.size[1]);
    assert(ceil((float)prev->vals.size[2] / (float)poolSize.second) == this->vals.size[2]);
}

void MaxPoolLayer::forward()
{
    (*forwardKernel)(cl::EnqueueArgs(*clData.queue, vals.totalSize), 
        *prev->vals.clBuff,
        *vals.clBuff,
        vals.size[0],
        prev->vals.size[1],
        prev->vals.size[2],
        vals.size[1],
        vals.size[2],
        poolSize.first,
        poolSize.second).wait();

    /*kernel void doMaxPool(
    global const float* prevVals,
    global float* outVals,
    int feats,
    int prevWidth,
    int prevHeight,
    int outWidth,
    int outHeight,
    int poolWidth,
    int poolHeight)*/
}


void MaxPoolLayer::backprop()
{
    (*backpropKernel)(cl::EnqueueArgs(*clData.queue, vals.totalSize), 
        *errorDerivative.clBuff,
        *prev->vals.clBuff,
        *prev->errorDerivative.clBuff,
        vals.size[0],
        prev->vals.size[1],
        prev->vals.size[2],
        vals.size[1],
        vals.size[2],
        poolSize.first,
        poolSize.second).wait();

    /*kernel void maxPoolBackprop(
    global const float* outErrDerivatives,
    global const float* prevVals,
    global float* prevDerivatives,
    int feats,
    int prevWidth,
    int prevHeight,
    int outWidth,
    int outHeight,
    int poolWidth,
    int poolHeight)*/

}


void MaxPoolLayer::updateWeights(float learnRate)
{} //no weights here

