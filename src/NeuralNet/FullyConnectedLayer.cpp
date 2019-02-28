#include "FullyConnectedLayer.hpp"
#include "utils/Logger.hpp"
#include "utils/utils.hpp"

void FullyConnectedLayer::setPrevious(Layer* l)
{
    prev = l;

    //Generate random weights
    weights = std::optional<Tensor>(std::in_place, prev->vals.totalSize+1, vals.totalSize); // prevLayer + 1 bias node
    weights->createClBuff(clData.context);
    weightsChanges = std::optional<Tensor>(std::in_place, prev->vals.totalSize+1, vals.totalSize);
    weightsChanges->createClBuff(clData.context);

    weights->forEach( [this](float& w){ w = getRandFloat() / (float)prev->vals.totalSize; } );
    weights->loadToGPU(clData.queue);
    
    (*zeroBuffKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(weights->totalSize)), *weightsChanges->clBuff).wait();
}

void FullyConnectedLayer::forward()
{
    (*forwardKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)),
        *prev->vals.clBuff, 
        *weights->clBuff, 
        *vals.clBuff, 
        prev->vals.totalSize+1
    ).wait();   
    /*
    kernel void doFullyConnected(
    global const float* prevVals,
    global const float* weights,
    global float* outVals,
    int prevSize
    )*/
}


void FullyConnectedLayer::backprop()
{
    (*backpropKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(prev->vals.totalSize+1)),
        *errorDerivative.clBuff,
        *weights->clBuff,
        *weightsChanges->clBuff,
        *prev->vals.clBuff,
        *prev->errorDerivative.clBuff,
        prev->vals.totalSize+1,
        vals.totalSize
    ).wait();
    /*
    kernel void fullyConnectedBackprop(
    global float* outErrDerivatives,
    global float* weights,
    global float* weightsChanges,
    global float* prevVals,
    global float* prevDerivatives,
    int prevSize,
    int outSize
    )*/
}

void FullyConnectedLayer::updateWeights(float learnRate)
{
    (*weightsUpdateKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(weights->totalSize)),
        *weights->clBuff, 
        *weightsChanges->clBuff, 
        learnRate
    ).wait();
    /*
    kernel void fullyConnectedUpdateWeights(
    global float* weights,
    global float* weightsChanges,
    float learnRate
    ) */
}