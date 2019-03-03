#include "SoftmaxLayer.hpp"

void SoftmaxLayer::setPrevious(Layer* prevLayer)
{
    prev = prevLayer;
}

void SoftmaxLayer::forward()
{
    prev->vals.loadFromGPU(clData.queue);

    float allExpSum = 0.f;
    prev->vals.forEach([&allExpSum](float& f)
    {
        allExpSum += exp(f);
    });
    
    (*forwardKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)),
                    *prev->vals.clBuff, *vals.clBuff, allExpSum).wait();
}

void SoftmaxLayer::backprop()
{
    (*backpropKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)),
                    *prev->errorDerivative.clBuff, *vals.clBuff, *errorDerivative.clBuff, prev->vals.totalSize).wait();

}

void SoftmaxLayer::updateWeights(float)
{} // no weights here