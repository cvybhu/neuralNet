#include "InputLayer.hpp"

void InputLayer::setVals(const Tensor& input)
{
    clData.queue->enqueueWriteBuffer
            (*vals.clBuff, CL_TRUE, 0, sizeof(float)*vals.totalSize, input.mem);
}


void InputLayer::setPrevious(Layer*)
{}

void InputLayer::forward()
{}

void InputLayer::backprop()
{}

void InputLayer::updateWeights(float learnRate)
{}