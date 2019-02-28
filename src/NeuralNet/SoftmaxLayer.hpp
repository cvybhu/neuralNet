#pragma once
#include "Layer.hpp"

class SoftmaxLayer : public Layer
{
public:
    template <class... Sizes>
    SoftmaxLayer(ClData, Sizes...);

    void setPrevious(Layer*);
    void forward();
    void backprop();
    void updateWeights(float learnRate);

private:
    std::optional< cl::make_kernel<cl::Buffer, cl::Buffer, float> > forwardKernel;
    std::optional< cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> > backpropKernel;
};

template <class... Sizes>
SoftmaxLayer::SoftmaxLayer(ClData data, Sizes... sizes)
    : Layer(Type::SoftMax, data, sizes...)
{
    forwardKernel = createKernelFromSources<cl::Buffer, cl::Buffer, float>(
        "doSoftmax",
        {
            "src/NeuralNet/kernelPrograms/doSoftmax.cl"
        },
        clData);

    backpropKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer, int>(
        "softmaxBackprop",
        {
            "src/NeuralNet/kernelPrograms/softmaxBackprop.cl"
        },
        clData);
}