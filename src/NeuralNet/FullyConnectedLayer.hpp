#pragma once
#include "Layer.hpp"

class FullyConnectedLayer : public Layer
{
public:
    template <class... Sizes>
    FullyConnectedLayer(ClData, Sizes...);

    void setPrevious(Layer*);
    void forward();

    void backprop();
    void updateWeights(float learningRate);

//private:
    std::optional<Tensor> weights;
    std::optional<Tensor> weightsChanges;
    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>> forwardKernel;
    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int>> backpropKernel;
    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer, float> > weightsUpdateKernel;
};


template <class... Sizes>
FullyConnectedLayer::FullyConnectedLayer(ClData data, Sizes... sizes)
    : Layer(Type::FullyConnected, data, sizes...)
{
    forwardKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer, int>(
        "doFullyConnected", 
        {
        "src/NeuralNet/kernelPrograms/n-dimGetters.cl",
        "src/NeuralNet/kernelPrograms/doFullyConnected.cl"
        },
        clData);

    backpropKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int>(
        "fullyConnectedBackprop",
        {
        "src/NeuralNet/kernelPrograms/n-dimGetters.cl",
        "src/NeuralNet/kernelPrograms/fullyConnectedBackprop.cl"
        },
        clData);

    weightsUpdateKernel = createKernelFromSources<cl::Buffer, cl::Buffer, float>(
        "fullyConnectedUpdateWeights",
        {
        "src/NeuralNet/kernelPrograms/n-dimGetters.cl",
        "src/NeuralNet/kernelPrograms/fullyConnectedUpdateWeights.cl"
        },
        clData);
}