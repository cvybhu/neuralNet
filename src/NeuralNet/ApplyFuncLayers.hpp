#pragma once
#include "Layer.hpp"

template <const char** forwardKernelName, const char** backpropKernelName>
class ApplyFuncLayer : public Layer
{
public:
    template <class... TensorSizes>
    ApplyFuncLayer(ClData, TensorSizes...);

    void setPrevious(Layer* prevLayer)
    { prev = prevLayer; }

    void forward()
    { (*forwardKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)), *prev->vals.clBuff, *vals.clBuff).wait(); }

    void backprop()
    { (*backpropKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)), *prev->errorDerivative.clBuff, *vals.clBuff, *errorDerivative.clBuff); }

    void updateWeights(float) 
    {} //No weights in here

//private:
    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer> > forwardKernel;
    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> > backpropKernel;
};

template <const char** forwardKernelName, const char** backpropKernelName>
template <class... TensorSizes>
ApplyFuncLayer<forwardKernelName, backpropKernelName>::ApplyFuncLayer(ClData data, TensorSizes... sizes)
    : Layer(Type::ApplyFunc, data, sizes...)
{
    forwardKernel = createKernelFromSources<cl::Buffer, cl::Buffer>(
        *forwardKernelName, 
        {
        (std::string("src/NeuralNet/kernelPrograms/") + *forwardKernelName + ".cl").c_str()
        },
        clData);

    backpropKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer>(
        *backpropKernelName,
        {
        (std::string("src/NeuralNet/kernelPrograms/") + *backpropKernelName + ".cl").c_str()
        },
        clData);
}

namespace FuncKernelsPaths
{
    static const char* sigmoidForward = "doSigmoid";
    static const char* sigmoidBackprop = "sigmoidBackprop";

    static const char* ReLUForward = "doReLU";
    static const char* ReLUBackprop = "ReLUBackprop";

    static const char* leakyReLUForward = "doLeakyReLU";
    static const char* leakyReLUBackprop = "leakyReLUBackprop";
}

using SigmoidLayer = ApplyFuncLayer<&FuncKernelsPaths::sigmoidForward, 
                                    &FuncKernelsPaths::sigmoidBackprop>;

using ReLULayer = ApplyFuncLayer<&FuncKernelsPaths::ReLUForward,
                                 &FuncKernelsPaths::ReLUBackprop>;

using LeakyReLULayer = ApplyFuncLayer<&FuncKernelsPaths::leakyReLUForward,
                                      &FuncKernelsPaths::leakyReLUBackprop>;
