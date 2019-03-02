#pragma once
#include "utils/Logger.hpp"
#include <vector>
#include <CL/cl.hpp>
#include "Layer.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "ApplyFuncLayers.hpp"
#include "SoftmaxLayer.hpp"

class NeuralNet
{
public:
    NeuralNet();

    template <class... Sizes> void addInputLayer(Sizes...); //addInputLayer(3, 20, 20) - adds 3x20x20 input layer
    template <class... Sizes> void addFullyConnectedLayer(Sizes...);    
    template <class... Sizes> void addSigmoidLayer(Sizes...);
    template <class... Sizes> void addReLULayer(Sizes...);
    template <class... Sizes> void addLeakyReLULayer(Sizes...);
    template <class... Sizes> void addSoftmaxLayer(Sizes...);
    void addConvoluteLayer(int featuresNumber, int width, int height, std::pair<int,int> kernelSize = {3, 3});

    
    void feed(const Tensor&);
    const Tensor& getOut() const;
    float backpropMSE(const Tensor& expectedOut); //returns MSE
    float backpropCrossEntropy(const Tensor& expectedOut); //returns cross entropy error
    void updateWeights(float learningRate);

    ~NeuralNet();

    void dump(int fromLayer = 0) const;

private:
    //OpenCL stuff
    cl::Context clContext;
    cl::Device clDevice;
    mutable cl::CommandQueue clCommandQueue;
    ClData clData; //For initializing layers

    Logger& myLogg;

    std::vector<Layer*> layers;

    template <class T, class... Args>
    void addLayer(Args&&... args)
    {
        T* newLayer = new T(std::forward<Args>(args)...);
        if(!layers.empty())
            newLayer->setPrevious(layers.back());
        layers.emplace_back(newLayer);
    }
};

template <class... Sizes>
void NeuralNet::addInputLayer(Sizes... sizes)
{
    assert(layers.empty());
    addLayer<InputLayer>(clData, sizes...);
}

template <class... Sizes>
void NeuralNet::addFullyConnectedLayer(Sizes... sizes)
{
    assert(!layers.empty());
    addLayer<FullyConnectedLayer>(clData, sizes...);
}

template <class... Sizes>
void NeuralNet::addSigmoidLayer(Sizes... sizes)
{
    assert(!layers.empty());
    addLayer<SigmoidLayer>(clData, sizes...);
}

template <class... Sizes>
void NeuralNet::addReLULayer(Sizes... sizes)
{
    assert(!layers.empty());
    addLayer<ReLULayer>(clData, sizes...);
}

template <class... Sizes>
void NeuralNet::addLeakyReLULayer(Sizes... sizes)
{
    assert(!layers.empty());
    addLayer<LeakyReLULayer>(clData, sizes...);
}

template <class... Sizes>
void NeuralNet::addSoftmaxLayer(Sizes... sizes)
{
    assert(!layers.empty());
    addLayer<SoftmaxLayer>(clData, sizes...);
}