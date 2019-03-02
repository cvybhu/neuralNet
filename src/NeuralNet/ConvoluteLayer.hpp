#pragma once

#include "Layer.hpp"

class ConvoluteLayer : public Layer
{
public:
    ConvoluteLayer(ClData, int featuresNumber, int width, int height, std::pair<int,int> kernelSize);
        
    void setPrevious(Layer*);
    void forward();
    void backprop();
    void updateWeights(float learningRate);

//private:
    std::pair<int, int> kernelSize;

    std::optional< Tensor > weights;   // weights[prevFeature][myFeature][x][y]
    std::optional< Tensor > weightsChanges;
    Tensor biases; //biases[myFeature]
    Tensor biasesChanges;

    std::optional< cl::make_kernel<cl::Buffer, int, int, int, cl::Buffer, int, int, cl::Buffer, int, int, int, cl::Buffer> > 
            forwardKernel;
    std::optional< cl::make_kernel<cl::Buffer> > backpropKernel;
    std::optional< cl::make_kernel<cl::Buffer> > weightsUpdateKernel;
};