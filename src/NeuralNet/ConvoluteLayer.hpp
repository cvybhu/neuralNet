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
    const std::pair<int, int> kernelSize;

    std::optional< Tensor > weights;   // weights[prevFeature][myFeature][kernelX][kernelY]
    std::optional< Tensor > weightsChanges; // weightsChanges[prevFeat][outFeat][kernelX][kernelY][x][y]
                                            // what each pixel thinks the weights should change
                                            // This will be averaged and the sum is gonna be in [0][0] for updates
    Tensor biases; //biases[myFeature]
    Tensor biasesChanges; //biasesChanges[myFeature][x][y] <- same as weightsChanges after average res will be in [0][0]

    std::optional< cl::make_kernel<cl::Buffer, int, int, int, cl::Buffer, int, int, cl::Buffer, int, int, int, cl::Buffer> > 
            forwardKernel;

    std::optional< cl::make_kernel<cl::Buffer, int, int, int, cl::Buffer, int, int, cl::Buffer, int, int, int> > 
            backpropValsKernel;
            
    std::optional< cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int> >
            backpropWeightsKernel;

    std::optional< cl::make_kernel<cl::Buffer> >
            sumWeightsUpdatesKernel;

    std::optional< cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int, float> > 
            weightsUpdateKernel;
};