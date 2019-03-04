#pragma once
#include "Layer.hpp"


class MaxPoolLayer : public Layer
{
public:
    MaxPoolLayer(ClData, int features, int width, int height, std::pair<int,int> poolSize);

    void setPrevious(Layer*);
    void forward();
    void backprop();
    void updateWeights(float learnRate);

//private:
    std::pair<int,int> poolSize;

    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer, int, int, int, int, int, int, int> >
            forwardKernel;
    std::optional<cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int, int> > 
            backpropKernel;
};