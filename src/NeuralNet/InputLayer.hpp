#pragma once
#include "Layer.hpp"
#include "utils/Logger.hpp"


class InputLayer : public Layer
{
public:
    template <class... Sizes>
    InputLayer(ClData, Sizes...);

    void setVals(const Tensor&);

    // These dont really do anything
    void setPrevious(Layer*);
    void forward();

    void backprop();
    void updateWeights(float learnRate);

//private:
    
};

template <class... Sizes>
InputLayer::InputLayer(ClData data, Sizes... sizes)
    : Layer(Type::Input, data, sizes...)
{}
