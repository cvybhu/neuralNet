#include "ConvoluteLayer.hpp"

ConvoluteLayer::ConvoluteLayer(ClData data, int featuresNumber, int width, int height, std::pair<int,int> kernelSizeIn)
    : Layer(Type::Convolute, data, featuresNumber, width, height),
      kernelSize(kernelSizeIn),
      biases(featuresNumber),
      biasesChanges(featuresNumber)
{
    forwardKernel = createKernelFromSources<cl::Buffer, int, int, int, cl::Buffer, int, int, cl::Buffer, int, int, int, cl::Buffer>(
        "doConvoluted", 
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/doConvoluted.cl"}, 
        clData);

    biases.forEach([](float& f){f = getRandFloat();});
    biases.createClBuff(clData.context);
    biases.loadToGPU(clData.queue);

    //biasesChanges.
}
        
void ConvoluteLayer::setPrevious(Layer* l)
{
    prev = l;

    weights = std::optional<Tensor>(std::in_place, prev->vals.size[0], vals.size[0], kernelSize.first, kernelSize.second);
    weights->forEach([this](float& w){w = getRandFloat() / (float)(kernelSize.first * kernelSize.second);});
    weights->createClBuff(clData.context);
    weights->loadToGPU(clData.queue);
}


void ConvoluteLayer::forward()
{
    (*forwardKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)),
                    *prev->vals.clBuff,
                    prev->vals.size[0],
                    prev->vals.size[1],
                    prev->vals.size[2],
                    *weights->clBuff,
                    kernelSize.first,
                    kernelSize.second,
                    *vals.clBuff,
                    vals.size[0],
                    vals.size[1],
                    vals.size[2],
                    *biases.clBuff);
    
    /*
    kernel void doConvoluted(
    global const float* prevVals,
    int prevFeats,
    int prevWidth,
    int prevHeight,
    global const float* weights,
    int kernelWidth,
    int kernelHeight
    global float* outVals,
    int outFeats,
    int width,
    int height
    */
}

void ConvoluteLayer::backprop()
{


}

void ConvoluteLayer::updateWeights(float learningRate)
{


}