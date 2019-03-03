#include "ConvoluteLayer.hpp"

ConvoluteLayer::ConvoluteLayer(ClData data, int featuresNumber, int width, int height, std::pair<int,int> kernelSizeIn)
    : Layer(Type::Convolute, data, featuresNumber, width, height),
      kernelSize(kernelSizeIn),
      biases(featuresNumber),
      biasesChanges(featuresNumber, width, height)
{
    forwardKernel = createKernelFromSources<cl::Buffer, int, int, int, cl::Buffer, int, int, cl::Buffer, int, int, int, cl::Buffer>(
        "doConvoluted", 
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/doConvoluted.cl"}, 
        clData);

    backpropValsKernel = createKernelFromSources<cl::Buffer, int, int, int, cl::Buffer, int, int, cl::Buffer, int, int, int>(
        "convolutedValsBackprop",
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/convolutedValsBackprop.cl"},
        clData);

    backpropWeightsKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int>(
        "convolutedWeightsBackprop",
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/convolutedWeightsBackprop.cl"},
        clData);

    weightsUpdateKernel = createKernelFromSources<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int, float>(
        "convolutedUpdateWeights",
        {"src/NeuralNet/kernelPrograms/n-dimGetters.cl",
         "src/NeuralNet/kernelPrograms/convolutedUpdateWeights.cl"},
        clData);

    biases.forEach([](float& f){f = getRandFloat();});
    biases.createClBuff(clData.context);
    biases.loadToGPU(clData.queue);

    biasesChanges.createClBuff(clData.context);
    biasesChanges.loadToGPU(clData.queue);
    (*zeroBuffKernel)(cl::EnqueueArgs(*clData.queue, biasesChanges.totalSize), *biasesChanges.clBuff).wait();
}
        
void ConvoluteLayer::setPrevious(Layer* l)
{
    prev = l;

    weights = std::optional<Tensor>(std::in_place, prev->vals.size[0], vals.size[0], kernelSize.first, kernelSize.second);
    weights->forEach([this](float& w){w = getRandFloat() / (float)(kernelSize.first * kernelSize.second);});
    weights->createClBuff(clData.context);
    weights->loadToGPU(clData.queue);

    weightsChanges = std::optional<Tensor>(std::in_place, prev->vals.size[0], vals.size[0], kernelSize.first, kernelSize.second, vals.size[1], vals.size[2]);
    weightsChanges->createClBuff(clData.context);
    weightsChanges->loadToGPU(clData.queue);
    (*zeroBuffKernel)(cl::EnqueueArgs(*clData.queue, weightsChanges->totalSize), *weightsChanges->clBuff).wait();
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
                    *biases.clBuff).wait();
    
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
    //First backprop vals beacuse its easier
    (*backpropValsKernel)(cl::EnqueueArgs(*clData.queue, prev->vals.totalSize),
            *prev->errorDerivative.clBuff,
            prev->vals.size[0],
            prev->vals.size[1],
            prev->vals.size[2],
            *weights->clBuff,
            kernelSize.first,
            kernelSize.second,
            *errorDerivative.clBuff,
            vals.size[0],
            vals.size[1],
            vals.size[2]).wait();
            
    /* kernel void convolutedValsBackprop(
    global float* prevErrDerivatives,
    int prevFeats,
    int prevWidth,
    int prevHeight,
    global const float* weights,
    int kernelWidth,
    int kernelHeight,
    global const float* outErrDerivatives,
    int outFeats,
    int outWidth,
    int outHeight
    )*/


    //Then backprop weights (hard part)
    (*backpropWeightsKernel)(cl::EnqueueArgs(*clData.queue, vals.totalSize),
            *prev->vals.clBuff,
            *errorDerivative.clBuff,
            *weightsChanges->clBuff,
            *biasesChanges.clBuff,
            prev->vals.size[0],
            vals.size[0],
            vals.size[1],
            vals.size[2],
            kernelSize.first,
            kernelSize.second
            ).wait();

    /* kernel void convolutedWeightsBackprop(
    global const float* prevVals,  //[prevFeat][prevX][prevY]
    global const float* outErrDerivatives,
    global float* weightsChanges, //[prevFeat][outFeat][kernelX][kernelY][outX][outY]
    global float* biasesChanges,  //[outFeat][outX][outY]
    int prevFeats,
    int outFeats,
    int width,
    int height,
    int kernelWidth,
    int kernelHeight
    )*/

    // Now i have to sum weightsChanges and biasesChanges... lets do it on CPU for now and check if it even works
    weightsChanges->loadFromGPU(clData.queue);
    biasesChanges.loadFromGPU(clData.queue);

    for(int prevFeat = 0; prevFeat < prev->vals.size[0]; prevFeat++)
    for(int outFeat = 0 ; outFeat < vals.size[0]; outFeat++)
    for(int kernelX = 0; kernelX < kernelSize.first; kernelX++)
    for(int kernelY = 0; kernelY < kernelSize.second; kernelY++)
    {
        float curSum = 0;

        for(int x = 0; x < vals.size[1]; x++)
        for(int y = 0; y < vals.size[2]; y++)
            curSum += weightsChanges->get(prevFeat, outFeat, kernelX, kernelY, x, y);

        weightsChanges->get(prevFeat, outFeat, kernelX, kernelY, 0, 0) = curSum;
    }

    for(int outFeat = 0; outFeat < vals.size[0]; outFeat++)
    {
        float curSum = 0;
        
        for(int x = 0; x < vals.size[1]; x++)
        for(int y = 0; y < vals.size[2]; y++)
            curSum += biasesChanges.get(outFeat, x, y);

        biasesChanges.get(outFeat, 0, 0) = curSum;
    }

    weightsChanges->loadToGPU(clData.queue);
    biasesChanges.loadToGPU(clData.queue);
}

void ConvoluteLayer::updateWeights(float learningRate)
{
    (*weightsUpdateKernel)(cl::EnqueueArgs(*clData.queue, prev->vals.size[0]*vals.size[0]),
        *weights->clBuff,
        *weightsChanges->clBuff,
        *biases.clBuff,
        *biasesChanges.clBuff,
        prev->vals.size[0],
        vals.size[0],
        vals.size[1],
        vals.size[2],
        kernelSize.first,
        kernelSize.second,
        learningRate
    ).wait();

    /*kernel void convolutedUpdateWeights(
        global float* weights,
        global float* weightsChanges,
        global float* biases,
        global float* biasesChanges,
        int prevFeats,
        int outFeats,
        int width,
        int height,
        int kernelWidth,
        int kernelHeight,
        float learningRate
    )*/

    (*zeroBuffKernel)(cl::EnqueueArgs(*clData.queue, weightsChanges->totalSize), *weightsChanges->clBuff).wait();
    (*zeroBuffKernel)(cl::EnqueueArgs(*clData.queue, biasesChanges.totalSize), *biasesChanges.clBuff).wait();
}