#pragma once
#include "Tensor.hpp"
#include "utils/utils.hpp"
#include <random>

class NeuralNet;

struct ClData
{
    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Device* device;
};

class Layer
{
public:
    enum struct Type{
        Input,
        FullyConnected,
        ApplyFunc,
        SoftMax,
        Convolute,
        MaxPool
    };

    const Type type;

    template <class... Sizes>
    Layer(Type t, ClData data,
          Sizes... valsSizes);

    virtual void setPrevious(Layer*) = 0;
    virtual void forward() = 0;

    void resetDerivatives();
    virtual void backprop() = 0;
    virtual void updateWeights(float learningRate) = 0;

    virtual ~Layer() = default;

    Layer* prev;

    ClData clData;

    Tensor vals;
    Tensor errorDerivative;

    template <class... ArgsTypes>
    static cl::make_kernel<ArgsTypes...> 
    createKernelFromSources(const char* kernelName, 
                            const std::vector<const char*>& sourcePaths,
                            ClData clData);

    
    static float getRandFloat(); //gets random float in [-1;1] range

    std::optional<cl::make_kernel<cl::Buffer>> zeroBuffKernel;

    friend NeuralNet;
};


template <class... Sizes>
Layer::Layer(Type t, ClData data, Sizes... valsSizes)
    : 
    type(t), 
    clData(data),
    vals(valsSizes...),
    errorDerivative(valsSizes...)
{
    prev = nullptr;
    vals.createClBuff(clData.context);
    errorDerivative.createClBuff(clData.context);
    zeroBuffKernel = createKernelFromSources<cl::Buffer>("zeroBuff", {"src/NeuralNet/kernelPrograms/zeroBuff.cl"}, clData);
    resetDerivatives();
}


template <class... ArgsTypes>
cl::make_kernel<ArgsTypes...> 
Layer::createKernelFromSources(const char* kernelName, 
                        const std::vector<const char*>& sourcePaths,
                        ClData clData)
{
    cl::Program::Sources sources;
    std::vector<std::string> sourceCodes;

    for(const char* sourcePath : sourcePaths)
        sourceCodes.emplace_back(std::move(readFile(sourcePath)));

    for(std::string& code : sourceCodes)
        sources.push_back({code.c_str(), code.length()});

    cl::Program program(*clData.context, sources);
    if(program.build({*clData.device}) != CL_SUCCESS)
    {
        logg <<" Error building: "
        << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*clData.device) << "\n";
        std::exit(1);
    }

    return cl::make_kernel<ArgsTypes...>(program, kernelName);
}
