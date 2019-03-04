#include "NeuralNet.hpp"
#include "ConvoluteLayer.hpp"
#include "MaxPoolLayer.hpp"

NeuralNet::NeuralNet() : myLogg(logg)
{
//OpenCL setup
    myLogg << "NeuralNet OpenCL setup...\n";

    //get all platforms (drivers)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if(platforms.empty())
    {
        myLogg << " No OpenCL 1.2 platforms found. Check OpenCL installation!\n";
        std::exit(1);
    }

    logg << " Found OpenCL platforms:\n";
    for(auto& p : platforms)
        myLogg << "  >" << p.getInfo<CL_PLATFORM_NAME>() << '\n';

    cl::Platform defaultPlatform = platforms[0];
    myLogg << "  Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    std::vector<cl::Device> devices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.empty()){
        myLogg << " No devices found. Check OpenCL 1.2 installation!\n";
        exit(1);
    }

    std::cout << " Platform's devices:\n";
    for(auto&& d : devices)
        myLogg << "  >" << d.getInfo<CL_DEVICE_NAME>() << "\n";


    clDevice = devices[0];
    myLogg << "  Using device: "<< clDevice.getInfo<CL_DEVICE_NAME>()<<"\n";

    clContext = cl::Context({clDevice});
    clCommandQueue = cl::CommandQueue(clContext, clDevice);

    clData.context = &clContext;
    clData.device = &clDevice;
    clData.queue = &clCommandQueue;

    myLogg << "done!\n";
}


NeuralNet::~NeuralNet()
{
    for(Layer* l : layers)
        delete l;
    layers.clear();
}


void NeuralNet::feed(const Tensor& input)
{
    assert(!layers.empty());
    assert(layers.front()->type == Layer::Type::Input);

    ((InputLayer*)layers.front())->setVals(input);
    for(int i = 1; i < (int)layers.size(); i++)
        layers[i]->forward();
}

const Tensor& NeuralNet::getOut() const
{
    assert(!layers.empty());
    layers.back()->vals.loadFromGPU(&clCommandQueue);
    return layers.back()->vals;
}

void NeuralNet::updateWeights(float learnRate)
{
    for(Layer* l : layers)
        l->updateWeights(learnRate);
}

float NeuralNet::backpropMSE(const Tensor& expectedOut)
{    
    assert(expectedOut.isSameSize(layers.back()->vals));
    float n = expectedOut.totalSize;

    for(Layer* l : layers)
        l->resetDerivatives();

    layers.back()->vals.loadFromGPU(&clCommandQueue);

    float err = 0.f;

    expectedOut.forEach3([&n, &err](float targetOut, float& out, float& errDerivative)
    {
        errDerivative = 2 * (targetOut - out) / n;
        err += (targetOut - out) * (targetOut - out);
    }, layers.back()->vals, layers.back()->errorDerivative);

    layers.back()->errorDerivative.loadToGPU(&clCommandQueue);

    for(int i = (int)layers.size()-1; i > 0; i--)
        layers[i]->backprop();

    return err;
}

float NeuralNet::backpropCrossEntropy(const Tensor& expectedOut)
{
    assert(expectedOut.isSameSize(layers.back()->vals));

    for(Layer* l : layers)
        l->resetDerivatives();

    layers.back()->vals.loadFromGPU(&clCommandQueue);

    float err = 0.f;

    expectedOut.forEach3([&err](float targetOut, float& out, float& errorDerivative)
    {
        out = std::max(out, 0.00000000001f);

        err -= targetOut * log(out);
        errorDerivative = targetOut / out;

    }, layers.back()->vals, layers.back()->errorDerivative);
    
    layers.back()->errorDerivative.loadToGPU(&clCommandQueue);

    for(int i = (int)layers.size()-1; i > 0; i--)
        layers[i]->backprop();

    return err;
}

void NeuralNet::addConvoluteLayer(int featuresNumber, int width, int height, std::pair<int,int> kernelSize)
{
    assert(!layers.empty());
    assert(layers.back()->vals.dimensions == 3);
    assert(layers.back()->vals.size[1] == width);
    assert(layers.back()->vals.size[2] == height);

    addLayer<ConvoluteLayer>(clData, featuresNumber, width, height, kernelSize);
}

void NeuralNet::addMaxPoolLayer(int featuresNumber, int width, int height, std::pair<int,int> poolSize)
{
    assert(!layers.empty());
    addLayer<MaxPoolLayer>(clData, featuresNumber, width, height, poolSize);
}


void NeuralNet::dump(int fromLayer) const
{
    myLogg << "====DUMP====\n";
    for(int l = 0; l < (int)layers.size(); l++)
    {
        if(l < fromLayer)
            continue;

        Layer* curLayer = layers[l];
        myLogg << "Layer #" << l << ":\n";
        curLayer->vals.loadFromGPU(&clCommandQueue);
        myLogg << curLayer->vals << '\n';

        curLayer->errorDerivative.loadFromGPU(&clCommandQueue);
        myLogg << curLayer->errorDerivative << '\n';

        if(curLayer->type == Layer::Type::FullyConnected)
        {
            FullyConnectedLayer* curFull = (FullyConnectedLayer*)curLayer;

            curFull->weights->loadFromGPU(&clCommandQueue);
            myLogg << "Weights: " << *curFull->weights << '\n';
            curFull->weightsChanges->loadFromGPU(&clCommandQueue);
            myLogg << "WeightsChanges: " << *curFull->weightsChanges << '\n';
        }
    }
}