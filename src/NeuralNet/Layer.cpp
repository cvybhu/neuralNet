#include "Layer.hpp"


float Layer::getRandFloat()
{
     static std::default_random_engine generator(time(NULL));
     static std::uniform_real_distribution<float> distribution(0.000000001, 1);
     return distribution(generator);
}

void Layer::resetDerivatives()
{
     (*zeroBuffKernel)(cl::EnqueueArgs(*clData.queue, cl::NDRange(vals.totalSize)), *errorDerivative.clBuff).wait();
}