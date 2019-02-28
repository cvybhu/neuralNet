#include "utils/Logger.hpp"
#include "NeuralNet/NeuralNet.hpp"
#include "utils/utils.hpp"
#include <iomanip>

void xorNet();
void mnistDigitsNet();

int main()
{
    logg << std::setprecision(6) << std::fixed;
    mnistDigitsNet();
    xorNet();
    return 0;

    NeuralNet net;
    net.addInputLayer(1);
    net.addFullyConnectedLayer(100);
    net.addSigmoidLayer(100);
    net.addFullyConnectedLayer(1);
    net.addSigmoidLayer(1);

    while(true)
    {
        net.feed({0});
        logg << net.getOut() << '\n';
        net.backpropMSE({1});
        net.updateWeights(0.1);
    }


    return 0;
}