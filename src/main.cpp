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
    //xorNet();


    int sizeX = 5, sizeY = 4;

    NeuralNet net;
    net.addInputLayer(1, sizeX, sizeY);
    net.addConvoluteLayer(2, sizeX, sizeY);

    Tensor in(1, sizeX, sizeY);
    in.forEach([](float& f){f = 1.f;});

    logg << "Feeding " << in << "...";
    net.feed(in);
    logg << "Got: " << net.getOut() << '\n';

    return 0;
}