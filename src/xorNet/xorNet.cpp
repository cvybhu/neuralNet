#include "utils/Logger.hpp"
#include "NeuralNet/NeuralNet.hpp"
#include <iomanip>

void xorNet()
{
    NeuralNet net;
    net.addInputLayer(2);
    net.addFullyConnectedLayer(4);
    net.addLeakyReLULayer(4);
    net.addFullyConnectedLayer(1);
    net.addReLULayer(1);


    float learningRate = 0.1;
    int batchSize = 100;

    float err = 1337.f;;
    for(int iter = 0; err > 1.f; iter++)
    {
        err = 0;

        logg << "================= ITER #" << iter << " (" << iter * batchSize << " test cases done)=================\n";
        for(int t = 0; t < batchSize; t++)
        {

            switch (rand()%4)
            {
                case 0:
                    net.feed({0, 0});
                    err += net.backpropMSE({0});
                    break;
                case 1:
                    net.feed({0, 1});
                    err += net.backpropMSE({1});
                    break;
                case 2:
                    net.feed({1, 0});
                    err += net.backpropMSE({1});
                    break;
                case 3:
                    net.feed({1, 1});
                    err += net.backpropMSE({0});
                    break;
            }

            net.updateWeights(learningRate);
        }

        net.feed({0, 0});
        logg << "net(0, 0) = " << net.getOut() << '\n';
        net.feed({0, 1});
        logg << "net(0, 1) = " << net.getOut() << '\n';
        net.feed({1, 0});
        logg << "net(1, 0) = " << net.getOut() << '\n';
        net.feed({1, 1});
        logg << "net(1, 1) = " << net.getOut() << '\n';
        logg << "ERROR: " << err << '\n';
    }
}