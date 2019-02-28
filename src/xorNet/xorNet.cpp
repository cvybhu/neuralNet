#include "utils/Logger.hpp"
#include "NeuralNet/NeuralNet.hpp"
#include <iomanip>

void xorNet()
{
    NeuralNet net;
    net.addInputLayer(2);
    net.addFullyConnectedLayer(2);
    net.addSigmoidLayer(2);
    net.addFullyConnectedLayer(2);
    net.addSigmoidLayer(2);
    //net.addSoftmaxLayer(2);
    //net.addLeakyReLULayer(1);


    float learningRate = 0.1;

    logg << std::setprecision(4) << std::fixed;


    float err = 1337.f;;
    for(int iter = 0; err > 1.f; iter++)
    {
        err = 0;

        logg << "================= ITER #" << iter << "=================\n";
        for(int t = 0; t < 100; t++)
        {

            switch (rand()%4)
            {
                case 0:
                    net.feed({0, 0});
                    err += net.backpropMSE({1, 0});
                    break;
                case 1:
                    net.feed({0, 1});
                    err += net.backpropMSE({0, 1});
                    break;
                case 2:
                    net.feed({1, 0});
                    err += net.backpropMSE({0, 1});
                    break;
                case 3:
                    net.feed({1, 1});
                    err += net.backpropMSE({1, 0});
                    break;
            }

            //net.dump();
            net.updateWeights(learningRate);
        }

        //std::exit(1);

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