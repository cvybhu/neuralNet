#include "utils/Logger.hpp"
#include "NeuralNet/NeuralNet.hpp"
#include "DataReader.hpp"
#include <thread>

std::string imageToStr(Tensor img);

void mnistDigitsNet()
{
//Setup neural network
    logg << "Creating neural network...\n";
    NeuralNet net;
    net.addInputLayer(1, 28, 28);
    net.addConvoluteLayer(1, 28, 28, {1, 1});
    net.addFullyConnectedLayer(50);
    net.addReLULayer(50);
    net.addFullyConnectedLayer(10);
    net.addSigmoidLayer(10);
    net.addSoftmaxLayer(10);
    logg << "Done\n";


//Load train data
    logg << "Loading MNIST Training images & labels...\n";
    std::vector<Tensor>&& trainImages = readImages("data/mnistDigits/train-images.bin");
    std::vector<char>&& trainLabels = readLabels("data/mnistDigits/train-labels.bin");
    assert(trainImages.size() == trainLabels.size());
    logg << "Done - loaded " << trainImages.size() << " train cases\n";
    logg << "Example image (label: " << (int)trainLabels.front() << ")\n";
    logg << imageToStr(trainImages.front());


//Load test data
    logg << "Loading MNIST Test images & labels...\n";
    std::vector<Tensor>&& testImages = readImages("data/mnistDigits/test-images.bin");
    std::vector<char>&& testLabels = readLabels("data/mnistDigits/test-labels.bin");
    assert(testImages.size() == testLabels.size());
    logg << "Done - loaded " << testImages.size() << " test cases\n";
    logg << "Example image (label: " << (int)testLabels.front() << ")\n";
    logg << imageToStr(testImages.front());


//Start training
    float learningRate = 0.1f;

    float err = 1337.f;
    int batchSize = 100;

    Tensor expectedOut(10); // To not alloc one every loop
    for(int iter = 1; err > 0.f; iter++)
    {
    //Do training
        err = 0;
        for(int t = 0; t < batchSize; t++)
        {
            int curCase = rand() % trainImages.size();

            net.feed(trainImages[curCase]);

            for(int i = 0; i < 10; i++)
                expectedOut.get(i) = (trainLabels[curCase] == i ? 1 : 0);
            err += net.backpropCrossEntropy(expectedOut);

            net.updateWeights(learningRate);
        }

    //Write training results
        logg << "============= ITER: " << iter << " (" << iter*batchSize << " train examples)===========\n";
        logg << "ERROR: " << err << '\n';
        net.feed(trainImages.front());
        logg << "net(traingImages[0]):" << net.getOut();
        
    //Test model on test images
        if(iter % 600 == 0)
        {
            logg << "[ITER = " << iter <<"] ---- testing...\n";
            int goodOuts = 0;
            for(int i = 0; i < (int)testImages.size(); i++)
            {
                net.feed(testImages[i]);
                Tensor output = net.getOut();
                int which = 0;
                for(int j = 0; j < 10; j++)
                    if(output.get(j) > output.get(which))
                        which = j;

                if(which == (int)testLabels[i])
                    goodOuts++;
            }

            logg << "Done! Accuracy: " << (float)goodOuts/(float)testImages.size() * 100.f << "%\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}




std::string imageToStr(Tensor img)
{   
    std::string res = "";

    auto intensity2Char = [](float intensity)->char
    {
        if(intensity < 0.4)
            return ' ';

        if(intensity < 0.8)
            return '%';
        
        return '#';
    };

    for(int y = 0; y < img.size[2]; y++)
    {
        for(int x = 0; x < img.size[1]; x++)
            res += intensity2Char(img.get(0, x, y));

        res += '\n';
    }

    return res;
}