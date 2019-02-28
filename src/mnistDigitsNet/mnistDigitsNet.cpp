#include "utils/Logger.hpp"
#include "NeuralNet/NeuralNet.hpp"
#include "DataReader.hpp"
#include <thread>

std::string imageToStr(Tensor img);

void mnistDigitsNet()
{
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

//Setup neural network
    logg << "Creating neural network...\n";
    NeuralNet net;
    net.addInputLayer(28, 28);
    net.addFullyConnectedLayer(50);
    net.addReLULayer(50);
    net.addFullyConnectedLayer(10);
    net.addSigmoidLayer(10);
    net.addSoftmaxLayer(10);
    logg << "Done\n";

//Start training
    float learningRate = 0.01f;

    Tensor expectedOut(10); // To not alloc one every loop
    float err = 1337.f;
    int batchSize = 100;
    for(int iter = 0; iter > -1 && err > 0.f; iter++)
    {
        err = 0;

        for(int t = 0; t < batchSize; t++)
        {
            int curCase = rand() % trainImages.size();

            net.feed(trainImages[curCase]);

            for(int i = 0; i < 10; i++)
                expectedOut.get(i) = (trainLabels[curCase] == i ? 1 : 0);
            err += net.backpropCrossEntropy(expectedOut);

            //net.dump(3);
            net.updateWeights(learningRate);
        }

        logg << "============= ITER: " << iter << " (" << iter*batchSize << " train examples)===========\n";
        logg << "ERROR: " << err << '\n';
        net.feed(trainImages.front());
        logg << "net(traingImages[0]):" << net.getOut();
        float sum = 0;
        Tensor out = net.getOut();
        out.forEach([&sum](float& f){sum += f;});
        logg << "sum: " << sum << '\n';


        if(iter % 600 == 0)
        {
            logg << "[ITER = " << iter <<"] ---- testing...\n";
            int goodOuts = 0;
            for(int i = 0; i < testImages.size(); i++)
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

    for(int y = 0; y < img.size[1]; y++)
    {
        for(int x = 0; x < img.size[0]; x++)
            res += intensity2Char(img.get(x, y));

        res += '\n';
    }

    return res;
}