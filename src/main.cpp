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

    return 0;
}