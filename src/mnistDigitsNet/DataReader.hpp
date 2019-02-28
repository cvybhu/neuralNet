#pragma once
#include <vector>
#include "NeuralNet/Tensor.hpp"

std::vector< Tensor > readImages(const char* imagesFile);

std::vector< char > readLabels(const char* labelsFile); // 0-9