# NeuralNet
Neural Net 'library' written in C++ / OpenCL  
It is not very fast but it works

## Features:

#### Layer types:
* Fully Connected
* Convolute
* Max Pool

#### Activation functions:
* Sigmoid
* ReLU
* Leaky ReLU

#### Loss functions:
* Mean Square Error
* Cross Entropy (log loss)

## Examples:
* XOR net [here](src/xorNet/xorNet.cpp)
* MNIST handwritten digits net [here](src/mnistDigitsNet/mnistDigitsNet.cpp)

## Building:
To build on linux run `make release` or `make debug`  
To run do `./main`  
