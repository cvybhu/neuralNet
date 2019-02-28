#include "utils.hpp"
#include <fstream>

std::string readFile(const char* filePath)
{
    std::string res;
    std::ifstream file(filePath);

    if(!file.is_open())
    {
        logg << "[ERROR] - Couldnt open file " << filePath << "!!!\n";
        std::exit(1);
    }

    std::string curLine;
    while(std::getline(file, curLine))
        res += curLine + "\n";

    return res;
}