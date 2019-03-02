#include "DataReader.hpp"
#include "utils/Logger.hpp"
#include <cstdio>
#include <arpa/inet.h>

std::vector< Tensor > readImages(const char* imagesFile)
{
    auto file = fopen(imagesFile, "rb");

    if(!file)
    {
        logg << "[ERRRRROORRR]: Couldnt open " << imagesFile << "!!\n";
        return {};
    }

    auto readUint = [file, imagesFile]()->unsigned
    {
        unsigned res;
        if(!fread(&res, sizeof(unsigned), 1, file))
        {
            logg << "Error reading file " << imagesFile << "!!\n";
        }
        return ntohl(res);
    };

    unsigned magicNumber = readUint();
    unsigned imagesNmbr = readUint();

    std::vector< Tensor > res;
    res.reserve(imagesNmbr);

    int height = readUint();
    int width = readUint();
    for(int i = 0; i < (int)imagesNmbr; i++)
    {
        std::vector<unsigned char> readBuff(width*height);
        if(!fread(&readBuff[0], 1, width*height, file))
        {
            logg << "Error reading file " << imagesFile << "!!!\n";
        }

        res.emplace_back(1, width, height);
        Tensor& cur = res.back();
        
        for(int x = 0; x < (int)width; x++)
        for(int y = 0; y < (int)height; y++)
        {
            int index = y * width + x;
            cur.get(0, x, y) = ((float)readBuff[index]) / 255.f;
        }
    }

    return res;
}

std::vector<char> readLabels(const char* labelsFile)
{
    auto file = fopen(labelsFile, "rb");

    if(!file)
    {
        logg << "[ERRRRROORRR]: Couldnt open " << labelsFile << "!!\n";
        return {};
    }

    auto readUint = [file, labelsFile]()->unsigned
    {
        unsigned res;
        if(!fread(&res, sizeof(unsigned), 1, file))
        {
            logg << "Error reading file " << labelsFile << "!!\n";
        }
        return ntohl(res);
    };

    unsigned magicNumber = readUint();
    unsigned labelsNmbr = readUint();

    std::vector<char> res(labelsNmbr);

    if(!fread(&res[0], 1, labelsNmbr, file))
        logg << "Error reading file " << labelsFile << "!!!\n";

    return res;
}