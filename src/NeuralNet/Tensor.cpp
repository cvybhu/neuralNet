#include "Tensor.hpp"
#include <functional>

Tensor::~Tensor()
{
    if(size)
        delete[] size;

    if(mem)
        delete[] mem;
}

Tensor::Tensor(const Tensor& b)
    : dimensions(b.dimensions),
      totalSize(b.totalSize)
{
    int* sizeMem = new int[dimensions];
    memcpy(sizeMem, b.size, dimensions*sizeof(int));
    size = sizeMem;
    mem = new float[totalSize];

    *this = b;
}

Tensor::Tensor(Tensor&& b)
    : dimensions(b.dimensions),
      totalSize(b.totalSize)
{
    int* sizeMem = new int[dimensions];
    memcpy(sizeMem, b.size, dimensions*sizeof(int));
    size = sizeMem;
    mem = new float[totalSize];

    *this = std::move(b);
}

void Tensor::createClBuff(cl::Context* context)
{
    clBuff = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(float) * totalSize);
}

void Tensor::loadToGPU(cl::CommandQueue* queue)
{
    assert(clBuff.has_value());
    queue->enqueueWriteBuffer(*clBuff, CL_TRUE, 0, sizeof(float) * totalSize, mem);
}

void Tensor::loadFromGPU(cl::CommandQueue* queue)
{
    assert(clBuff.has_value());
    queue->enqueueReadBuffer(*clBuff, CL_TRUE, 0, sizeof(float) * totalSize, mem);
}

bool Tensor::isSameSize(const Tensor& b) const
{
    if(dimensions != b.dimensions)
        return false;

    for(int d = 0; d < dimensions; d++)
        if(size[d] != b.size[d])
            return false;

    return true;
}


Tensor& Tensor::operator=(const Tensor& b)
{
    assert(this->isSameSize(b));

    memcpy(mem, b.mem, sizeof(float) * totalSize);

    return *this;
}

Tensor& Tensor::operator=(Tensor&& b)
{
    assert(this->isSameSize(b));

    std::swap(mem, b.mem);
    std::swap(clBuff, b.clBuff);

    return *this;
}

void Tensor::iter(const std::function< void(const std::vector<int>&) >& func) const
{
    std::vector<int> coords(dimensions, 0);

    std::function<void(int)> doDim = [&doDim, &func, &coords, this](int curDim)
    {
        for(int i = 0; i < this->size[curDim]; i++)
        {
            coords[curDim] = i;
            if(curDim+1 == dimensions)
                func(coords);
            else
                doDim(curDim+1);
        }
    };

    doDim(0);
}

void Tensor::forEach(const std::function<void(float&)>& func)
{
    for(int i = 0; i < totalSize; i++)
        func(mem[i]);
}

void Tensor::forEach2(const std::function<void(float&, float&)>& func, Tensor& b)
{
    assert(this->isSameSize(b));

    for(int i = 0; i < totalSize; i++)
        func(mem[i], b.mem[i]);
}

void Tensor::forEach3(const std::function<void(float&, float&, float&)>& func, Tensor& b, Tensor& c)
{
    assert(this->isSameSize(b) && b.isSameSize(c));

    for(int i = 0; i < totalSize; i++)
        func(mem[i], b.mem[i], c.mem[i]);
}

void Tensor::forEach3(const std::function<void(float, float&, float&)>& func, Tensor& b, Tensor& c) const
{
    assert(this->isSameSize(b) && b.isSameSize(c));

    for(int i = 0; i < totalSize; i++)
        func(mem[i], b.mem[i], c.mem[i]);
}


std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    os << t.dimensions << " dimensional Tensor(";

    for(int d = 0; d < t.dimensions; d++)
    {
        os << t.size[d];
        if(d+1 != t.dimensions)
            os << 'x';
    }
    os << "):\n";

    std::vector<int> coords(t.dimensions);

    std::function<void(int)> write = [&t, &os, &write, &coords](int curDim)
    {
        //If dimensions left to write <= 2 then just write 2d array
        if(curDim+2 >= t.dimensions)
        {
            for(int y = 0; y < t.size[t.dimensions-1]; y++)
            {
                coords.back() = y;

                if(coords.size() > 1)
                {
                    for(int x = 0; x < t.size[t.dimensions-2]; x++)
                    {
                        coords[t.dimensions-2] = x;

                        os << t.get(coords) << ' ';
                    }
                }
                else
                    os << y << ": " << t.get(coords) << ' ';

                os << '\n';
            }
        }
        else
        {
            for(int i = 0; i < t.size[curDim]; i++)
            {
                coords[curDim] = i;
                for(int d = 0; d <= curDim; d++)
                    os << "[" << coords[d] << "]";
                os << ":\n";

                write(curDim+1);
            }            
        }
    };

    write(0);

    return os;
}

