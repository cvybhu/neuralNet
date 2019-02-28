#pragma once
#include <cassert>
#include <optional>
#include <ostream>
#include "utils/Logger.hpp"
#include <CL/cl.hpp>
#include <functional>
#include <type_traits>

struct Tensor
{
    template <class... DimensionSizes, 
    class E = typename std::enable_if<(... && std::is_same<DimensionSizes, int>::value), int>::type> 
    Tensor(DimensionSizes...);          //Tensor(10, 20, 30) - 10x20x30 tensor

    template <class T>  
    Tensor(const std::initializer_list<T>&); //Tensor({{1, 2}, {3, 4}})
    //This has trouble deducing T for >= 2 dimensional init lists... maybe gonna fix later
    //For now have to write Tensor(init_list<init_list<int>>{{1, 2}, {3, 4}}) :C

    Tensor(const Tensor&);
    Tensor(Tensor&&);

    const int dimensions; // number of tensors dimensions
    const int* size; //size[0] size[1] ...

    template <class... Coordinates>
    float& get(Coordinates...);         //tensor.get(2, 3)

    template <class IterableType, class I = typename IterableType::const_iterator>
    float& get(const IterableType& coords); //tensor.get(std::vector{2, 3})

    bool isSameSize(const Tensor&) const;

    Tensor& operator=(const Tensor&);
    Tensor& operator=(Tensor&&);


    template <class... Coordinates>
    const float& get(Coordinates...) const;

    template <class IterableType, class I = typename IterableType::const_iterator>
    const float& get(const IterableType& coords) const;

    void iter(const std::function<void(const std::vector<int>&)>&) const;
    //function will be called for every set of coords

    void forEach(const std::function<void(float&)>&);
    //Does <function> for each element in Tensor

    void forEach2(const std::function<void(float&, float&)>&, Tensor& b);
    //Does <function> for each two corresponding elements - one from this Tensor and one from other
    void forEach3(const std::function<void(float&, float&, float&)>&, Tensor& b, Tensor& c);
    void forEach3(const std::function<void(float, float&, float&)>&, Tensor& b, Tensor& c) const;

    ~Tensor();

    void createClBuff(cl::Context*);
    void loadToGPU(cl::CommandQueue*);
    void loadFromGPU(cl::CommandQueue*);

    std::optional<cl::Buffer> clBuff;

//private:
    float* mem;
    const int totalSize;

    template <class T> struct isInitializerListStruct : std::false_type {};
    template<class T> struct isInitializerListStruct<std::initializer_list<T>> : std::true_type{};
    template <class T> 
    using isInitializerList = isInitializerListStruct<typename std::remove_const<typename std::remove_reference<T>::type>::type >;

    template <class T> static constexpr int getDepth()
    {
        if constexpr(isInitializerList<T>::value)
            return 1 + getDepth<decltype(*begin(std::declval<T>()))>();
        else
            return 0;
    }
    template <class T> static int getTotalSize(const T& t)
    {
        if constexpr(isInitializerList<T>::value)
            return getTotalSize(*begin(t)) * t.size();
        else
            return 1;
    }
    template <class T> void setSizes(int curDim, int* sizeMem,
                                    const std::initializer_list<T>& curList)
    {
        sizeMem[curDim] = curList.size();
        if constexpr (isInitializerList<T>::value)
            setSizes(curDim+1, sizeMem, *begin(curList));
    }
    template <class T> void copyInitList(int curDim, 
                                        std::vector<int>& coords, 
                                        const std::initializer_list<T>& curList)
    {
        if constexpr (isInitializerList<T>::value)
        {
            int index = 0;
            for(auto&& l : curList)
            {
                coords[curDim] = index++;
                copyInitList(curDim+1, coords, l);
            }
        }
        else
        {
            int index = 0;
            for(auto&& f : curList)
            {
                coords.back() = index++;
                get(coords) = f;
            }
        }
    }
};


template <class... DimensionSizes,
class E = typename std::enable_if<(... && std::is_same<DimensionSizes, int>::value), int>::type> 
Tensor::Tensor(DimensionSizes... sizes)
    //Set dimensions number
    : dimensions(sizeof...(sizes)),
      totalSize((... * sizes))
{
    //Set sizes
    int* sizesMem = new int[dimensions];
    size = sizesMem;

    {
        int curSizeIndex = 0;
        auto setSize = [&sizesMem, &curSizeIndex](int nextSize)
            {sizesMem[curSizeIndex++] = nextSize;};

        (..., setSize(sizes));
    }

    //Create tensor memory
    mem = new float[totalSize];
}

template <class T>
Tensor::Tensor(const std::initializer_list<T>& initList)
    : dimensions(getDepth<std::initializer_list<T>>()),
    totalSize(getTotalSize(initList))
{
    int* sizesMem = new int[dimensions];
    size = sizesMem;

    setSizes(0, sizesMem, initList);

    mem = new float[totalSize];

    std::vector<int> coords(dimensions);
    copyInitList(0, coords, initList);
}

template <class... Coordinates>
float& Tensor::get(Coordinates... coords)
{
    assert(sizeof...(coords) == dimensions);

    float* res = mem;
    int curMul = 1;
    int curDim = 0;

    auto addDimToRes = [this, &curDim, &curMul, &res](int dimCoord)
    {
        res += curMul * dimCoord;
        curMul *= size[curDim++];
    };

    (... , addDimToRes(coords));

    return *res;
}

template <class... Coordinates>
const float& Tensor::get(Coordinates... coords) const
{
    assert(sizeof...(coords) == dimensions);

    float* res = mem;
    int curMul = 1;
    int curDim = 0;

    auto addDimToRes = [this, &curDim, &curMul, &res](int dimCoord)
    {
        res += curMul * dimCoord;
        curMul *= size[curDim++];
    };

    (... , addDimToRes(coords));

    return *res;
}

template <class IterableType, class I = typename IterableType::const_iterator>
float& Tensor::get(const IterableType& coords)
{
    float* res = mem;
    int curMul = 1;
    int curDim = 0;

    auto addDimToRes = [this, &curDim, &curMul, &res](int dimCoord)
    {
        res += curMul * dimCoord;
        curMul *= size[curDim++];
    };

    int coordsSize = 0;
    for(auto it = coords.begin(); it != coords.end(); it++)
    {
        addDimToRes(*it);
        coordsSize++;
    }

    assert(coordsSize == dimensions);

    return *res;
}


template <class IterableType, class I = typename IterableType::const_iterator>
const float& Tensor::get(const IterableType& coords) const
{
    float* res = mem;
    int curMul = 1;
    int curDim = 0;

    auto addDimToRes = [this, &curDim, &curMul, &res](int dimCoord)
    {
        res += curMul * dimCoord;
        curMul *= size[curDim++];
    };

    int coordsSize = 0;
    for(auto it = coords.begin(); it != coords.end(); it++)
    {
        addDimToRes(*it);
        coordsSize++;
    }

    assert(coordsSize == dimensions);

    return *res;
}

std::ostream& operator<<(std::ostream&, const Tensor&);
