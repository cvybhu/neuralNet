/*
    Convert n-d coordinates to 1d as buffer data is stored
    Gets x, y, z, ... and returns 1-D index
*/

int get2D(int x, int y, int sizeX)
{
    int res = 0;
    int curMul = 1;
    res += x * curMul;
    curMul *= sizeX;
    res += y * curMul;
    return res;
}

int get3D(int x, int y, int z, int sizeX, int sizeY)
{
    int res = 0;
    int curMul = 1;
    res += x * curMul;
    curMul *= sizeX;
    res += y * curMul;
    curMul *= sizeY;
    res += z * curMul;
    return res;
}

int get4D(int x, int y, int z, int d, int sizeX, int sizeY, int sizeZ)
{
    int res = 0;
    int curMul = 1;
    res += x * curMul;
    curMul *= sizeX;
    res += y * curMul;
    curMul *= sizeY;
    res += z * curMul;
    curMul *= sizeZ;
    res += d * curMul;
    return res;
}

/*
    convert from 1-D to n-D
    gets 1-D index and sets x, y, z,... appropriately
*/

void to2D(int index, int* x, int* y, int sizeX)
{
    *x = index % sizeX;
    index /= sizeX;
    *y = index;
}

void to3D(int index, int* x, int* y, int* z, int sizeX, int sizeY)
{
     *x = index % sizeX;
     index /= sizeX; 
     *y = index % sizeY;
     index /= sizeY;
     *z = index;
}

void to4D(int index, int* x, int* y, int* z, int* d, int sizeX, int sizeY, int sizeZ)
{
     *x = index % sizeX;
     index /= sizeX; 
     *y = index % sizeY;
     index /= sizeY;
     *z = index % sizeZ;
     index /= sizeZ;
     *d = index;
}

//... etc gonna write more if needed
