/*
    Convert n-d coordinates to 1d as buffer data is stored
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

//... etc gonna write more if needed
