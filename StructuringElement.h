#ifndef MORPHOPSOPENMP_STRUCTURINGELEMENT_H
#define MORPHOPSOPENMP_STRUCTURINGELEMENT_H

#endif //MORPHOPSOPENMP_STRUCTURINGELEMENT_H

class StructElem{

protected:
    int radius_x, radius_y, height, width;
    StructElem();

public:
    float *data;
    StructElem(int radius, float* data);
    StructElem(int radius_x, int radius_y);
    StructElem(int radius_x, int radius_y, float *data);
    ~StructElem();
    void setData(float* data);
    int get_radius_x();
    int get_radius_y();
    int get_width();
    int get_height();
    void print();
};

class SquareShaper_SE : public StructElem{
public:
    SquareShaper_SE(int radius);
};

class DiamondShape_SE: public StructElem{
public:
    DiamondShape_SE(int radius);
};

class LineShape_SE: public StructElem{
public:
    LineShape_SE(int lenght, int angle);
};

class CircleShape_SE : public StructElem{
public:
    CircleShape_SE(int radius);
};

