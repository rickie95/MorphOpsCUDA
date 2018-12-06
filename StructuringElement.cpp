#include "StructuringElement.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

// CLASS StructElem: base class.

StructElem::StructElem() {}

StructElem::StructElem(int radius_x, int radius_y){
    this->radius_x = radius_x;
    this->radius_y = radius_y;
    this->width = 2 * radius_x + 1 ;
    this->height = 2 * radius_y + 1;
    this->data = (float*)malloc(width * height * sizeof(float));
    for(int i = 0; i < height * width; i += 1){
        this->data[i] = 0;
        }
}

StructElem::StructElem(int radius_x, int radius_y, float *data){
    StructElem(radius_x, radius_y);
    free(data);
    this->data = data;
}

StructElem::StructElem(int radius, float* data){
    StructElem(radius, radius);
    free(data);
    this->data = data;
}

StructElem::~StructElem(){
    if(data != NULL)
        free(data);
}

void StructElem::setData(float* data){
    if(data != NULL)
        free(data);
    this->data = data;
}

void StructElem::print(){
    for(int r = 0; r < height; r += 1){
        for(int c = 0; c < width; c +=1)
            printf("%f \t", this->data[r * width + c]);
        printf("\n");
    }
}

// Getters
int StructElem::get_radius_x(){ return this->radius_x; }
int StructElem::get_radius_y(){ return this->radius_y; }
int StructElem::get_width(){ return width; }
int StructElem::get_height(){ return height; }

// CLASS SquareShaper_SE: class derived from StructElem.

SquareShaper_SE::SquareShaper_SE(int radius) : StructElem(radius, radius){
    for(int i = 0; i < width*height; i+=1){
        this->data[i] = 1;
    }
}

DiamondShape_SE::DiamondShape_SE(int radius) : StructElem(radius, radius){
    for(int r = 0; r < height/2 + 1; r += 1){
        for(int c = width - radius_x - 1 - r; c < width - radius_x + r; c += 1){
            this->data[r * width + c] = 1;
            this->data[(height - r - 1) * width + c] = 1;
        }
    }
}

LineShape_SE::LineShape_SE(int length, int angle) : StructElem(){
    // 0°, +-45°, 90°
    int a;
    if(abs(angle) > 360)
        a = std::signbit(angle)*angle%360;

    a = (int)round(angle/45) * 45;
    printf("Angle is approximated with %d°", a);

}

CircleShape_SE::CircleShape_SE(int radius) : StructElem(radius, radius){
    float* t_data = (float *)malloc(height * width * sizeof(float));
    for(int i = 0; i < height; i+=1){
        for(int j = 0; j < width; j+=1){
            int x = j - radius;
            int y = radius - i;
            t_data[i * radius + j] = 0;
            if( x * x + y * y <= radius * radius)
                t_data[i * radius + j] = 1;
            printf("%f \t", t_data[i * radius + j]);
        }
    printf("\n");
    }
    this->setData(t_data);
}

