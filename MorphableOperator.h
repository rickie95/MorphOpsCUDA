/*
 * MorphableOperator.h

 *
 *  Created on: Dec 5, 2018
 *      Author: riccardo_malavolti
 */

#ifndef MORPHABLEOPERATOR_H_
#define MORPHABLEOPERATOR_H_
#include <iostream>
#include <cstdio>
#include <cassert>
#include <chrono>
#include "Image.h"
#include "StructuringElement.h"

#define EROSION 1
#define DILATATION 0
#define TILE_WIDTH 32

__constant__ float deviceSEdata[25 * 25]; // preallocate 2.5KB for structured element.

__global__ void process(float *input_img, float *output_img, int img_W, int img_H,const float *__restrict__ SE, int SE_W, int SE_H, int operation);

__host__ Image_t* erosion(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);
__host__ Image_t* dilatation(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);
__host__ Image_t* opening(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);
__host__ Image_t* closing(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);
__host__ Image_t* topHat(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);
__host__ Image_t* bottomHat(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);

__device__ float max(float* array, int length);
__device__ float min(float* array, int length);

__host__ float max_pixel(float a, float b);
__host__ float min_pixel(float a, float b);

#endif /* MORPHABLEOPERATOR_H_ */
