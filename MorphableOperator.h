/*
 * MorphableOperator.h

 *
 *  Created on: Dec 5, 2018
 *      Author: riccardo_malavolti
 */

#ifndef MORPHABLEOPERATOR_H_
#define MORPHABLEOPERATOR_H_
#include <chrono>
#include "Image.h"
#include "CUDAUtils.h"
#include "StructuringElement.h"

#define EROSION 1
#define DILATATION 0
#define BLOCK_DIM 16

__constant__ float deviceSEdata[25 * 25]; // preallocate 2.5KB for structured element.

__global__ void process(float *input_img, float *output_img, int img_W, int img_H,
		const float *__restrict__ SE, int SE_W, int SE_H, int operation);

__host__ Image_t* erosion(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);
__host__ Image_t* dilatation(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span);

__device__ float max(float* array, int length);
__device__ float min(float* array, int length);

#endif /* MORPHABLEOPERATOR_H_ */
