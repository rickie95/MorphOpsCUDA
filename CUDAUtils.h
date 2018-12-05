/*
 * CUDAUtils.h
 *
 *  Created on: Dec 5, 2018
 *      Author: riccardo_malavolti
 */

#ifndef CUDAUTILS_H_
#define CUDAUTILS_H_

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err);

#endif /* CUDAUTILS_H_ */
