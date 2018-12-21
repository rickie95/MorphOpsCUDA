#include "MorphableOperator.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err){
	if (err == cudaSuccess)
			return;
		std::cerr << statement <<
				" returned " <<
				cudaGetErrorString(err) << "("
				<< err << ") at " << file << ":" << line << std::endl;
		exit(1);
}

__host__ Image_t* opening(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){ // EROSION then DILATATION
	std::chrono::duration<double> erosion_time, dilatation_time;
    Image_t* opened = dilatation(erosion(input, structElem, &erosion_time), structElem, &dilatation_time);

    if(time_span != NULL)
        *time_span = std::chrono::duration_cast<std::chrono::duration<double>>(erosion_time + dilatation_time);

    return opened;
}

__host__ Image_t* closing(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	std::chrono::duration<double> erosion_time, dilatation_time;
    Image_t* closed = erosion(dilatation(input, structElem, &dilatation_time), structElem, &erosion_time);

    if(time_span != NULL)
    	*time_span = std::chrono::duration_cast<std::chrono::duration<double>>(erosion_time + dilatation_time);

    return closed;
}

__host__ Image_t* topHat(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	// Originale - Apertura
	Image_t* opened = opening(input, structElem, time_span);
	float *topHat_data = (float*)malloc(input->width * input->height * sizeof(float));

	for(int i = 0; i < input->width * input->height; i += 1) // Maybe should transposed on GPU?
		topHat_data[i] = max_pixel(opened->data[i] - input->data[i], 0);

	Image_delete(opened);
	return Image_new(input->width, input->height, 1, topHat_data);
}

__host__ Image_t* bottomHat(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	// Chiusura - originale
	Image_t* closed = closing(input, structElem, time_span);
	float *bottomHat_data = (float*)malloc(input->width * input->height * sizeof(float));

	for(int i = 0; i < input->width * input->height; i += 1) // Maybe should transposed on GPU?
	    bottomHat_data[i] = max_pixel(input->data[i] + closed->data[i], 0);

	Image_delete(closed);
	return Image_new(input->width, input->height, 1, bottomHat_data);
}

__host__ Image_t* erosion(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	// malloc for I/O images and SE
	float *deviceInputImage, *deviceOutputImage, *deviceSEData, *hostOutputImage=NULL;
    dim3 dimGrid(ceil((float) input->width / TILE_WIDTH), ceil((float) input->height / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	std::chrono::high_resolution_clock::time_point t_start, t_end;

	// Alloc memory on device
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceInputImage, sizeof(float) * input->height * input->width));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceOutputImage, sizeof(float) * input->height * input->width));

	// Send data (Input and SE)
	CUDA_CHECK_RETURN(cudaMemcpy(deviceInputImage, input->data, input->height * input->width * sizeof(float),
	        cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(deviceSEdata, structElem->data,
            structElem->get_width() * structElem->get_height() * sizeof(float)));

    // COMPUTE
	t_start = std::chrono::high_resolution_clock::now();
	process<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, input->width, input->height, deviceSEData,
		structElem->get_width(), structElem->get_height(), EROSION);
	cudaDeviceSynchronize(); // wait for completion
	t_end = std::chrono::high_resolution_clock::now();

	if(time_span != NULL)
	    *time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);

    hostOutputImage = (float*)malloc(input->height * input->width * sizeof(float));
	// download data
	CUDA_CHECK_RETURN(cudaMemcpy(hostOutputImage, deviceOutputImage, input->height * input->width * sizeof(float),cudaMemcpyDeviceToHost));
    Image_t *output = Image_new(input->width, input->height, 1, hostOutputImage);
	// free memory on GPU
	cudaFree(deviceInputImage);
	cudaFree(deviceOutputImage);
	cudaFree(deviceSEData);
    return output;
}

__host__ Image_t* dilatation(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
// malloc for I/O images and SE
    float *deviceInputImage, *deviceOutputImage, *deviceSEData, *hostOutputImage=NULL;
    dim3 dimGrid(ceil((float) input->width / TILE_WIDTH), ceil((float) input->height / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    std::chrono::high_resolution_clock::time_point t_start, t_end;

    // Alloc memory on device
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceInputImage, sizeof(float) * input->height * input->width));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceOutputImage, sizeof(float) * input->height * input->width));

    // Send data (Input and SE)
    CUDA_CHECK_RETURN(cudaMemcpy(deviceInputImage, input->data, input->height * input->width * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(deviceSEdata, structElem->data, structElem->get_width() * structElem->get_height() * sizeof(float)));

    // COMPUTE
    t_start = std::chrono::high_resolution_clock::now();
    process<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, input->width, input->height, deviceSEData,
    structElem->get_width(), structElem->get_height(), DILATATION);
    cudaDeviceSynchronize(); // wait for completion
    t_end = std::chrono::high_resolution_clock::now();

    if(time_span != NULL)
        *time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);

    hostOutputImage = (float*)malloc(input->height * input->width * sizeof(float));
    // download data
    CUDA_CHECK_RETURN(cudaMemcpy(hostOutputImage, deviceOutputImage, input->height * input->width * sizeof(float),cudaMemcpyDeviceToHost));
    Image_t *output = Image_new(input->width, input->height, 1, hostOutputImage);
    // free memory
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    cudaFree(deviceSEData);

    return output;
}

__global__ void process(float *input_img, float *output_img, int img_W, int img_H,
		const float *__restrict__ SE,const int SE_W, const int SE_H, int operation) {

	// Assumes that input_img->data is a NxM float matrix with values 0 or 1
	// OPERATION: 1 if erosion, 0 if dilatation

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int w = TILE_WIDTH + SE_W - 1;
	int globalX = blockIdx.x * TILE_WIDTH + tx;
	int globalY = blockIdx.y * TILE_WIDTH + ty;
	int globalCoord = globalY * img_W + globalX;

    // 2) COMPUTE - load neighborhood and write max/min
    if((globalY >= 0 && globalY < img_H && globalX >= 0 && globalX < img_W)){

        float max = input_img[globalCoord];
        float min = max;

    	for(int row = - SE_H/2 ; row < SE_H/2; row+=1){
    		for(int col = -SE_W/2; col < SE_W/2; col+=1){
    			if(deviceSEdata[(row + SE_H/2) * SE_W + col + SE_W/2] > 0 &&
    			        input_img[(globalY + row) * img_W + globalX + col] > -1){

    				if (max < input_img[(globalY + row) * img_W + globalX + col])
    					max = input_img[(globalY + row) * img_W + globalX + col];

    				if (min > input_img[(globalY + row) * img_W + globalX + col])
    					min = input_img[(globalY + row) * img_W + globalX + col];
    			}
    		}
    	}
    	// Write value

    	if(operation == EROSION)
    	    output_img[globalCoord] = max;
    	if(operation == DILATATION)
    		output_img[globalCoord] = min;

    }
    __syncthreads();

}

__host__ float max_pixel(float a, float b){
	return a>b ? a : b;
}

__host__ float min_pixel(float a, float b){
	return a<b ? a : b;
}


