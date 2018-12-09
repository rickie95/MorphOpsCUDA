#include "MorphableOperator.h"


__host__ Image_t* opening(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){ // EROSION then DILATATION
    std::chrono::duration_cast<std::chrono::duration<double>> erosion_time, dilatation_time;
    Image_t* opened = dilatation(erosion(input, structElem, erosion_time), structElem, dilatation_time);
    if(time_span != NULL)
        *time_span = std::chrono::duration_cast<std::chrono::duration<double>>(erosion_time + dilatation_time);
    return opened;
}

__host__ Image_t* closing(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
    std::chrono::duration_cast<std::chrono::duration<double>> erosion_time, dilatation_time;
    Image_t* closed = erosion(dilatation(input, structElem, dilatation_time), structElem, erosion_time);
    if(time_span != NULL)
    	*time_span = std::chrono::duration_cast<std::chrono::duration<double>>(erosion_time + dilatation_time);
    return closed;
}

__host__ Image_t* topHat(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	// Originale - Apertura
	Image_t* opened = opening(input, structElem, time_span);
	float *topHat_data = (float*)malloc(input->width * input->height * sizeof(float));
	for(int i = 0; i < input->width * input->height; i += 1) // Maybe should transposed on GPU?
		topHat_data[i] = max(input->data[i] - opened->data[i], 0);

	Image_delete(opened);
	return Image_new(input->width, input->height, 1, topHat_data);
}

__host__ Image_t* bottomHat(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	// Chiusura - originale
	Image_t* closed = closing(input, structElem, time_span);
	float *bottomHat_data = (float*)malloc(input->width * input->height * sizeof(float));
	for(int i = 0; i < input->width * input->height; i += 1) // Maybe should transposed on GPU?
	    bottomHat_data[i] = max(closed->data[i] - input->data[i], 0);

	Image_delete(closed);
	return Image_new(input->width, input->height, 1, topHat_data);
}

__host__ Image_t* erosion(Image_t* input, StructElem* structElem, std::chrono::duration<double> *time_span){
	// malloc for I/O images and SE
	float *deviceInputImage, *deviceOutputImage, *deviceSEData, *hostOutputImage=NULL;

	std::chrono::high_resolution_clock::time_point t_start, t_end;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceInputImage, sizeof(float) * input->height * input->width));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceOutputImage, sizeof(float) * input->height * input->width));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceSEData, sizeof(float) * structElem->get_width() * structElem->get_height()));
	// Send data (Input and SE)
	CUDA_CHECK_RETURN(cudaMemcpy(deviceInputImage, input->data, input->height * input->width * sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(deviceSEData, structElem->data, structElem->get_width() * structElem->get_height() * sizeof(float),cudaMemcpyHostToDevice));

	// COMPUTE
	t_start = std::chrono::high_resolution_clock::now();
	process<<<1, 1>>>(deviceInputImage, deviceOutputImage, input->width, input->height, deviceSEData,
		structElem->get_width(), structElem->get_height(), EROSION);
	cudaDeviceSynchronize(); // wait for completion
	t_end = std::chrono::high_resolution_clock::now();
	if(time_span != NULL)
	    *time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);

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

    std::chrono::high_resolution_clock::time_point t_start, t_end;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceInputImage, sizeof(float) * input->height * input->width));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceOutputImage, sizeof(float) * input->height * input->width));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&deviceSEData, sizeof(float) * structElem->get_width() * structElem->get_height()));
    // Send data (Input and SE)
    CUDA_CHECK_RETURN(cudaMemcpy(deviceInputImage, input->data, input->height * input->width * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceSEData, structElem->data, structElem->get_width() * structElem->get_height() * sizeof(float),cudaMemcpyHostToDevice));

    // COMPUTE
    t_start = std::chrono::high_resolution_clock::now();
    process<<<1, 1>>>(deviceInputImage, deviceOutputImage, input->width, input->height, deviceSEData,
    structElem->get_width(), structElem->get_height(), DILATATION);
    cudaDeviceSynchronize(); // wait for completion
    t_end = std::chrono::high_resolution_clock::now();
    *time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);

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
		const float *__restrict__ SE, int SE_W, int SE_H, int operation){

    // Assumes that input_img->data is a NxM float matrix with values 0 or 1
    // OPERATION: 1 if erosion, 0 if dilatation

    __shared__ float input_ds[BLOCK_DIM * BLOCK_DIM];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int tx_global = (BLOCK_DIM * bx + tx + (int)(SE_W/2)); // COLONNA
    int ty_global = (BLOCK_DIM * by + ty + (int)(SE_H/2)); // RIGA

    // LOAD DS
    // FOR EVERY PIXEL OF BLOCK COMPUTE MIN OR MAX
    // WRITE IN OUTPUT

    // 1) LOADING - carica il pixel corrispondente se non esce dai confini, altrimenti metti un valore dummy

    if( (bx * img_H + tx < img_H) && (by * img_W + ty < img_W) ){
    	input_ds[ty * BLOCK_DIM + tx] = input_img[ty_global * img_W + tx_global];
    }else{
    	input_ds[ty * BLOCK_DIM + tx] = -1.0;
    }

    __syncthreads();

    // 2) COMPUTE - carica il neighbrhood e ci fa il max/min a seconda dell'operazione, poi scrive

    if( (bx * img_H + tx < img_H) && (by * img_W + ty < img_W) &&// deve essere un pixel dell'immagine
    		(tx > SE_W/2 && tx < BLOCK_DIM + SE_W/2 && ty > SE_H/2 && ty < BLOCK_DIM + SE_H/2) ){ // e deve essere un pixel del tile interno, non del padding
        float *neighborhood = new float[SE_W * SE_H];
    	int k = 0;
    	for( int row = 0 ; row < SE_H; row+=1){
    		for(int col = 0; col < SE_W; col+=1){
    			if(input_ds[(ty - SE_H/2 + row) * BLOCK_DIM + (tx - SE_W/2 + col)] > -1 &&  // pixel valido
    					deviceSEdata[row * SE_W + col] > 0){ // è coperto dal SE?
    				neighborhood[k] = input_ds[(ty - SE_H/2 + row) * BLOCK_DIM + (tx - SE_W/2 + col)];
    				k =+ 1;
    			}
    		}
    	}
    	// Write value
    	if(operation == EROSION)
    	    output_img[ty_global * img_W + tx_global] = max(neighborhood, k);
    	if(operation == DILATATION)
    		output_img[ty_global * img_W + tx_global] = min(neighborhood, k);
    }

    __syncthreads();

}

// TODO: check out max and min
__host__ float max(float a, float b){
	return a>b ? a : b;
}

__host__ float min(float a, float b){
	return a<b ? a : b;
}

__device__ float max(float* array, int length){
	float max = array[0];
	for(int i = 0 ; i < length; i += 1)
		max < array[i] ? max = array[i] : true;

	return max;
}
__device__ float min(float* array, int length){
	float min = array[0];
	for(int i = 0 ; i < length; i += 1)
			min > array[i] ? min = array[i] : true;

	return min;
}
