#include <stdio.h>

__global__ void HelloFromGPU(){
	printf("Initial commit.\n");
}

int main (int argc, char **argv){

	HelloFromGPU<<<1, 10>>>();
	cudaDeviceSynchronize();
	return 0;
}
