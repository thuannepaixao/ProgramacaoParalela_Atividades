#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Device code  - GPU 
__global__ void HelloGPU(void) {
	printf("  Hello CUDA GPU\n");
}
// Device ADD two int , a, b
__global__ void add(int* a, int* b, int* c) {

	printf("GPU ADD a=%d + b= %d \n", *a, *b);
	*c = *a + *b;
	printf("  ADD at GPU : DONE Result= %d\n", *c);
}
int main() {


	int h_a, h_b, h_c;	        // host copies of a, b, c
	int* d_a, * d_b, * d_c;	    // device copies of a, b, c

	int size = sizeof(int);
	// Allocate space for device (GPU) copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	// Setup input values
	h_a = 2;
	h_b = 7;

	// Copy inputs to device
	// cudaMemcpy(Destination, Source, size, Directions);

	cudaMemcpy(d_a, &h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add << <1, 1 >> > (d_a, d_b, d_c);

	// Copy result back to host
	cudaMemcpy(&h_c, d_c, size, cudaMemcpyDeviceToHost);
	
	//Check result C
	printf(" Output = %d\n", h_c);

	// Cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	printf("  Hello CPU 01 \n");


	HelloGPU << < 1, 1 >> > (); // Call GPU


	printf("  Hello CPU 02 \n");
	return 0;
}