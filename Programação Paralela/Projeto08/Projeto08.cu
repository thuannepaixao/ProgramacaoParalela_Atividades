/**********************************/
// Pavic  - CUDA -  Add Block & Threads

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<random>
#include <stdio.h>


//populate vectors with random ints
void random_ints(int* a, int N) {
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 1000;
	}
}

//  cuda = ADD() BLOCK
__global__ void add_blocks(int* a, int* b, int* c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

//  cuda = ADD() THREADS
__global__ void add_threads(int* a, int* b, int* c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

#define N 5120000             // Parallel Problem !!

int main(void) {
	int* a, * b, * c;	// host copies of a, b, c
	int* d_a, * d_b, * d_c;	// device copies of a, b, c
	int size = N * sizeof(int); // More Memory!!

	// Alloc space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	// Alloc space for host copies of a, b, c and setup input values

	a = (int*)malloc(size);

	random_ints(a, N);
	b = (int*)malloc(size);
	random_ints(b, N);

	c = (int*)malloc(size);
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N blocks
	//add_blocks <<<N, 1 >> > (d_a, d_b, d_c); // Parallel - Blocks
	add_threads <<< N, 1 >>> (d_a, d_b, d_c); // PArallel - Threads
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	printf(" Size of C= %d", sizeof(c));

	/*
	for (int i = 0; i < N; i++) {
		printf(c[i]);
		sizeof(c);
	}
	*/

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}