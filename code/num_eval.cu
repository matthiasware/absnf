#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "cuutils.h"
#include "absnf.h"
#include "utils.hpp"
#include <chrono>
#define t_def double

typedef std::chrono::high_resolution_clock::time_point TimeVar;
void test(int s)
{
	t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_b = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Z = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_J = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_Y = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dx = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_dy = (t_def *)malloc(s*sizeof(t_def));

	t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
	t_def *d_b; cudaMalloc((void **)&d_b, s*sizeof(t_def));
	t_def *d_Z; cudaMalloc((void **)&d_Z, s*s*sizeof(t_def));
	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_J; cudaMalloc((void **)&d_J, s*s*sizeof(t_def));
	t_def *d_Y; cudaMalloc((void **)&d_Y, s*s*sizeof(t_def));		
	t_def *d_dx; cudaMalloc((void **)&d_dx, s*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	t_def *d_abs_dz; cudaMalloc((void **)&d_abs_dz, s*sizeof(t_def));
	t_def *d_dy; cudaMalloc((void **)&d_dy, s*sizeof(t_def));

	utils::fillRandVector(h_a, s,-10,10);
	utils::fillRandVector(h_b, s,-10,10);
	utils::fillRandVector(h_Z, s*s,-10,10);
	utils::fillRandVector(h_J, s*s,-10,10);
	utils::fillRandVector(h_Y, s*s,-10,10);
	utils::fillRandVector(h_dx, s,-10,10);
	utils::fillRandMatrix(h_L, s,s,-10,10,0,utils::MATRIXOPT::LOWER);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	// TimeVar t_0 = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z, h_Z,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_J, h_J,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, h_dx, s*sizeof(t_def), cudaMemcpyHostToDevice);

	int k = 1000;
	TimeVar t_0 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<k; i++)
	{
		absnf::eval_core(cublas_handle, d_a, d_b,
			 		  	 d_Z, d_L,
			 		  	 d_J, d_Y,
			 		  	 d_dx,
			 		  	 s, s, s,
			 		  	 d_dz, d_dy,
			 		  	 d_abs_dz);
	}
	cudaDeviceSynchronize();
	TimeVar t_1 = std::chrono::high_resolution_clock::now();
	
	cudaMemcpy(h_dz, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dy, d_dy, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	// TimeVar t_3 = std::chrono::high_resolution_clock::now();
	
	// auto int_upload = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	auto int_exec = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	// auto int_download = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_2 ).count();
	// auto int_total = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_0 ).count();
	
	std::cout << "----" << s  << " : " << k << "----" << std::endl;
	std::cout <<"exec:  " << int_exec << std::endl;
	free(h_a);
	free(h_b);
	free(h_Z);
	free(h_L);
	free(h_J);
	free(h_Y);
	free(h_dx);

	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_Z);
	cudaFree(d_L);
	cudaFree(d_J);
	cudaFree(d_Y);
	cudaFree(d_dx);
	cudaFree(d_dz);
	cudaFree(d_abs_dz);
	cudaFree(d_dy);

	cublasDestroy(cublas_handle);	
}

int main()
{
	for(int i=1000; i<=10000; i+=1000)
	{
		test(i);
	}

	return 0;
}