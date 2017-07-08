#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "cuutils.h"
#include "absnf.h"
#include "utils.hpp"
#include <chrono>
#define t_def double

typedef std::chrono::high_resolution_clock::time_point TimeVar;
void single_execution(int s)
{
	t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_b = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Z = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_J = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_Y = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));

	t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
	t_def *d_b; cudaMalloc((void **)&d_b, s*sizeof(t_def));
	t_def *d_Z; cudaMalloc((void **)&d_Z, s*s*sizeof(t_def));
	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_J; cudaMalloc((void **)&d_J, s*s*sizeof(t_def));
	t_def *d_Y; cudaMalloc((void **)&d_Y, s*s*sizeof(t_def));		
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	t_def *d_gamma; cudaMalloc((void **)&d_gamma, s*sizeof(t_def));
	t_def *d_Gamma; cudaMalloc((void **)&d_Gamma, s*s*sizeof(t_def));
	t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));
	t_def *d_I; cudaMalloc((void **)&d_I, s*s*sizeof(t_def));
	t_def *d_K; cudaMalloc((void **)&d_K, s*s*sizeof(t_def));

	utils::fillRandVector(h_a, s,-10,10);
	utils::fillRandVector(h_b, s,-10,10);
	utils::fillRandVector(h_Z, s*s,-10,10);
	utils::fillRandMatrix(h_L, s,s,-10,10,0,utils::MATRIXOPT::LOWER);
	utils::fillRandVector(h_J, s*s,-10,10);
	utils::fillRandVector(h_Y, s*s,-10,10);
	utils::fillRandVector(h_dz, s,-10,10);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	TimeVar t_0 = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z, h_Z,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_J, h_J,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	TimeVar t_1 = std::chrono::high_resolution_clock::now();
	auto time_copy = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	int gridsize, blocksize;
	cuutils::getGridBlockSize(&gridsize, &blocksize);

	t_0 = std::chrono::high_resolution_clock::now();
	absnf::gradient_core(cublas_handle,
							 d_a, d_b, d_Z, d_L, d_J,
							 d_Y, d_dz, d_Tss, d_I, d_K,
							 s, s, s,
							 gridsize, blocksize,
							 d_gamma, d_Gamma);
	
	cudaDeviceSynchronize();
	t_1 = std::chrono::high_resolution_clock::now();
	auto time_exec = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	
	// TimeVar t_3 = std::chrono::high_resolution_clock::now();
	
	// auto int_upload = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	// auto int_download = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_2 ).count();
	// auto int_total = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_0 ).count();
	
	std::cout << s << ", " << time_copy << ", " << time_exec << std::endl;
	// std::cout << "---------------" << std::endl;
	// std::cout << "s=m=n: " << s  << std::endl;
	// std::cout << "mem:   " << time_copy << std::endl;
	// std::cout << "exec:  " << time_exec << std::endl;
	free(h_a);
	free(h_b);
	free(h_Z);
	free(h_L);
	free(h_J);
	free(h_Y);
	free(h_dz);

	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_Z);
	cudaFree(d_L);
	cudaFree(d_J);
	cudaFree(d_Y);
	cudaFree(d_dz);
	cudaFree(d_Tss);
	cudaFree(d_I);
	cudaFree(d_K);
	cudaFree(d_gamma);
	cudaFree(d_Gamma);

	cublasDestroy(cublas_handle);
}
void multiple_execution(int s, int executions)
{
	t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_b = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Z = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_J = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_Y = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));

	t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
	t_def *d_b; cudaMalloc((void **)&d_b, s*sizeof(t_def));
	t_def *d_Z; cudaMalloc((void **)&d_Z, s*s*sizeof(t_def));
	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_J; cudaMalloc((void **)&d_J, s*s*sizeof(t_def));
	t_def *d_Y; cudaMalloc((void **)&d_Y, s*s*sizeof(t_def));		
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	t_def *d_gamma; cudaMalloc((void **)&d_gamma, s*sizeof(t_def));
	t_def *d_Gamma; cudaMalloc((void **)&d_Gamma, s*s*sizeof(t_def));
	t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));
	t_def *d_I; cudaMalloc((void **)&d_I, s*s*sizeof(t_def));
	t_def *d_K; cudaMalloc((void **)&d_K, s*s*sizeof(t_def));

	utils::fillRandVector(h_a, s,-10,10);
	utils::fillRandVector(h_b, s,-10,10);
	utils::fillRandVector(h_Z, s*s,-10,10);
	utils::fillRandMatrix(h_L, s,s,-10,10,0,utils::MATRIXOPT::LOWER);
	utils::fillRandVector(h_J, s*s,-10,10);
	utils::fillRandVector(h_Y, s*s,-10,10);
	utils::fillRandVector(h_dz, s,-10,10);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);


	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z, h_Z,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_J, h_J,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	int gridsize, blocksize;
	cuutils::getGridBlockSize(&gridsize, &blocksize);

	TimeVar t_0 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<executions; i++)
	{
		absnf::gradient_core(cublas_handle,
							 d_a, d_b, d_Z, d_L, d_J,
							 d_Y, d_dz, d_Tss, d_I, d_K,
							 s, s, s,
							 gridsize, blocksize,
							 d_gamma, d_Gamma);
	}
	cudaDeviceSynchronize();
	TimeVar t_1 = std::chrono::high_resolution_clock::now();
	auto time_exec = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	
	// TimeVar t_3 = std::chrono::high_resolution_clock::now();
	
	// auto int_upload = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	// auto int_download = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_2 ).count();
	// auto int_total = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_0 ).count();
	
	std::cout << s  << ", " << time_exec << std::endl;
	free(h_a);
	free(h_b);
	free(h_Z);
	free(h_L);
	free(h_J);
	free(h_Y);
	free(h_dz);

	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_Z);
	cudaFree(d_L);
	cudaFree(d_J);
	cudaFree(d_Y);
	cudaFree(d_dz);
	cudaFree(d_Tss);
	cudaFree(d_I);
	cudaFree(d_K);
	cudaFree(d_gamma);
	cudaFree(d_Gamma);

	cublasDestroy(cublas_handle);
}
void devInfo()
{
	cudaDeviceProp prop;
	int devcount;
	cudaGetDeviceCount(&devcount);
	std::cout << "Devices found: " << devcount << std::endl;
	for(int i=0; i<devcount; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		std::cout << "------------------" << std::endl;
		std::cout << "Device: " << i << std::endl;
		std::cout << "------------------" << std::endl;
		std::cout << "Name:\t\t\t" << prop.name << std::endl;
		std::cout << "GlobalMemory:\t\t" << prop.totalGlobalMem << std::endl;
		std::cout << "WarpSize:\t\t" << prop.warpSize << std::endl;
		std::cout << "MaxThreadsPerBlock:\t" << prop.maxThreadsPerBlock << std::endl;
		std::cout << "MaxThreadsDim:\t\t" << prop.maxThreadsDim[0] << " : " << prop.maxThreadsDim[1] << " : " << prop.maxThreadsDim[2] << std::endl;
		std::cout << "MaxGridSize:\t\t" << prop.maxGridSize[0] << " : " << prop.maxGridSize[1] << " : " << prop.maxGridSize[2] << std::endl;
		std::cout << "MultiProcessorCount:\t" << prop.multiProcessorCount << std::endl;
	}	
}
void single_execution_series()
{

	devInfo();
	std::cout << "------------SINGLE EXECUTION -------------" << std::endl;
	std::cout << "s, time_copy, time_exec" << std::endl;
	for(int s=1000; s <= 5000; s+=1000)
	{
		single_execution(s);
	}
}
void multiple_execution_series()
{
	int executions = 1000;
	devInfo();
	std::cout << "------------MULTIPLE EXECUTIONS -------------" << std::endl;
	std::cout << "s, time_exec" << std::endl;
	for(int s=1000; s <= 5000; s+=1000)
	{
		multiple_execution(s, executions);
	}
}
int main()
{
	single_execution_series();
	multiple_execution_series();
	return 0;
}