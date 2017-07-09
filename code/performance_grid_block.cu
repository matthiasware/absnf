#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include "cuutils.h"
#include "absnf.h"
#include "utils.hpp"
#include <chrono>
#define t_def double

typedef std::chrono::high_resolution_clock::time_point TimeVar;

template <typename T>
void __global__ calcColwise(T *matrix, int s)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int id = i*s + j;
	int size = s*s;
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	while(id < size && j < s)
	{
		matrix[id] = std::sin((double) global_id);
		// matrix[id] = global_id;
		i +=  blockDim.x;
		if(i >= s)
		{
			i = i % s;
			j = j + gridDim.x;
		}
		id = i*s + j;
	}
}
template <typename T>
void __global__ calcRowwise(T *matrix, int s)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int id = i*s + j;
	int size = s*s;
	while(id < size && i < s)
	{
		matrix[id] = std::sin((double) global_id);
		// matrix[id] = global_id;
		j += blockDim.x;
		if (j>=s)
		{
			j = j % s;
			i = i + gridDim.x;
		}
		id = i*s+j;
	}
}

void test_rowwise(int s, int gridsize, int blocksize, int times, bool printf=false)
{
	t_def *h_matrix = (t_def *)malloc(s*s*sizeof(t_def));
	utils::fillVector(h_matrix, s*s, 0.0);
	t_def *d_matrix; cudaMalloc((void **)&d_matrix, s*s*sizeof(t_def));
	cudaMemcpy(d_matrix, h_matrix,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);

	TimeVar t_0 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i<times; i++)
		calcRowwise <<<gridsize, blocksize>>> (d_matrix, s);
	cudaDeviceSynchronize();
	TimeVar t_1 = std::chrono::high_resolution_clock::now();
	auto int_time = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	
	cudaMemcpy(h_matrix, d_matrix, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	if(printf)
	{
		utils::printf_matrix(h_matrix, s, s);
	}
	std::cout << "rowwise: " << int_time << std::endl;
}
void test_colwise(int s, int gridsize, int blocksize, int times, bool printf=false)
{
	t_def *h_matrix = (t_def *)malloc(s*s*sizeof(t_def));
	utils::fillVector(h_matrix, s*s, 0.0);
	t_def *d_matrix; cudaMalloc((void **)&d_matrix, s*s*sizeof(t_def));
	cudaMemcpy(d_matrix, h_matrix,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);

	TimeVar t_0 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i<times; i++)
		calcColwise <<<gridsize, blocksize>>> (d_matrix, s);
	cudaDeviceSynchronize();
	TimeVar t_1 = std::chrono::high_resolution_clock::now();
	auto int_time = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	
	cudaMemcpy(h_matrix, d_matrix, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	if(printf)
	{
		utils::printf_matrix(h_matrix, s, s);
	}
	std::cout << "colwise: " << int_time << std::endl;
}
cudaDeviceProp devInfo()
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
	cudaGetDeviceProperties(&prop, 0);	
	return prop;
}

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int mpu = prop.multiProcessorCount;
	// int warpsize = prop.warpSize;
	int maxThreadsPerBlock =  prop.maxThreadsDim[0];

	int s = 4000;
	int gridsize = mpu;
	int blocksize = maxThreadsPerBlock;
	int times = 100;
	bool printf = false;
	test_rowwise(s, gridsize, blocksize, times, printf);
	test_colwise(s, gridsize, blocksize, times, printf);
	return 0;
}