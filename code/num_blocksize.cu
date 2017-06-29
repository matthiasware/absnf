#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "cuutils.h"
#include "absnf.h"
#include "utils.hpp"
#include <chrono>
#include <fstream>
#define t_def double

typedef std::chrono::high_resolution_clock::time_point TimeVar;

void getGridBlockSize(int *gridsize, int *blocksize)
{
		cudaDeviceProp prop;
		int devcount;
		cudaGetDeviceCount(&devcount);
		// Take first device, 
		// TODO: room for improvements
		cudaGetDeviceProperties(&prop, 0);
		// we decided to run 8 blocks / MPU
		// TODO: room for improvements
		*gridsize = prop.multiProcessorCount * 8;
		*blocksize = prop.maxThreadsPerBlock;
};

void gridsize()
{
	cudaDeviceProp prop;
	int devcount;
	cudaGetDeviceCount(&devcount);
	cudaGetDeviceProperties(&prop, 0);

	// int gridsize = prop.multiProcessorCount;
	// int blocksize = prop.maxThreadsPerBlock;

	int s = 16000;
	
	// t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_A = (t_def *)malloc(s*s*sizeof(t_def));

	// utils::fillRandMatrix(h_L, s,s,-10,10,0,utils::MATRIXOPT::LOWER);
	utils::fillRandVector(h_dz, s,-10,10);
	utils::fillRandVector(h_A, s*s,-10,10);

	// t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	// t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));
	t_def *d_A; cudaMalloc((void **)&d_A, s*s*sizeof(t_def));

	TimeVar t_1;
	TimeVar t_0;	
	
	t_0 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_A, h_A,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);	
	t_1 = std::chrono::high_resolution_clock::now();
	auto time_mem = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
	std::cout << "Memory upload: " << time_mem << std::endl;

	// std::cout << prop.maxThreadsPerBlock << std::endl;
	int gs = prop.multiProcessorCount;
	for(int i=1; i<=100; i+=1)
	{
		int bs = prop.maxThreadsPerBlock * i / 100;
		t_0 = std::chrono::high_resolution_clock::now();
		absnf::multWithDz<<<gs, bs>>>(d_A, d_dz, s);
		cudaDeviceSynchronize();
		t_1 = std::chrono::high_resolution_clock::now();
		auto time_exec = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
		std::cout << "[" << bs << "," << time_exec << "]," << std::endl;
	}

	free(h_A);
	free(h_dz);
	cudaFree(d_A);
	cudaFree(d_dz);
}
void blocksize()
{
	cudaDeviceProp prop;
	int devcount;
	cudaGetDeviceCount(&devcount);
	cudaGetDeviceProperties(&prop, 0);

	int s = 16000;
	
	// t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_A = (t_def *)malloc(s*s*sizeof(t_def));

	utils::fillRandVector(h_dz, s,-10,10);
	utils::fillRandVector(h_A, s*s,-10,10);

	// t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	// t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));
	t_def *d_A; cudaMalloc((void **)&d_A, s*s*sizeof(t_def));

	TimeVar t_1;
	TimeVar t_0;	
	
	t_0 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_A, h_A,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);	
	t_1 = std::chrono::high_resolution_clock::now();
	auto time_mem = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
	std::cout << "Memory upload: " << time_mem << std::endl;

	// std::cout << prop.maxThreadsPerBlock << std::endl;
	int gs;
	int bs = prop.maxThreadsPerBlock
	for(int i=1; i<=1000; i+=1)
	{	
		gs = i;
		t_0 = std::chrono::high_resolution_clock::now();
		absnf::multWithDz<<<gs, bs>>>(d_A, d_dz, s);
		cudaDeviceSynchronize();
		t_1 = std::chrono::high_resolution_clock::now();
		auto time_exec = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
		std::cout << "[" << gs << "," << time_exec << "]," << std::endl;
	}

	free(h_A);
	free(h_dz);
	cudaFree(d_A);
	cudaFree(d_dz);
}
void block_grid()
{
	std::ofstream myfile;
  	myfile.open ("data.csv");

	cudaDeviceProp prop;
	int devcount;
	cudaGetDeviceCount(&devcount);
	cudaGetDeviceProperties(&prop, 0);

	// int gridsize = prop.multiProcessorCount;
	// int blocksize = prop.maxThreadsPerBlock;

	int s = 16000; // 2GB
	
	// t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_A = (t_def *)malloc(s*s*sizeof(t_def));

	// utils::fillRandMatrix(h_L, s,s,-10,10,0,utils::MATRIXOPT::LOWER);
	utils::fillRandVector(h_dz, s,-10,10);
	utils::fillRandVector(h_A, s*s,-10,10);

	// t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	// t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));
	t_def *d_A; cudaMalloc((void **)&d_A, s*s*sizeof(t_def));

	TimeVar t_1;
	TimeVar t_0;	
	
	t_0 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_A, h_A,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);	
	t_1 = std::chrono::high_resolution_clock::now();
	auto time_mem = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
	std::cout << "Memory upload: " << time_mem << std::endl;

	// std::cout << prop.maxThreadsPerBlock << std::endl;
	int blocksize;
	int gridsize;
	for (int j=1; j<=100; j++)
	{
		blocksize = (int) prop.maxThreadsPerBlock * j / 100;
		for(int i=1; i<=100; i+=1)
		{
			gridsize = i;
			t_0 = std::chrono::high_resolution_clock::now();
			absnf::multWithDz<<<gridsize, blocksize>>>(d_A, d_dz, s);
			cudaDeviceSynchronize();
			t_1 = std::chrono::high_resolution_clock::now();
			auto time_exec = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
			myfile << blocksize << "," <<gridsize << "," << time_exec << "\n";
		}
	}

	free(h_A);
	free(h_dz);
	cudaFree(d_A);
	cudaFree(d_dz);
	// cudaFree(d_Tss);
	myfile.close();
}

// 1 - 1000
// 0.1 - 1
int main()
{
	blocksize();
	return 0;
}