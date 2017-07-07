#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "cuutils.h"
#include "absnf.h"
#include "utils.hpp"
#include <chrono>
#include <typeinfo>
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

	TimeVar t_0 = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z, h_Z,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_J, h_J,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, h_dx, s*sizeof(t_def), cudaMemcpyHostToDevice);

	TimeVar t_1 = std::chrono::high_resolution_clock::now();
	absnf::eval_core(cublas_handle, d_a, d_b,
			 		  	 d_Z, d_L,
			 		  	 d_J, d_Y,
			 		  	 d_dx,
			 		  	 s, s, s,
			 		  	 d_dz, d_dy,
			 		  	 d_abs_dz);
	cudaDeviceSynchronize();
	TimeVar t_2 = std::chrono::high_resolution_clock::now();
	
	cudaMemcpy(h_dz, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dy, d_dy, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	TimeVar t_3 = std::chrono::high_resolution_clock::now();
	
	auto int_upload = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
	auto int_exec = std::chrono::duration_cast<std::chrono::milliseconds>( t_2 - t_1 ).count();
	auto int_download = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_2 ).count();
	auto int_total = std::chrono::duration_cast<std::chrono::milliseconds>( t_3 - t_0 ).count();
	
	std::cout << "----" << s << "----" << std::endl;
	std::cout <<"upload:  " << int_upload << std::endl;
	std::cout <<"exec:  " << int_exec << std::endl;
	std::cout <<"download:  " << int_download << std::endl;
	std::cout <<"total:  " << int_total << std::endl;
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

void multiple_executions(int s, int executions)
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

	TimeVar t_0 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<executions; i++)
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
	
	std::cout << "----" << s  << " : " << executions << "----" << std::endl;
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
long int getGlobalMemory()
{
	long int globalMemory = 0;
	cudaDeviceProp prop;
	int devcount;
	cudaGetDeviceCount(&devcount);
	if (devcount > 0)
	{
		cudaGetDeviceProperties(&prop, 0);
		globalMemory = prop.totalGlobalMem;

	}
	return globalMemory;
}
long int calcRequiredMemory(int s)
{
	return (4*s*s + 6*s) * sizeof(t_def);
}

void single_execution_series()
{
	devInfo();
	long int globalMemory = getGlobalMemory();
	std::cout << globalMemory << std::endl;
	// SINGLE EXECUTIONS
	int size = 1000;
	int maxsize = 20000;
	while(true)
	{
		long int requiredMemory = calcRequiredMemory(size);
		if(requiredMemory > (long int) (globalMemory * 0.9) && size < maxsize)
		{
			break;
		}
		else
		{
			single_execution(size);
			std::cout << "Required Memory: " << requiredMemory * 1e-9 << std::endl;
			size+=1000;	
		}
	}
}
void multiple_executions_series(int times)
{
	devInfo();
	long int globalMemory = getGlobalMemory();
	std::cout << globalMemory << std::endl;
	int size = 1000;
	int maxsize = 20000;
	while(true)
	{
		long int requiredMemory = calcRequiredMemory(size);
		if(requiredMemory > (long int) (globalMemory * 0.9) && size < maxsize)
		{
			break;
		}
		else
		{
			multiple_executions(size, times);
			std::cout << "Required Memory: " << requiredMemory * 1e-9 << std::endl;
			size+=1000;	
		}
	}
}
int main()
{
	std::cout << "------------------------------------------------" << std::endl;
	std::cout << "Type: " << typeid(t_def).name() <<  std::endl;
	std::cout << "------------------------------------------------" << std::endl;
	single_execution_series();
	multiple_executions_series(100);

	return 0;
}