#ifndef __CUUTILS_H_INCLUDED__
#define __CUUTILS_H_INCLUDED__
#include <stdio.h>
#include "utils.hpp"
namespace cuutils
{
	 	template <typename T>
	void printf_vector(T *d_v, int size, const std::string& name = "")
	{
		T *h_v = (T *) malloc(size*sizeof(T));
		cudaMemcpy(h_v, d_v, size*sizeof(T),cudaMemcpyDeviceToHost);
		utils::printf_vector(h_v, size, name);
		free(h_v);
	}
	template <typename T>
	void transpose_matrix(cublasHandle_t &handle, T* Z, int s, int n)
	{
		double alpha = 1;
		double beta = 1;
		T* Z_rm; cudaMalloc((void **)&Z_rm, s*n*sizeof(T));
		cudaMemcpy(Z_rm, Z, s*n*sizeof(T), cudaMemcpyDeviceToDevice);
		cublasDgeam(handle,
			CUBLAS_OP_T,
			CUBLAS_OP_T,
			n, s,
			&alpha,
			Z_rm,
			s,
			&beta,
			Z_rm,
			s,
			Z,
			n);
		cudaFree(Z_rm);
	}
	template <typename T>
	__global__ void vvAdd(T *u, T *v, T *z, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			z[id] = u[id] + v[id];
			// increment by the total number of threads running
			// such that we can handle structures of arbitrary size
			id += blockDim.x * gridDim.x;
		}
	}
	template <typename T>
	__global__ void vvSub(T *u, T *v, T *z, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			z[id] = u[id] - v[id];
			// increment by the total number of threads running
			// such that we can handle structures of arbitrary size
			id += blockDim.x * gridDim.x;
		}
	}
	template <typename T>
	__global__ void makeSignumVector(T *v_source, T *v_target, int size)
	{	
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			v_target[id] = (T(0)  < v_source[id]) - (v_source[id] < T(0));
			id += blockDim.x * gridDim.x;
		}
	}
	template <typename T>
	__global__ void abs(T *v_source, T *v_target, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			v_target[id] = (T) fabs(v_source[id]);
			id += blockDim.x * gridDim.x;
		}	
	}
	void getGridBlockSize(int *gridsize, int *blocksize)
	{
		/*
			We want to be able to work with structures of 
			arbitrary sizes. Therefore we chose the gridsize,
			depending on the amout of MPUs.
			the blocksize is the maximum amout of threads, 
			that can be executed within a thread.
		*/
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
}
#endif // __CUUTILS_H_INCLUDED__