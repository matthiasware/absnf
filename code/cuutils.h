#ifndef __CUUTILS_H_INCLUDED__
#define __CUUTILS_H_INCLUDED__
#include <stdio.h>
#include "utils.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <cusolverDn.h>
#include <sstream>

#define DEBUG_CUUTILS true

namespace cuutils
{	
	// ---------------------------------------------------
	//  DECLARATIONS
	// ---------------------------------------------------
	void cuda_err_handler(const char *msg, const char *file, int line, bool abort);
	const char* cublasGetErrorString(cublasStatus_t status);
	const char* cusolverGetErrorString(cusolverStatus_t error);
	void check(cudaError_t code);
	void check(cublasStatus_t code);
	void check(cusolverStatus_t code);
	void getGridBlockSize(int *gridsize, int *blocksize);

	template <typename T>
	T __device__ sign(T *val);

	template <typename T>
	void printf_vector(T *d_v, int size, const std::string& name = "");

	template <typename T>
	void transpose_matrix(cublasHandle_t &handle, T* Z, int s, int n);

	template <typename T>
	__global__ void vvAdd(T *u, T *v, T *z, int size);

	template <typename T>
	__global__ void vvSub(T *u, T *v, T *z, int size);

	template <typename T>
	__global__ void makeSignumVector(T *v_source, T *v_target, int size);

	template <typename T>
	__global__ void abs(T *v_source, T *v_target, int size);


	// ---------------------------------------------------
	//  IMPLEMENTATION
	// ---------------------------------------------------
	void cuda_err_handler(const char *msg, const char *file, int line, bool abort=true)
	{
	std::stringstream ss;
	ss << file << "(" << line << ")" << " : " << msg;
	std::string file_and_line;
	ss >> file_and_line;
	if (abort)
		throw std::runtime_error(file_and_line);
	else
		std::cout << "GPUassert: " << file_and_line << std::endl;
	}
	const char* cublasGetErrorString(cublasStatus_t status)
	{
	    switch(status)
	    {
	        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
	        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
	        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
	        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	    }
	    	return "unknown error";
	}
	const char* cusolverGetErrorString(cusolverStatus_t error)
	{
	    switch (error)
	    {
	        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_SUCCESS";
	        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
	        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
	        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
	        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
	        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
	        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
	        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	    }
    	return "unknown error";
	}
	void check(cudaError_t code)
	{
		if(code != cudaSuccess)
			cuda_err_handler(cudaGetErrorString(code), __FILE__, __LINE__);
	}
	void check(cublasStatus_t code)
	{
		if(code != CUBLAS_STATUS_SUCCESS)
			cuda_err_handler(cuutils::cublasGetErrorString(code), __FILE__, __LINE__);
	}
	void check(cusolverStatus_t code)
	{
		if(code != CUSOLVER_STATUS_SUCCESS)
			cuda_err_handler(cuutils::cusolverGetErrorString(code), __FILE__, __LINE__);
	}
	template <typename T>
	T __device__ sign(T *val)
	{
		return (T(0) < *val) - (*val < T(0));
	}
	template <typename T>
	void printf_vector(T *d_v, int size, const std::string& name)
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