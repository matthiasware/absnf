#ifndef __ABSNF_H_INCLUDED__
#define __ABSNF_H_INCLUDED__

#include <cublas_v2.h>
#include "cuutils.h"

namespace absnf
{
	/** Helper for eval
		All memoy pointer are device pointer
	*/
	template <typename T>
	void eval_core(cublasHandle_t &handle,
			  	   T *a, T *b, 
			  	   T *Z, T *L, 
			  	   T *J, T *Y,
			  	   T *dx,
			  	   int m, int n, int s,
			  	   T *dz, T *dy,
			  	   T *abs_dz)

	{ 
		
	// dz = a
		cudaMemcpy(dz, a, s*sizeof(T), cudaMemcpyDeviceToDevice);
		
	// dz = Z * dx + dz
	// dz = alpha * (Z * dx) + beta * dz
		double alpha = 1;
		double beta = 1;
		cublasDgemv(handle, CUBLAS_OP_N, s, n, &alpha,
					Z, s, dx, 1, &beta, dz, 1);

	// dz = dz + a
	// dz = dz + L * |dz|
		for(int i=0; i<s; i++)
		{
			cublasDgemv(handle, CUBLAS_OP_N,
					    1, i,
					    &alpha,
					    &L[i * s], 1,
						abs_dz, 1,
						&beta,
						&dz[i], 1);
			cuutils::abs <<<1,1>>>(&dz[i], &abs_dz[i], 1);
		}
	// dy = b
		cudaMemcpy(dy, b, m*sizeof(T), cudaMemcpyDeviceToDevice);
	// dy = J * dx
		cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha,
					J, m, dx, 1, &beta, dy, 1);

	// dy = dy + Y * |dz|
	// dy = beta * dy + alpha(Y*abs_dz)
		cublasDgemv(handle, CUBLAS_OP_N, m, s, &alpha,
					Y, m, abs_dz, 1, &beta, dy, 1);	
	};
	/** Evaluation of ABS-NF-Function
		Assumes sufficient memoy to available on the device
	
		@param h_a: host mem a (1*s)
		@param h_b: host mem b (1*m)
		@param h_Z: host mem Z (s*n), column-major
		@param h_L: host mem L (s*s), row-major
		@param h_J: host mem L (m*n), column-major
		@param h_Y: host mem Y (m*s), column-major
		@param h_dx: host mem dx, (1*n)
		@param m
		@param n
		@param s
		@param h_dz: (result) host mem dz (1*s)
		@param h_dy: (result) host mem dy (1*m)

	*/
	template <typename T>
	void eval(T *h_a, T *h_b, 
			  T *h_Z, T *h_L, 
			  T *h_J, T *h_Y,
			  T *h_dx,
			  int m, int n, int s,
			  T *h_dz, T *h_dy)
	{
		T *d_a; cudaMalloc((void **)&d_a, s*sizeof(T));
		T *d_b; cudaMalloc((void **)&d_b, m*sizeof(T));
		T *d_Z; cudaMalloc((void **)&d_Z, s*n*sizeof(T));
		T *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(T));
		T *d_J; cudaMalloc((void **)&d_J, m*n*sizeof(T));
		T *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(T));		
		T *d_dx; cudaMalloc((void **)&d_dx, n*sizeof(T));
		T *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(T));
		T *d_abs_dz; cudaMalloc((void **)&d_abs_dz, s*sizeof(T));
		T *d_dy; cudaMalloc((void **)&d_dy, m*sizeof(T));

		cudaMemcpy(d_a, h_a,  s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b,  m*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Z, h_Z,  s*n*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_L, h_L,  s*s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_J, h_J,  m*n*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Y, h_Y,  m*s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dx, h_dx, n*sizeof(T), cudaMemcpyHostToDevice);

		cublasHandle_t handle;
		cublasCreate(&handle);
		eval_core(handle, d_a, d_b,
		 		  d_Z, d_L,
		 		  d_J, d_Y,
		 		  d_dx,
		 		  m, n, s,
		 		  d_dz, d_dy,
		 		  d_abs_dz);
		cudaMemcpy(h_dz, d_dz, s*sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dy, d_dy, m*sizeof(T), cudaMemcpyDeviceToHost);

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
	}
	template <typename T>
	void __global__ initTss(T *Tss, T *L, T *dz, int s, int size)
	{
		int j = blockIdx.x;
		int i = threadIdx.x;
		int id = j*s + i;
		while(id < size)
		{
			if(i < s)
			{
				if(i == j)
				{
					Tss[id] = 1;
				}
				else if(j > i)
				{
					Tss[id] = 0;
				}
				else
				{
					Tss[id] = L[id] * (double(0) < dz[j]) - (dz[j] < double(0));
				}
				i += blockDim.x;
			}
			else
			{
				i = threadIdx.x;
				j = j + gridDim.x;
			}
			id = j*s + i;
		}
	}
}


		// cublasDgeam(handle,
		// 	CUBLAS_OP_T,
		// 	CUBLAS_OP_T,
		// 	n, s,
		// 	&alpha,
		// 	Z,
		// 	s,
		// 	&beta,
		// 	Z,
		// 	s,
		// 	Z_rm,
		// 	n);

		// int gridsize, blocksize;
		// cuutils::getGridBlockSize(&gridsize, &blocksize);
		// cuutils::vvAdd <<<gridsize, blocksize >>>(a, dz, dz, s);

#endif // __ABSNF_H_INCLUDED__

