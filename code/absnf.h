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
		//  ----------------------------------
		// dz = a
		//  ----------------------------------
		cudaMemcpy(dz, a, s*sizeof(T), cudaMemcpyDeviceToDevice);
		//  ----------------------------------		
		// dz = Z * dx + dz
		// dz = alpha * (Z * dx) + beta * dz
		//  ----------------------------------
		double alpha = 1;
		double beta = 1;
		cublasDgemv(handle, CUBLAS_OP_N, s, n, &alpha,
					Z, s, dx, 1, &beta, dz, 1);
		//  ----------------------------------
		// dz = dz + a
		// dz = dz + L * |dz|
		//  ----------------------------------
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
		//  ----------------------------------
		// dy = b
		//  ----------------------------------
		cudaMemcpy(dy, b, m*sizeof(T), cudaMemcpyDeviceToDevice);
		//  ----------------------------------
		// dy = J * dx
		//  ----------------------------------
		cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha,
					J, m, dx, 1, &beta, dy, 1);
		//  ----------------------------------
		// dy = dy + Y * |dz|
		// dy = beta * dy + alpha(Y*abs_dz)
		//  ----------------------------------
		cublasDgemv(handle, CUBLAS_OP_N, m, s, &alpha,
					Y, m, abs_dz, 1, &beta, dy, 1);	
	};
	/** Evaluation of ABS-NF-Function
		Assumes sufficient memoy to available on the device
	
		INPUT:
		@param h_a: host mem (1*s)
		@param h_b: host mem (1*m)
		@param h_Z: host mem (s*n), column-major
		@param h_L: host mem (s*s), row-major
		@param h_J: host mem (m*n), column-major
		@param h_Y: host mem (m*s), column-major
		@param h_dx: host mem (1*n)
		@param m
		@param n
		@param s
		OUTPUT:
		@param h_dz: host mem (1*s)
		@param h_dy: host mem (1*m)

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
	T __device__ sign(T *val)
	{
		return (T(0) < *val) - (*val < T(0));
	}
	template <typename T>
	void __device__ initTssValue(T *Tss, T *L, T *dz, int i, int j, int id)
	{
		if(i==j)
		{
			Tss[id] = 1;
		}
		else if(j>i)
		{
			Tss[id] = 0;
		}
		else
		{
			// Tss[id] = 0 - L[id] * ((double(0) < dz[j]) - (dz[j] < double(0)));
			Tss[id] = 0 - L[id] * sign(&dz[j]);
		}
	}
	template <typename T>
	// row major or column major?
	void __global__ initTss(T *Tss, T *L, T *dz, int s, int size)
	{
		int i = threadIdx.x;
		int j = blockIdx.x;
		int id = i*s + j;
		while(id < size && j < s)
		{
			if(i<s)
			{
				initTssValue(Tss,L,dz,i,j,id);
				i+=blockDim.x;
			}
			else
			{
				i = i%s;
				j = j + gridDim.x;
			}
			id = i*s + j;
		}
	}
	template <typename T>
	void __global__ initIdentity(T *I, int s)
	{
		int i = threadIdx.x;
		int j = blockIdx.x;
		int id = i*s + j;
		while(id < s*s && j < s)
		{
			if(i<s)
			{
				if(i == j)
				{
					I[id] = 1;
				}
				else
				{
					I[id] = 0;
				}
				i+=blockDim.x;
			}
			else
			{
				i = i%s;
				j = j + gridDim.x;
			}
			id = i*s + j;
		}		
	}
	template <typename T>
	void __global__ multWithDz(T *A, T *dz, int s)
	{
		int i = threadIdx.x;
		int j = blockIdx.x;
		int id = i*s + j;
		while(id < s*s && j < s)
		{
			if(i<s)
			{
				if(A[id] != T(0)) 
					A[id] = A[id] * sign(&dz[j]);
				i+=blockDim.x;
			}
			else
			{
				i = i%s;
				j = j + gridDim.x;
			}
			id = i*s + j;
		}			
	}
	template <typename T>
	/** Calculates Inverse of matrix A
	 	results are written to I
	 	@param A: device mem 
	 			  lower triangular matrix (s*s)
	 			  row major
	 	@param I: device mem (s*s)
	 			  Identity matrix
	 			  result (s*s) row major inverse of A
	*/
	void getTriangularInverse(cublasHandle_t &handle,
					  		   			 T *A, T *I, int s)
	{
		double alpha = 1;
		// cublasStrsm_v2(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
		// 			   CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,
		// 			   nCols,nCols,&t_alphA,D_L,nCols,D_B,nCols);
		// stores inverse of Tss in I
		// cuutils::printf_vector(A,s*s, "L");
		// cuutils::printf_vector(I,s*s, "I");

		cublasStatus_t stat = cublasDtrsm(
								handle,
						      	CUBLAS_SIDE_LEFT,
				    	      	CUBLAS_FILL_MODE_UPPER,
				    		  	CUBLAS_OP_N,
				    		  	CUBLAS_DIAG_UNIT,
				    		  	s,s,
				    	   	  	&alpha,
				    	      	A,
				    		  	s,
				    	      	I,
				    	      	s);
	}
	template <typename T>
	void mmp(cublasHandle_t &handle, T *d_Y, T* d_I, T *d_K, int m, int s)
	{	
		cuutils::printf_vector(d_Y ,m*s, "Y");
		cuutils::printf_vector(d_I ,s*s, "I");

		double alpha = 1;
		double beta = 0;
		cublasDgemm(handle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					m,s,s,
					&alpha,
					d_Y,
					m,
					d_I,
					s,
					&beta,
					d_K,
					m);
	}
	/** Calculates the gradient
		No checks for memory availability are done

		INPUT:
		@param h_a host mem (1*s)
		@param h_b host mem (1*m)
		@param h_Z host mem (s*n) column major
		@param h_L host mem (s*s) row major
		@param h_J host mem (m*n) column major
		@param h_dz host mem (1*s)
		OUTPUT:
		@param h_gamma host mem (1*m)
		@paran h_Gamma host mem (m*n) column major

	*/
	template <typename T>
	void gradient(T *h_a, T *h_b, 
			  	  T *h_Z, T *h_L, 
			  	  T *h_J, T *h_Y,
			  	  T* h_dz,
			  	  int m, int n, int s,
			  	  T *h_gamma, T *h_Gamma)
	{
		T *d_a; cudaMalloc((void **)&d_a, s*sizeof(T));
		T *d_b; cudaMalloc((void **)&d_b, m*sizeof(T));
		T *d_Z; cudaMalloc((void **)&d_Z, s*n*sizeof(T));
		T *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(T));
		T *d_J; cudaMalloc((void **)&d_J, m*n*sizeof(T));
		T *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(T));		
		T *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(T));
		T *d_gamma; cudaMalloc((void **)&d_gamma, m*sizeof(T));
		T *d_Gamma; cudaMalloc((void **)&d_Gamma, m*n*sizeof(T));
		T *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(T));
		T *d_I; cudaMalloc((void **)&d_I, s*s*sizeof(T));
		T *d_K; cudaMalloc((void **)&d_K, m*s*sizeof(T));

		cudaMemcpy(d_a, h_a,  s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b,  m*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Z, h_Z,  s*n*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_L, h_L,  s*s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_J, h_J,  m*n*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Y, h_Y,  m*s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dz, h_dz, s*sizeof(T), cudaMemcpyHostToDevice);

		cublasHandle_t handle;
		cublasCreate(&handle);
		//  ----------------------------------
		//  d_Tss = diag(1) - L * diag(sign(dz))
		//  ----------------------------------
		int gridsize, blocksize;
		cuutils::getGridBlockSize(&gridsize, &blocksize);
		initTss <<<gridsize, blocksize >>>(d_Tss,d_L, d_dz, s, s*s);
		//  ----------------------------------
		//  d_I = diag(1) // room for improvement, operations can be merged
		//  ----------------------------------		
		initIdentity <<<gridsize, blocksize >>> (d_I, s);
		//  ----------------------------------
		//  d_I = d_Tss * X
		//  ----------------------------------	
		getTriangularInverse(handle, d_Tss, d_I, s);
		//  ----------------------------------
		//	d_I = d_I * diag(sign(dz))
		//  ----------------------------------
		multWithDz <<<gridsize, blocksize >>>(d_I, d_dz, s);
		//  ----------------------------------
		//	d_K = d_Y * d_I
		//  ----------------------------------
		double alpha = 1;
		double beta = 0;
		cublasDgemm(handle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,	// d_I is in row major format
					m,s,s,
					&alpha,
					d_Y,
					m,
					d_I,
					s,
					&beta,
					d_K,
					m);
		// cuutils::printf_vector(d_K, m*s, "K");
		//  ----------------------------------
		//	d_gamma = d_b
		//  d_Gamma = J
		//  ----------------------------------
		cudaMemcpy(d_gamma, d_b, m*sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_Gamma, d_J, m*n*sizeof(T), cudaMemcpyDeviceToDevice);
		//  ----------------------------------
		//	d_gamma = d_gamma + K*a
		//  ----------------------------------
		beta = 1;
		cublasDgemv(handle, CUBLAS_OP_N, m, s, &alpha,
					d_K, m, d_a, 1, &beta, d_gamma, 1);
		//  ----------------------------------
		//  d_Gamma = d_Gamma + K*Z
		//  ----------------------------------
		cublasDgemm(handle,
					CUBLAS_OP_N,
					CUBLAS_OP_N,	// d_I is in row major format
					m,n,s,
					&alpha,
					d_K,
					m,
					d_Z,
					s,
					&beta,
					d_Gamma,
					m);

		// ----------------------------------
		cudaMemcpy(h_gamma, d_gamma, m*sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Gamma, d_Gamma, m*n*sizeof(T), cudaMemcpyDeviceToHost);

		cudaFree(d_a); 
		cudaFree(d_b);
		cudaFree(d_Z);
		cudaFree(d_L);
		cudaFree(d_J);
		cudaFree(d_Y);
		cudaFree(d_dz);
		cudaFree(d_gamma);
		cudaFree(d_Gamma);
		cudaFree(d_Tss);
		cudaFree(d_I);
		cudaFree(d_K);
	}
	template <typename T>
	void solve(T *h_a, T *h_b,
			   T *h_Z, T *h_L,
			   T *h_J, T *h_Y,
			   T *dy,
			   int m, int n, int s,
			   T *h_dx, T *h_dz)
	{

	}
}

#endif // __ABSNF_H_INCLUDED__

