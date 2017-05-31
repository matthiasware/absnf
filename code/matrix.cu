// C(m,n) = A(m,k) * B(k,n)
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#define DT_MF double

void fill_rand(float *A, int size)
{
	for(int i=0; i<size;i++)
	{
		A[i] = i;
	}
}

void printf_matrix(float *A, int cols, int rows)
{
	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<cols; j++)
		{
			int idx = i*cols + j;
			printf(" %f", A[idx]);
		}
		printf("\n");
	}
}

void gpu_blas_mul(const float *A, const float *B, float *C, 
				  const int m, const int k, const int n)
{
	int lda=m, ldb=k, ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// create handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
			    A, lda, B, ldb, beta, C, ldc);

	cublasDestroy(handle);
}

int main() {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
	
	float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

	// Randomly fill arrays
	fill_rand(h_A, nr_rows_A * nr_cols_A);
	fill_rand(h_B, nr_rows_B * nr_cols_B);


	// copy data to device
	cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);


	// MULTIPLICATION
	gpu_blas_mul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	// cody result to host
	cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

	printf_matrix(h_A, nr_rows_A, nr_cols_A);
	printf_matrix(h_B, nr_rows_B, nr_cols_B);
	printf_matrix(h_C, nr_rows_C, nr_cols_C);


	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}