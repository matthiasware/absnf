#include <iostream>
#include <stdio.h>

void printf_matrix(double *A, int cols, int rows)
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
void printf_vector(double *A, int cols)
{
	for(int i=0; i<cols; i++)
	{
		printf(" %f", A[i]);
	}
	printf("\n");
}
void LZ(double *L, double *Z, int s)
{
	// go through each row
	for(int i=0, i<s; i++)
	{
		// get index of L
		l_start = Ĺ + i * sizeof(double);

		// get index of Z
		z_start = Z + i * sizeof(double);
		for(int j=0; j<i; j++)
		{
			
		}
	}
}

int main()
{
	int s=3;

	// host memory
	double *h_a =  (double *) malloc(s * sizeof(double));
	double *h_dz = (double *) malloc(s * sizeof(double));
	double *h_abs_dz = (double *) malloc(s * sizeof(double));
	double *h_L = (double *) malloc(s * s * sizeof(double));

	// divice memory
	double *d_a;
	double *d_dz;
	double *d_abs_dz;
	double *d_L;

	// init data
	for(int i=0; i<s; i++)
	{
		h_a[i] = 1.0;
		h_dz[i] = 0.0;
		h_abs_dz[i] = 0.0;
		for (int j=0; j<s; j++)
		{
			int idx =  i*s + j;
			if (j<i)
			{
				h_L[idx] = 1.0;
			}
			else
			{
				h_L[idx] = 0.0;
			}
		}
	}
	printf_vector(h_a, s);
	printf_vector(h_dz, s);
	printf_vector(h_abs_dz, s);
	printf_matrix(h_L, s, s);

	cudaMalloc(&d_a, s * sizeof(double));
	cudaMalloc(&d_dz, s * sizeof(double));
	cudaMalloc(&d_abs_dz, s* sizeof(double));
	cudaMalloc(&d_L, s * s * sizeof(double));

	// free resoures
	cudaFree(d_a), cudaFree(d_dz), cudaFree(d_abs_dz), cudaFree(d_L);
	free(h_a), free(h_dz), free(h_abs_dz), free(h_L);
	return 0;
}