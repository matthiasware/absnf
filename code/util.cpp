#include <stdlib.h>
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

void Ay(double *A, double *y, int s, int i)
{
	for (int j=0; j<i; j++)
	{
		y[i] += A[i*s + j] * abs(y[j]); 
	}
}


void AyPLUSb(double *A, double *y, double *b, int s)
{
	for(int i=0; i<s; i++)
	{
		y[i] += b[i];
		Ay(A, y, s, i);
	}
}

// init vector with random values
void initData(double *x, int s)
{
	for(int i=0; i<s; i++)
	{
		x[i] = (double) rand();
	}
}
// init vector with val
void initData(double *x, int s, double val)
{
	for(int i=0; i<s; i++)
	{
		x[i] = val;
	}
}
void matrixMvector(double *A, double *x, double *y, int rows, int cols)
{
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			int idx = i * cols + j;
			y[i] += A[idx] * x[j];
		}
	}
}
void vecAvec(double *a, double *b, double *c, int s)
{
	for(int i=0; i<s; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void initLowerTriangular(double *A, int rows, int cols)
{
	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<cols; j++)
		{
			int idx =  i*cols + j;
			if(j<i)
			{
				A[idx] = (double) rand();
			}
			else
			{
				A[idx] = 0.0;
			}
		}
	}
}
