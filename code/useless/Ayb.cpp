#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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
		y[i] += A[i*s + j] * y[j]; 
	}
}


void AyPLUSb(double *A, double *y, double *b, int s)
{
	for(int i=0; i<s; i++)
	{
		y[i] = b[i];
		Ay(A, y, s, i);
	}
}

int main()
{
	int s=3;

	// host memory
	double *y = (double *) malloc(s * sizeof(double));
	double *A = (double *) malloc(s * s * sizeof(double));
	double *b = (double *) malloc(s * sizeof(double));

	// init data
	for(int i=0; i<s; i++)
	{
		y[i] = 0.0;
		b[i] = (double) i + 1;
		for (int j=0; j<s; j++)
		{
			int idx =  i*s + j;
			if (j<i)
			{
				A[idx] = 1.0;
			}
			else
			{
				A[idx] = 0.0;
			}
		}
	}
	printf("y\n");
	printf_vector(y, s);
	printf("b\n");
	printf_vector(b, s);
	printf("A\n");
	printf_matrix(A, s, s);

	// calculate y = Ay + b
	AyPLUSb(A, y, b, s);


	printf("y\n");
	printf_vector(y, s);
	printf("b\n");
	printf_vector(b, s);
	printf("A\n");
	printf_matrix(A, s, s);

	free(A), free(y), free(b);
	return 0;
}