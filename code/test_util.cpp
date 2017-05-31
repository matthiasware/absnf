#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

int main()
{
	int m = 3, n=2;
	double *A = (double *)malloc(n*m*sizeof(double));
	double *y = (double *)malloc(n * sizeof(double));
	double *b = (double *)malloc(n * sizeof(double));
	double *x = (double *)malloc(m * sizeof(double));
	
	// init A
	for (int i=0; i<m*n; i++)
	{
		A[i] = i + 1;
	}
	// init y, b
	for (int i=0; i<n; i++)
	{
		y[i] = 0;
		b[i] = i + 2;
	}
	// init x
	for (int i=0; i<m; i++)
	{
		x[i] = i + 1;
	}

	printf("A\n");
	printf_matrix(A, m, n);
	printf("x\n");
	printf_vector(x, m);
	printf("b\n");
	printf_vector(b, n);
	printf("y\n");

	matrixMvector(A, x, y, n, m);
	printf("y\n");
	printf_vector(y, n);

	vecAvec(y, b, y, m);
	printf("y\n");
	printf_vector(y, n);
	return 0;
}