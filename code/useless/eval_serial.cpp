#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

int main()
{
	// allocate memory
	int n = 4;
	int m = 2;
	int s = 3;

	double *dx = (double *) malloc(n * sizeof(double));
	double *dz = (double *) malloc(s * sizeof(double));
	double *dy = (double *) malloc(m * sizeof(double));
	double *a = (double *) malloc(s * sizeof(double));
	double *b = (double *) malloc(m * sizeof(double));
	double *Z = (double *) malloc(s * n * sizeof(double));
	double *L = (double *) malloc(s * s * sizeof(double));
	double *J = (double *) malloc(m * n * sizeof(double));
	double *Y = (double *) malloc(m * s * sizeof(double));


	// init data
	initData(dx, n);
	initData(dz, s, 0);
	initData(dy, m, 0);
	initData(a, s);
	initData(b, m);
	initData(Z, s * n);
	initLowerTriangular(L, s, s);
	initData(J, m*n);
	initData(Y, m*s);

	printf("dz\n");
	printf_vector(dz, s);
	printf("a\n");
	printf_vector(a, s);
	printf("Z\n");
	printf_matrix(Z, s, n);
	printf("dx\n");
	printf_vector(dx, n);
	printf("L\n");
	printf_matrix(L,s,s);

	// Z*dx -> dz
	matrixMvector(Z, dx, dz, s, n);
	// dz <- L |dz| + db;
	AyPLUSb(L, dz, a, s);


	// free memory
	free(dx), free(dz), free(dy), free(a);
	free(b), free(Z), free(L), free(J), free(Y);
	return 0;
}