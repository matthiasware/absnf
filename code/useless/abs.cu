#include <iostream>
#include <curand.h>

int main()
{
	int n, m, s;
	float *dx, *dz, *dy;
	float *d_dx, *d_dz, *d_dy;
	float *a, *b;
	float *d_a, *d_b;
	float *Z, *L, *J, *Y;
	float *d_Z, *d_L, *d_J, *d_Y;

	n = 10;
	m = 8;
	s = 5;

	int s_dx = n * sizeof(float);
	int s_dy = m * sizeof(float);
	int s_dz = s * sizeof(float);
	int s_a = s * sizeof(float);
	int s_b = m * sizeof(float);
	int s_Z = s * n * sizeof(float);
	int s_L = s * s * sizeof(float);
	int s_J = m * m * sizeof(float);
	int s_Y = m * s * sizeof(float);

	// Host Memory
	dx =  (float *) malloc(s_dx);
	dy =  (float *) malloc(s_dy);
	dz =  (float *) malloc(s_dz);
	a = (float *) malloc(s_a);
	b = (float *) malloc(s_b);
	Z = (float *) malloc(s_Z);
	L = (float *) malloc(s_L);
	J = (float *) malloc(s_J);
	Y = (float *) malloc(s_Y);

	// Device Memory
	cudaMalloc((void **) &d_dx, s_dx);
	cudaMalloc((void **) &d_dy, s_dy);
	cudaMalloc((void **) &d_dz, s_dz);
	cudaMalloc((void **) &d_a, s_a);
	cudaMalloc((void **) &d_b, s_b);
	cudaMalloc((void **) &d_Z, s_Z);
	cudaMalloc((void **) &d_L, s_L);
	cudaMalloc((void **) &d_J, s_J);
	cudaMalloc((void **) &d_Y, s_Y);

	// Free resources
	free(dx), free(dy), free(dz);
	free(a), free(b);
	free(Z), free(L), free(J), free(Y);
	cudaFree(d_dx), cudaFree(d_dy), cudaFree(d_dz);
	cudaFree(d_a), cudaFree(d_b);
	cudaFree(d_Z), cudaFree(d_L), cudaFree(d_J), cudaFree(d_Y);

	return 0;
}