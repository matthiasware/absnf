#include <cublas_v2.h>
#include "utils.hpp"

#define t_def double

int main()
{
	int n=4, s=3;

	t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Z = (t_def *)malloc(s*n*sizeof(t_def));
	t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dx =(t_def *)malloc(n*sizeof(t_def));
	t_def *h_dz =(t_def *)malloc(s*sizeof(t_def));

	// HOST MEMORY
	utils::fillVector(h_dz, s, (t_def) 0);
	utils::fillRandVector(h_a, s, 0, 5, 1, utils::VALUEOP::INT);
	utils::fillRandVector(h_dx, n, 0, 5, 2, utils::VALUEOP::INT);
	utils::fillRandMatrix(h_Z, s, n, 0, 5, 3, utils::MATRIXOPT::NONE, utils::VALUEOP::INT);
	utils::fillRandMatrix(h_L, s, s, 1, 5, 4, utils::MATRIXOPT::LOWER, utils::VALUEOP::INT);

	// DEVICE MEMORY
	t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
	t_def *d_Z; cudaMalloc((void **)&d_Z, s*n*sizeof(t_def));
	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_dx; cudaMalloc((void **)&d_dx, n*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	t_def *d_abs_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));

	//COPY DATA
	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z, h_Z,  s*n*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, h_dx, n*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_abs_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);


	utils::printf_vector(h_a, s);
	utils::printf_vector(h_dz, s);
	utils::printf_vector(h_dx, n);
	utils::printf_matrix(h_Z, s, n);
	utils::printf_matrix(h_L, s, s);


	// FREE STUFF
	free(h_a); free(h_L); free(h_Z);
	free(h_dz); free(h_dx);
	cudaFree(d_a);
	cudaFree(d_Z);
	cudaFree(d_L);
	cudaFree(d_dx);
	cudaFree(d_dz);
	return 0;
}