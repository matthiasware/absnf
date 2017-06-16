#include <cublas_v2.h>
#include "utils.hpp"
#include "cuutils.h"

#define t_def double
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define BLOCKSIZE 1024

void eval(cublasHandle_t &handle,
		  t_def *a, t_def *b, 
		  t_def *Z, t_def *L, 
		  t_def *J, t_def *Y,
		  t_def *dx,
		  int m, int n, int s,
		  t_def *dz, t_def *dy,
		  t_def *abs_dz)

{
	
	double alpha = 1;
	double beta = 1;
	// dz = Z * dx
	// dz = alpha * (Z * dx) + beta * dz
	cublasDgemv(handle, CUBLAS_OP_N, s, n, &alpha,
				Z, s, dx, 1, &beta, dz, 1);

	// dz = dz + a
	cuutils::vvAdd <<<(s + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >>>(a, dz, dz, s);

	// dz = dz + L * |dz|
	for(int i=0; i<s; i++)
	{
		cublasDgemv(handle, CUBLAS_OP_N, 1, i, &alpha, (L + i * s), 1,
					abs_dz, 1, &beta, &dz[i], 1);
		// TODO MAKE SMARTER
		cuutils::abs <<<1,1>>>(&dz[i], &abs_dz[i], 1);
	}
	// dy = b
	cudaMemcpy(dy, b, m*sizeof(t_def), cudaMemcpyDeviceToDevice);
	// dy = J * dx
	cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha,
				J, m, dx, 1, &beta, dy, 1);

	// dy = dy + Y * |dz|
	// dy = beta * dy + alpha(Y*abs_dz)
	cublasDgemv(handle, CUBLAS_OP_N, m, s, &alpha,
				Y, m, abs_dz, 1, &beta, dy, 1);	
}

// test
int main()
{
	int n=4, s=3, m=2;

	t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Z = (t_def *)malloc(s*n*sizeof(t_def));
	t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dx =(t_def *)malloc(n*sizeof(t_def));
	t_def *h_dz =(t_def *)malloc(s*sizeof(t_def));
	t_def *h_abs_dz =(t_def *)malloc(s*sizeof(t_def));
	//
	t_def *h_dy = (t_def *)malloc(m*sizeof(t_def));
	t_def *h_b = (t_def *)malloc(m*sizeof(t_def));
	t_def *h_J = (t_def *)malloc(m*n*sizeof(t_def));
	t_def *h_Y = (t_def *)malloc(m*s*sizeof(t_def));


	// HOST MEMORY
	utils::fillVector(h_dz, s, (t_def) 0);
	utils::fillVector(h_abs_dz, s, (t_def) 0);
	utils::fillRandVector(h_a, s, -5, 5, 1, utils::VALUEOP::INT);
	utils::fillRandVector(h_dx, n, -5, 5, 2, utils::VALUEOP::INT);
	utils::fillRandMatrixCM(h_Z, s, n, -5, 5, 3, utils::MATRIXOPT::NONE, utils::VALUEOP::INT);
	utils::fillRandMatrix(h_L, s, s, -1, 5, 4, utils::MATRIXOPT::LOWER, utils::VALUEOP::INT);
	//
	utils::fillVector(h_dy, m, (t_def) 0);
	utils::fillRandVector(h_b, m, 0, 5, 1, utils::VALUEOP::INT);
	utils::fillRandMatrixCM(h_J, m, n, 0, 5, 3, utils::MATRIXOPT::NONE, utils::VALUEOP::INT);
	utils::fillRandMatrixCM(h_Y, m, s, 0, 5, 3, utils::MATRIXOPT::NONE, utils::VALUEOP::INT);


	// DEVICE MEMORY
	t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
	t_def *d_Z; cudaMalloc((void **)&d_Z, s*n*sizeof(t_def));
	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_dx; cudaMalloc((void **)&d_dx, n*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	t_def *d_abs_dz; cudaMalloc((void **)&d_abs_dz, s*sizeof(t_def));
	//
	t_def *d_dy; cudaMalloc((void **)&d_dy, m*sizeof(t_def));
	t_def *d_b; cudaMalloc((void **)&d_b, m*sizeof(t_def));
	t_def *d_J; cudaMalloc((void **)&d_J, m*n*sizeof(t_def));
	t_def *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(t_def));


	//COPY DATA TO DEVICE
	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z, h_Z,  s*n*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, h_dx, n*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_abs_dz, h_abs_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);
	//
	cudaMemcpy(d_dy, h_dy,  m*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b,  m*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_J, h_J,  m*n*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y,  m*s*sizeof(t_def), cudaMemcpyHostToDevice);

	// COUT
	utils::printf_vector(h_a, s, "a");
	utils::printf_vector(h_b, m, "b");
	std::cout << "-------------" << std::endl;
	utils::printf_vector(h_dz, s, "dz");
	utils::printf_vector(h_dy, m, "dy");
	std::cout << "-------------" << std::endl;
	utils::printf_matrix_C2R(h_Z, s, n, "Z");
	utils::printf_matrix(h_L, s, s, "L");
	utils::printf_matrix_C2R(h_J, m, n, "J");
	utils::printf_matrix_C2R(h_Y, m, s, "Y");
	std::cout << "-------------" << std::endl;
	utils::printf_vector(h_dx, n, "dx");

	// CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	eval(handle, d_a, d_b,
		 d_Z, d_L,
		 d_J, d_Y,
		 d_dx,
		 m, n, s,
		 d_dz, d_dy,
		 d_abs_dz);

	// RESULTS
	std::cout << "-------------" << std::endl;
	cudaMemcpy(h_dz, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(h_dz, s, "dz");

	std::cout << "-------------" << std::endl;
	cudaMemcpy(h_abs_dz, d_abs_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(h_abs_dz, s, "abs_dz");

	std::cout << "-------------" << std::endl;
	cudaMemcpy(h_dy, d_dy, m*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(h_dy, m, "dy");

//  http://cuda-programming.blogspot.de/2013/01/thread-and-block-heuristics-in-cuda.html

	// FREE STUFF
	free(h_a); free(h_L); free(h_Z);
	free(h_dz); free(h_dx);
	cudaFree(d_a);
	cudaFree(d_Z);
	cudaFree(d_L);
	cudaFree(d_dx);
	cudaFree(d_dz);

	cublasDestroy(handle);
	return 0;
}