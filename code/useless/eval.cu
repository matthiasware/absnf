#include <cublas_v2.h>
#include "utils.hpp"
// #include "cuutils.h"
#include "absnf.h"

#define t_def double

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
	//
	t_def *h_gamma = (t_def *)malloc(m*sizeof(t_def));
	t_def *h_Gamma = (t_def *)malloc(m*n*sizeof(t_def));
	t_def *h_sigma = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Tss = (t_def *)malloc(s*s*sizeof(t_def));


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
	//
	t_def *d_Gamma; cudaMalloc((void **)&d_Gamma, m*n*sizeof(t_def));
	t_def *d_gamma; cudaMalloc((void **)&d_gamma, m*sizeof(t_def));
	t_def *d_sigma; cudaMalloc((void **)&d_sigma, s*sizeof(t_def));
	t_def *d_K; cudaMalloc((void **)&d_K, m*s*sizeof(t_def));
	t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));


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
	std::cout << "-----------------------------------" << std::endl;
	std::cout << "INPUT EVAL" << std::endl;
	std::cout << "-----------------------------------" << std::endl;
	utils::printf_vector(h_a, s, "a");
	utils::printf_vector(h_b, m, "b");
	std::cout << "-------------" << std::endl;
	utils::printf_vector(h_dz, s, "dz");
	utils::printf_vector(h_dy, m, "dy");
	std::cout << "-------------" << std::endl;
	utils::printf_matrix_C2R(h_Z, s, n, "Z");
	utils::printf_matrix_C2R(h_L, s, s, "L");
	utils::printf_matrix_C2R(h_J, m, n, "J");
	utils::printf_matrix_C2R(h_Y, m, s, "Y");
	std::cout << "-------------" << std::endl;
	utils::printf_vector(h_dx, n, "dx");

	// CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// absnf::eval_core(handle, d_a, d_b,
	// 	 			 d_Z, d_L,
	// 	 			 d_J, d_Y,
	// 	 			 d_dx,
	// 	 			 m, n, s,
	// 	 			 d_dz, d_dy,
	// 	 			 d_abs_dz);

	// RESULTS
	std::cout << "-----------------------------------" << std::endl;
	std::cout << "RESULTS EVAL" << std::endl;
	std::cout << "-----------------------------------" << std::endl;
	cudaMemcpy(h_dz, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(h_dz, s, "dz");

	std::cout << "-------------" << std::endl;
	cudaMemcpy(h_abs_dz, d_abs_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(h_abs_dz, s, "abs_dz");

	std::cout << "-------------" << std::endl;
	cudaMemcpy(h_dy, d_dy, m*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(h_dy, m, "dy");

	std::cout << "------------------------------------------------------" << std::endl;
	std::cout << "GRADIENT" << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;
	// int gridsize, blocksize;
	// cuutils::getGridBlockSize(&gridsize, &blocksize);
	// initTss <<<gridsize, blocksize >>>(d_Tss,d_L, d_dz, s, s*s);
	// cudaMemcpy(h_Tss, d_Tss, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	// utils	::printf_matrix_C2R(h_Tss,s, s, "Tss");

//  http://cuda-programming.blogspot.de/2013/01/thread-and-block-heuristics-in-cuda.html

	std::cout << "------------------------------------------------------" << std::endl;
	utils::printf_vector(h_Z, s*n, "Z");
	utils::printf_vector(h_L, s*s, "L");
	utils::printf_vector(h_J, m*n, "J");
	utils::printf_vector(h_Y, m*s, "Y");

	// FREE STUFF
	free(h_a); 
	free(h_b);
	free(h_Z);
	free(h_L);
	free(h_dx);
	free(h_dz);
	free(h_abs_dz);
	free(h_Gamma);
	free(h_gamma);
	free(h_Y);
	free(h_J);
	free(h_sigma);
	free(h_Tss);

	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_Z);
	cudaFree(d_L);
	cudaFree(d_dx);
	cudaFree(d_dz);
	cudaFree(d_abs_dz);
	cudaFree(d_Gamma);
	cudaFree(h_gamma);
	cudaFree(d_dy);
	cudaFree(d_Y);
	cudaFree(d_J);
	cudaFree(d_sigma);
	cudaFree(d_Tss);


	cublasDestroy(handle);
	return 0;
}