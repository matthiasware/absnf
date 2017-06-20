#include <cublas_v2.h>
#include "absnf.h"
#include <vector>
#include "utils.hpp"
#define t_def double

bool test_eval_Singular(t_def *h_a, t_def *h_b, 
			   			t_def *h_Z, t_def *h_L, 
			   			t_def *h_J, t_def *h_Y,
			   			t_def *h_dx,
			   			int m, int n, int s,
			   			t_def *h_dz_expected, t_def *h_dy_expected)
{
	// convert to column major storage
	utils::rowColConversion(h_Z, s, n, true);
	utils::rowColConversion(h_J, m, n, true);
	utils::rowColConversion(h_Y, m, s, true);

	// results
	t_def *h_dz =(t_def *)malloc(s*sizeof(t_def));
	t_def *h_dy = (t_def *)malloc(m*sizeof(t_def));

	// calculate results
	absnf::eval(h_a, h_b,
			    h_Z, h_L, 
			    h_J, h_Y, 
			    h_dx, 
			    m, n, s, 
			    h_dz, h_dy);

	// compare results
	if(!utils::vectors_equals(h_dz, h_dz_expected, s, false))
		return false;
	if(!utils::vectors_equals(h_dy, h_dy_expected, m, false))
		return false;
	return true;
}
bool test_initTss_Singular(t_def *h_L, 
			   			   t_def *h_dz,
						   int s,
			   			   t_def *h_Tss_expected)
{
	bool success = true;
	t_def *h_Tss = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));

	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	int gridsize, blocksize;
	cuutils::getGridBlockSize(&gridsize, &blocksize);
	absnf::initTss <<<gridsize, blocksize >>>(d_Tss,d_L, d_dz, s, s*s);
	cudaMemcpy(h_Tss, d_Tss, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);

	if(!utils::vectors_equals(h_Tss, h_Tss_expected, s*s, false))
		success = false;


	cudaFree(d_L);
	cudaFree(d_dz);
	cudaFree(d_Tss);
	free(h_Tss);
	return success;
}
bool test_getTriangularInverse_Singular(t_def *h_A, t_def *h_I_expected, int s)
{
	bool success = true;
	t_def *h_I_actual = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *d_A; cudaMalloc((void **)&d_A, s*s*sizeof(t_def));
	t_def *d_I; cudaMalloc((void **)&d_I, s*s*sizeof(t_def));

	cudaMemcpy(d_A, h_A, s*s*sizeof(t_def), cudaMemcpyHostToDevice);
	// ..............
	int gridsize, blocksize;
	cuutils::getGridBlockSize(&gridsize, &blocksize);
	absnf::initIdentity<<<gridsize, blocksize>>>(d_I, s);
	// ..............
	cublasHandle_t handle;
	cublasCreate(&handle);
	absnf::getTriangularInverse(handle, d_A, d_I, s);
	// ..............
	cudaMemcpy(h_I_actual, d_I, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	if(!utils::vectors_equals(h_I_actual, h_I_expected, s*s))
		success = false;
	// utils::printf_vector(h_I_actual,s*s, "Inverse");
	cudaFree(d_A);
	cudaFree(d_I);
	free(h_I_actual);
	return success;
}
bool test_getTriangularInverse()
{
	bool success = true;
	int s=4;
	std::vector<t_def> h_matrix = {1, 0, 0, 0,
								   4, 1, 0, 0,
								   8, 0, 1, 0,
								   2, 0, -7, 1};
	std::vector<t_def> h_I_expected = { 1, 0, 0, 0,
								       -4, 1, 0, 0,
								       -8, 0, 1, 0,
								       -58,0, 7, 1};
    // utils::rowColConversion(&h_matrix[0], s, s, true);
	test_getTriangularInverse_Singular(&h_matrix[0], &h_I_expected[0], s);

	return success;
}
bool test_initTss()
{
	int s=4;
	std::vector<t_def> L = {0, 0, 0, 0,
						    4, 0, 0, 0,
						    8, 9, 0, 0,
						    2, 1, 7, 0};
	std::vector<t_def> dz = {-1, 0, 1, -1};
	std::vector<t_def> Tss_expected = {1, 0, 0, 0,
									   4, 1, 0, 0,
									   8, 0, 1, 0,
									   2, 0, -7, 1};
	test_initTss_Singular(&L[0], &dz[0], s, &Tss_expected[0]);
	return true;
}
bool test_initIdentity()
{
	bool success = true;
	int s=4;
	std::vector<t_def> h_I_expected = {1,0,0,0,
							 0,1,0,0,
							 0,0,1,0,
							 0,0,0,1};
	t_def *h_I_actual = (t_def *) malloc(s*s*sizeof(t_def));
	t_def *d_I; cudaMalloc((void **)&d_I, s*s*sizeof(t_def));

	//  ----------------------------------
	int gridsize, blocksize;
	cuutils::getGridBlockSize(&gridsize, &blocksize);
	absnf::initIdentity<<<gridsize,blocksize>>>(d_I, s);
	//  ----------------------------------

	cudaMemcpy(h_I_actual, d_I, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	if(!utils::vectors_equals(h_I_actual, &h_I_expected[0], s*s, false))
		success = false;

	cudaFree(d_I);
	free(h_I_actual);
	return success;

}
bool test_eval()
{
	int n=4;
	int s=3;
	int m=2;
	// n
	std::vector<t_def> a = {4, 4,-3};
	// m
	std::vector<t_def> b = {4, 4};
	// s * n
	std::vector<t_def> Z = {-4,  0, -4,  1,
							 3,  0, -2, -3,
							-3, -4, -4,  0};
	// s * s
	std::vector<t_def> L = {0, 0, 0,
						    4, 0, 0,
						    0, 4, 0};
	// m * n
	std::vector<t_def> J = {0, 0, 2, 0,
						    4, 2, 0, 1};
	// m * s
	std::vector<t_def> Y = {0, 0, 2,
							4, 2, 0};
	// n
	std::vector<t_def> dx = {-3, 4, 4, 0};
	// s
	std::vector<t_def> dz_expected = {0, -13, 26};
	// m
	std::vector<t_def> dy_expected = {64, 26};

	if(!test_eval_Singular(&a[0], &b[0], &Z[0], &L[0], &J[0], &Y[0], 
			               &dx[0], m, n, s, &dz_expected[0], &dy_expected[0]))
		return false;


	// n = 5;
	// s = 4;
	// m = 3;
	// a = {0, 4, -3, 10};
	// b = {-8, 11, 7};
	// Z = {-4,  0, -4,  1 ,-1,
	// 	  3,  0, -2, -3, -21,
	// 	 -3, -4, -4, -1,  33,
	// 	 -9,  0, -5,  3,  4};
	// L = {0, 0, 0, 0,
	//      4, 0, 0, 0,
	//      8, 9, 0, 0,
	//      2, 1, 7, 0};
	// J = {0, 0, 2, 1, 3,
	//      4, 2, 0, 1, 2,
	//      1, 3, -2, 1, 8};
	// Y = {0, 0, 2, 1,
	// 	 4, 2, 0, 4,
	// 	 1, 4, 7, 4};


	return true;
}


int main()
{	
	test_eval();
	test_initTss();
	test_initIdentity();
	test_getTriangularInverse();
	return 0;
}
// bool test_initTss_Singular(t_def *h_a, t_def *h_b, 
// 			   			   t_def *h_Z, t_def *h_L, 
// 			   			   t_def *h_J, t_def *h_Y,
// 			   			   t_def *h_dz,
// 			   			   int m, int n, int s,
// 			   			   t_def *h_Tss_expected)
// {
// 	t_def *h_Tss = (t_def *)malloc(s*s*sizeof(t_def));
// 	t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
// 	t_def *d_b; cudaMalloc((void **)&d_b, m*sizeof(t_def));
// 	t_def *d_Z; cudaMalloc((void **)&d_Z, s*n*sizeof(t_def));
// 	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
// 	t_def *d_J; cudaMalloc((void **)&d_J, m*n*sizeof(t_def));
// 	t_def *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(t_def));		
// 	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
// 	t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));

// 	cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_b, h_b,  m*sizeof(t_def), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_Z, h_Z,  s*n*sizeof(t_def), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_J, h_J,  m*n*sizeof(t_def), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_Y, h_Y,  m*s*sizeof(t_def), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);

// 	cublasHandle_t handle;
// 	cublasCreate(&handle);
// 	//  ----------------------------------
// 	int gridsize, blocksize;
// 	cuutils::getGridBlockSize(&gridsize, &blocksize);
// 	absnf::initTss <<<gridsize, blocksize >>>(d_Tss,d_L, d_dz, s, s*s);
// 	cudaMemcpy(h_Tss, d_Tss, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);

// 	// ----------------------------------

// 	cudaFree(d_a); 
// 	cudaFree(d_b);
// 	cudaFree(d_Z);
// 	cudaFree(d_L);
// 	cudaFree(d_J);
// 	cudaFree(d_Y);
// 	cudaFree(d_dz);
// 	cudaFree(d_Tss);
// 	free(h_Tss);
// 	return true;
// }