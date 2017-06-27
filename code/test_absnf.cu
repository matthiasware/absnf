#include <cublas_v2.h>
#include <cusolverDn.h>
#include "absnf.h"
#include "cuutils.h"
#include "utils.hpp"
#include "assert.h"
#include <vector>
#define t_def double

bool test_eval_Singular(t_def *h_a, t_def *h_b, 
			   			t_def *h_Z, t_def *h_L, 
			   			t_def *h_J, t_def *h_Y,
			   			t_def *h_dx,
			   			int m, int n, int s,
			   			t_def *h_dz_expected, t_def *h_dy_expected)
{
	bool success = true;
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
		success = false;
	if(!utils::vectors_equals(h_dy, h_dy_expected, m, false))
		success = false;
	
	free(h_dz);
	free(h_dy);
	return success;
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
// bool test_multWithDz()
// {
// 	int s=4;
// 	std::vector<t_def> L = {1, 0, 0, 0,
// 						    4, 1, 0, 0,
// 						    8, 9, 1, 0,
// 						    2, 1, 7, 1};
// 	std::vector<t_def> dz = {-1, 0, 1, -1};
// 	std::vector<t_def> R = {-1, 0, 0, 0,
// 									   -4, 0, 0, 0,
// 									   -8, 0, 1, 0,
// 									   -2, 0, 7, -1};
// 	t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
// 	t_def *d_R; cudaMalloc((void **)&d_R, s*s*sizeof(t_def));
// 	t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));

// }

bool test_eval()
{
	bool success = true;
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
		success = false;
	return success;
}
bool test_gradient_Singular(t_def *h_a, t_def *h_b, 
			   				t_def *h_Z, t_def *h_L, 
			   				t_def *h_J, t_def *h_Y,
			   				t_def *h_dz,
			   				int m, int n, int s,
			   				t_def *h_gamma_expected,
			   				t_def *h_Gamma_expected)
{
	bool success = true;
	// convert to column major storage
	utils::rowColConversion(h_Z, s, n, true);
	utils::rowColConversion(h_J, m, n, true);
	utils::rowColConversion(h_Y, m, s, true);

	// results
	t_def *h_gamma_actual =(t_def *)malloc(m*sizeof(t_def));
	t_def *h_Gamma_actual = (t_def *)malloc(m*n*sizeof(t_def));

	// calculate results
	absnf::gradient(h_a, h_b, h_Z, h_L, h_J, h_Y, h_dz,
					m, n, s, h_gamma_actual, h_Gamma_actual);

	// convert back from cm to rm
	utils::rowColConversion(h_Gamma_actual, m, n, false);

	if(!utils::vectors_equals(h_gamma_actual, h_gamma_expected, m, false))
		success = false;
	if(!utils::vectors_equals(h_Gamma_actual, h_Gamma_expected, m*n, false))
		success = false;

	return success;
}
bool test_gradient()
{
	bool success = true;
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
	// s
	std::vector<t_def> dz = {0, -13, 26};

	std::vector<t_def> h_gamma_expected = {30, -4};
	std::vector<t_def> h_Gamma_expected = { 18, -8, -22, -24,
											-2,  2,   4,   7};

	test_gradient_Singular(&a[0], &b[0], &Z[0], &L[0], &J[0],
						   &Y[0], &dz[0],
						   m,n,s,
						   &h_gamma_expected[0],
						   &h_Gamma_expected[0]);

	return success;
}
bool test_mmp()
{
	bool success = true;
	int m=2;
	int s=3;
	std::vector<t_def> Y = {0, 0, 2,
							4, 2, 0};
	utils::rowColConversion(&Y[0], m, s, true);							
	std::vector<t_def> I = {0, 0, 0,
						    0, -1, 0,
						    0, 4, 1};
	// utils::rowColConversion(&I[0], s, s, true);		

	t_def *K = (t_def *)malloc(m*s*sizeof(t_def));
	t_def *d_K; cudaMalloc((void **)&d_K, m*s*sizeof(t_def));
	t_def *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(t_def));
	t_def *d_I; cudaMalloc((void **)&d_I, s*s*sizeof(t_def));

	cudaMemcpy(d_Y, &Y[0],  m*s*sizeof(t_def), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, &I[0],  s*s*sizeof(t_def), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	// ----------------------------------
	absnf::mmp(handle, d_Y, d_I, d_K, m, s);
	// ----------------------------------
	cudaMemcpy(K, d_K, m*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	utils::printf_vector(K, m*s, "K");

	cudaFree(d_K);
	cudaFree(d_Y);
	cudaFree(d_I);
	free(K);

	return success;
}
bool test_calculate_S_and_c()
{
	bool success = true;

	cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

	int m = 4;
    int s = 3;
    // m x m
    std::vector<t_def> h_J = {100, 1, 2 ,1,
                              4, 120, 0, 1,
                              3, 5, 120, 6,
                              1, 1, 0, 130};
    utils::rowColConversion(&h_J[0], m, m, true);                                
    // m x s                                 
    std::vector<t_def> h_Y = {0, 0, 2,
                              4, 2, 0,
                              2, 1, 3,
                              0, 1, 3};
    utils::rowColConversion(&h_Y[0], m, s, true);

    // s x s
    std::vector<t_def> h_L = {0,0,0,
                              1,0,0,
                              0,1,0};
    utils::rowColConversion(&h_L[0], s, s, true);

    // s x m
    std::vector<t_def> h_Z = {-4,  0, -4 , 1,
                               3,  0, -2, -3,
                              -3, -4, -4,  0};
    utils::rowColConversion(&h_Z[0], s, m, true);

    // m
    // std::vector<t_def> h_b = {-275, -126, -484, -450};
    // b = b - y IMPORTANT !!!
    std::vector<t_def> h_b = {-223, -432, -200, -48};

    // s
    std::vector<t_def> h_a = {4, 4, -3};

    t_def h_S_expected[] = {0.0589243, 1.03177,0.192736,
    						0.0199725,0.0384088,1.09439,
    						0.14793,0.0576823,0.148214};
    t_def h_c_expected[] = {-10.1223,6.61218,-29.3861};

    // (m x s) = (m x n) x (m * s)
    t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
    t_def *d_b; cudaMalloc((void **)&d_b, m*sizeof(t_def));
    t_def *d_Z; cudaMalloc((void **)&d_Z, s*m*sizeof(t_def));
    t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
    t_def *d_J; cudaMalloc((void **)&d_J, m*m*sizeof(t_def));
    t_def *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(t_def));
    t_def *d_S; cudaMalloc((void **)&d_S, s*s*sizeof(t_def));
    t_def *d_c; cudaMalloc((void **)&d_c, s*sizeof(t_def));
    t_def *h_c_actual = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_S_actual = (t_def *)malloc(s*s*sizeof(t_def));

    cudaMemcpy(d_a, &h_a[0], s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b[0], m*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, &h_Z[0], s*m*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, &h_L[0], s*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, &h_J[0], m*m*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, &h_Y[0], m*s*sizeof(t_def), cudaMemcpyHostToDevice);

    absnf::calculate_S_and_c(cublas_handle, solver_handle,
                      		 d_a, d_b, d_Z, d_L, d_J, d_Y, 
                      		 m, s,
                      		 d_c, d_S);

    // std::cout << "HERE \n";
    cudaMemcpy(h_c_actual, d_c, s*sizeof(t_def), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S_actual, d_S, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);

    if(!utils::vectors_almost_equal(h_c_actual, h_c_expected, s, 1e-4, false))
    	success = false;
    if(!utils::vectors_almost_equal(h_S_actual, h_S_expected, s*s, 1e-4, false))
    	success = false;

    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);

    free(h_c_actual);
    free(h_S_actual);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_Z);
    cudaFree(d_L);
    cudaFree(d_J);
    cudaFree(d_Y);
    cudaFree(d_S);
    cudaFree(d_c);

	return success;
}
bool test_modulus()
{
	bool success = true;
	
	int m = 4;
    int s = 3;
    // m x m
    std::vector<t_def> h_J = {100, 1, 2 ,1,
                              4, 120, 0, 1,
                              3, 5, 120, 6,
                              1, 1, 0, 130};
    utils::rowColConversion(&h_J[0], m, m, true);                                
    // m x s                                 
    std::vector<t_def> h_Y = {0, 0, 2,
                              4, 2, 0,
                              2, 1, 3,
                              0, 1, 3};
    utils::rowColConversion(&h_Y[0], m, s, true);

    // s x s
    std::vector<t_def> h_L = {0,0,0,
                            1,0,0,
                            0,1,0};
    utils::rowColConversion(&h_L[0], s, s, true);

    // s x m
    std::vector<t_def> h_Z = {-4,  0, -4 , 1,
                               3,  0, -2, -3,
                              -3, -4, -4,  0};
    utils::rowColConversion(&h_Z[0], s, m, true);

    // m
    // b = b - y IMPORTANT
    std::vector<t_def> h_b = {-223, -432, -200, -48};

    // s
    std::vector<t_def> h_a = {4, 4, -3};

    // dz_start
    t_def h_dz[] = {-1.59449432,  9.28890523,  9.39411967};
    t_def h_dz_expected[] = {-8,  16,  -9};

    // (m x s) = (m x n) x (m * s)
    t_def *d_J; cudaMalloc((void **)&d_J, m*m*sizeof(t_def));
    t_def *d_Y; cudaMalloc((void **)&d_Y, m*s*sizeof(t_def));
    t_def *d_Z; cudaMalloc((void **)&d_Z, s*m*sizeof(t_def));
    t_def *d_b; cudaMalloc((void **)&d_b, m*sizeof(t_def));
    t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
    t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
    t_def *d_S; cudaMalloc((void **)&d_S, s*s*sizeof(t_def));
    t_def *d_c; cudaMalloc((void **)&d_c, s*sizeof(t_def));
    t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
    t_def *d_abs_dz; cudaMalloc((void **)&d_abs_dz, s*sizeof(t_def));
    t_def *d_dz_old; cudaMalloc((void **)&d_dz_old, s*sizeof(t_def));
    t_def *d_dx; cudaMalloc((void **)&d_dx, m*sizeof(t_def));

    cudaMemcpy(d_J, &h_J[0], m*m*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, &h_Y[0], m*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, &h_Z[0], s*m*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b[0], m*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, &h_a[0], s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, &h_L[0], s*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);

    // --------------------------------------------------------------
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    absnf::calculate_S_and_c(cublas_handle, solver_handle,
                      d_a, d_b, d_Z, d_L, d_J, d_Y, 
                      m, s,
                      d_c, d_S); 

    int gridsize, blocksize;
    cuutils::getGridBlockSize(&gridsize, &blocksize);

    absnf::modulus(cublas_handle, d_S, d_c, d_dz,
                   d_abs_dz, d_dz_old, m, s, blocksize, gridsize,
                   100, 1e-8, true);

    cudaMemcpy(h_dz, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);

    if(!utils::vectors_almost_equal(h_dz, h_dz_expected, s, 1e-4, false))
    	success = false;
    
    // --------------------------------------------------------------
    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);
    // --------------------------------------------------------------
    cudaFree(d_J);
    cudaFree(d_Y);
    cudaFree(d_Z);
    cudaFree(d_b);
    cudaFree(d_a);
    cudaFree(d_L);
    cudaFree(d_S);
    cudaFree(d_c);
    cudaFree(d_dx);
    cudaFree(d_dz);
    cudaFree(d_abs_dz);
    cudaFree(d_dz_old);

	return success;
}

int main()
{	
	assert(test_eval());
	assert(test_initTss());
	assert(test_initIdentity());
	assert(test_getTriangularInverse());
	assert(test_gradient());
	assert(test_calculate_S_and_c());
	assert(test_modulus());
	// test_modulus();
	return 0;
}