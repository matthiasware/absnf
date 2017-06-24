#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuutils.h"
#include <vector>
#include "utils.hpp"
#define t_def double

void test()
{
    int m = 4;
    int n = 4;
    int s = 3;
    // m x n
    std::vector<t_def> h_A = {1, 1, 2 ,1,
                                 4, 1, 0, 1,
                                 3, 5, 1, 6,
                                 1, 1, 0, 1};
    utils::rowColConversion(&h_A[0], m, n, true);                                
    // m x s                                 
    std::vector<t_def> h_B = {0, 0, 2,
                            4, 2, 0,
                            2, 1, 3,
                            0, 1, 3};
    utils::rowColConversion(&h_B[0], m, s, true);

    // (m x s) = (m x n) x (m * s)
    // t_def *h_IJY =(t_def *)malloc(m*s*sizeof(t_def));
    t_def *d_A; cudaMalloc((void **)&d_A, m*n*sizeof(t_def));
    t_def *d_B; cudaMalloc((void **)&d_B, m*s*sizeof(t_def));
    t_def *d_TAU; cudaMalloc((void **)&d_TAU, m*sizeof(t_def));
    // t_def *d_IJY; cudaMalloc((void **)&d_IJY, m*s*sizeof(t_def));

    cudaMemcpy(d_A, &h_A[0], m*n*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B[0], m*s*sizeof(t_def), cudaMemcpyHostToDevice);
    // --------------------------------------------------------------
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    // --------------------------------------------------------------
    // CALCULATE WORKING SPACE 
    int lwork = 0;
    cusolverDnDgeqrf_bufferSize(solver_handle,
                                m,n,
                                d_A, m,
                                &lwork);
    t_def *d_work; cudaMalloc((void **)&d_work, sizeof(t_def)*lwork);
    int *devInfo; cudaMalloc((void **)&devInfo, sizeof(int));
    
    // COMPUTE QR factorization
    cusolverDnDgeqrf(solver_handle,
                     m,n,
                     d_A,
                     m,
                     d_TAU,
                     d_work,
                     lwork,
                     devInfo);
    // COMPUTE Q^T B
    cusolverDnDormqr(solver_handle,
                     CUBLAS_SIDE_LEFT,
                     CUBLAS_OP_T,
                     m,
                     s,
                     m,
                     d_A,
                     m,
                     d_TAU,
                     d_B,
                     m,
                     d_work,
                     lwork,
                     devInfo);
    // COMPUTE X = R \ Q^T * B
    double one = 1;
    cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                m,
                s,
                &one,
                d_A,
                m,
                d_B,
                m);
    cuutils::printf_vector(d_B, m*s, "X");
    // --------------------------------------------------------------
    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);
    // --------------------------------------------------------------
    // cudaMemcpy(h_IJY, d_IJY, m*s*sizeof(t_def), cudaMemcpyDeviceToHost);
    // free(h_A);
    cudaFree(d_A);
    cudaFree(d_B);
}

	/** Solves AX=B with QR Decomposition
	
		INPUT:
		@param d_A: device mem (m*m) column major
		@param d_B: device mem (m*s) column major
		OUTPUT:
		@param d_B: X device mem (m*s) column major
	*/
template <typename T>
void solveAXeqB(cublasHandle_t &cublas_handle,
			    cusolverDnHandle_t &solver_handle,
			   T *d_A, T *d_B, int m, int s)
{
    int *d_devInfo; cudaMalloc((void **)&d_devInfo, sizeof(int));
    // scaling factors for householder reflectors
    T *d_TAU; cudaMalloc((void **)&d_TAU, m*sizeof(T));

    // calculate working space
    int h_swork = 0; // working space
    cusolverDnDgeqrf_bufferSize(solver_handle,
                                m,m,
                                d_A, m,
                                &h_swork);
    T *d_work; cudaMalloc((void **)&d_work, sizeof(T)*h_swork);

    // ----------------------------------
    // d_A <-- A=Q*R  d_TAU <--
    // ----------------------------------
    cusolverDnDgeqrf(solver_handle,
                     m,m,
                     d_A,
                     m,
                     d_TAU,
                     d_work,
                     h_swork,
                     d_devInfo);
    // ----------------------------------
    // d_B <- Q^T B
    // ----------------------------------
    cusolverDnDormqr(solver_handle,
                     CUBLAS_SIDE_LEFT,
                     CUBLAS_OP_T,
                     m,
                     s,
                     m,
                     d_A,
                     m,
                     d_TAU,
                     d_B,
                     m,
                     d_work,
                     h_swork,
                     d_devInfo);
    // ----------------------------------
    // d_B <- X = R \ Q^T * B
    // ----------------------------------
    double alpha = 1;
    cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                m,
                s,
                &alpha,
                d_A,
                m,
                d_B,
                m);

    cudaFree(d_devInfo);
    cudaFree(d_TAU);
}

int main()
{
    int m = 4;
    int n = 4;
    int s = 3;
    // m x n
    std::vector<t_def> h_A = {1, 1, 2 ,1,
                                 4, 1, 0, 1,
                                 3, 5, 1, 6,
                                 1, 1, 0, 1};
    utils::rowColConversion(&h_A[0], m, n, true);                                
    // m x s                                 
    std::vector<t_def> h_B = {0, 0, 2,
                            4, 2, 0,
                            2, 1, 3,
                            0, 1, 3};
    utils::rowColConversion(&h_B[0], m, s, true);

    // m
    std::vector<t_def> h_b = {-275, -126, -484, -450};
    // (m x s) = (m x n) x (m * s)
    t_def *d_A; cudaMalloc((void **)&d_A, m*n*sizeof(t_def));
    t_def *d_B; cudaMalloc((void **)&d_B, m*s*sizeof(t_def));
    t_def *d_b; cudaMalloc((void **)&d_b, m*sizeof(t_def));

    cudaMemcpy(d_A, &h_A[0], m*n*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B[0], m*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b[0], m*sizeof(t_def), cudaMemcpyHostToDevice);
    // --------------------------------------------------------------
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    // --------------------------------------------------------------
    // solveAXeqB(cublas_handle, solver_handle, d_A, d_B, m, s);
    // cuutils::printf_vector(d_B, m*s, "X");
    solveAXeqB(cublas_handle, solver_handle, d_A, d_b, m, 1);
    cuutils::printf_vector(d_b, m, "x");
    // --------------------------------------------------------------
    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);
    // --------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_b);
	return 0;
}