#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuutils.h"
#include "absnf.h"
#include "utils.hpp"
#include <vector>
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
void test2()
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
    // solveAXeqB(cublas_handle, solver_handle, d_A, d_b, m, 1);
    // cuutils::printf_vector(d_b, m, "x");
    // --------------------------------------------------------------
    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);
    // --------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_b);
}

int main()
{
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
    // b = b - y IMPORTANT
    std::vector<t_def> h_b = {-223, -432, -200, -48};

    // s
    std::vector<t_def> h_a = {4, 4, -3};

    // dz_start
    // t_def *h_dz = (t_def *) malloc(s * sizeof(t_def));
    // utils::fillRandVector(h_dz, s, -10, 10, 4);
    t_def h_dz[] = {-1.59449432,  9.28890523,  9.39411967};

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
    // Calculate S, c
    // calculate_S(cublas_handle, solver_handle,
    //             d_L, d_Z, d_J, d_Y, m, s, d_S);
    
    // cudaMemcpy(d_J, &h_J[0], m*m*sizeof(t_def), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Y, &h_Y[0], m*s*sizeof(t_def), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Z, &h_Z[0], s*m*sizeof(t_def), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, &h_b[0], m*sizeof(t_def), cudaMemcpyHostToDevice);

    // calculate_c(cublas_handle, solver_handle,
    //             d_a, d_Z, d_J, d_b, m, s, d_c);

    absnf::calculate_S_and_c(cublas_handle, solver_handle,
                      d_a, d_b, d_Z, d_L, d_J, d_Y, 
                      m, s,
                      d_c, d_S); 

    cuutils::printf_vector(d_S, s*s, "d_S");
    cuutils::printf_vector(d_c, s, "d_c");
    //--------------------------------------------------------------
    cuutils::printf_vector(d_dz, s, "dz_start");

    int gridsize, blocksize;
    cuutils::getGridBlockSize(&gridsize, &blocksize);
    std::cout << gridsize << " : " << blocksize << std::endl;
    // int maxiter = 100;
    // int i = 0;
    // double tol = 1e-8;
    // double diff = tol + 1;
    // // cuutils::printf_vector(d_dz, s, "d_dz");
    // while(i < maxiter && diff > tol)
    // {
    //     // dz_old = dz
    //     cudaMemcpy(d_dz_old, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToDevice);
    //     // abs_dz = |dz|
    //     cuutils::abs<<<gridsize,blocksize>>>(d_dz, d_abs_dz, s);
    //     std::cout << "----" << i << "----"  << std::endl;
    //     cuutils::printf_vector(d_dz_old, s, "d_dz_old");
    //     cuutils::printf_vector(d_abs_dz, s, "d_abs_dz");
    //     // dz = calculateDZ()
    //     modulus(cublas_handle, d_S, d_c, d_abs_dz, s, d_dz);
    //     cuutils::printf_vector(d_dz, s, "d_dz_new");
    //     // calculate diff
    //     cuutils::vvSub<<<gridsize, blocksize>>>(d_dz, d_dz_old, d_dz_old, s);
    //     cublasDnrm2(cublas_handle,
    //                 s,
    //                 d_dz_old,
    //                 1,
    //                 &diff);
    //     // std::cout << i << ": " << diff << std::endl;
    //     i++;
    // }
    absnf::modulus(cublas_handle, d_S, d_c, d_dz,
                   d_abs_dz, d_dz_old, m, s,blocksize, gridsize);
    cuutils::printf_vector(d_dz, s, "Result: d_dz"); 
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
	return 0;
}