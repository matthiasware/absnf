#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuutils.h"
#include <iostream>
#include "absnf.h"
#include "utils.hpp"
#include <vector>
#define t_def double

// template <typename T>
// void modulus_core(cublasHandle_t &cublas_handle,
//                       T *d_S, T *d_c, T *d_abs_dz, int s, T *d_dz)
// {
//     // d_dz = c
//     cudaMemcpy(d_dz, d_c, s*sizeof(T), cudaMemcpyDeviceToDevice);
//     // d_dz = beta * d_dz + S * d_abs_dz
//     double alpha = 1;
//     double beta = 1;
//     cublasDgemv(cublas_handle,
//                 CUBLAS_OP_N,
//                 s, s,
//                 &alpha,
//                 d_S, s,
//                 d_abs_dz, 1,
//                 &beta,
//                 d_dz, 1);
// }

// template <typename T>
// void modulus(cublasHandle_t &cublas_handle, 
//                  T *d_S, T *d_c, T *d_dz,
//                  T *d_abs_dz, T *d_dz_old,
//                  int m, int s,
//                  int blocksize, int gridsize,
//                  int maxiter, double tol, bool verbose)
//     {
//         int i=0;
//         double diff = tol + 1;
//         while(i < maxiter && diff > tol)
//         {
//             // dz_old = dz
//             cuutils::check(cudaMemcpy(d_dz_old, d_dz, s*sizeof(T), cudaMemcpyDeviceToDevice));
//             // abs_dz = |dz|
//             cuutils::abs<<<gridsize,blocksize>>>(d_dz, d_abs_dz, s);
//             // dz = calculateDZ()
//             modulus_core(cublas_handle, d_S, d_c, d_abs_dz, s, d_dz);
//             // calculate diff
//             cuutils::vvSub<<<gridsize, blocksize>>>(d_dz, d_dz_old, d_dz_old, s);
//             cuutils::check(cublasDnrm2(cublas_handle,
//                     s,
//                     d_dz_old,
//                     1,
//                     &diff));
//             if(verbose)
//                 std::cout << i << ": " << diff << std::endl;
//             i++;
//         }
// }
int main()
{
    int seed = 0;
    // int s = 5; // works seed=1
    int s = 200; // works
    t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_b = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_Z = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_J = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_Y = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_dx = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_dy = (t_def *)malloc(s*sizeof(t_def));

    t_def *d_a; cudaMalloc((void **)&d_a, s*sizeof(t_def));
    t_def *d_b; cudaMalloc((void **)&d_b, s*sizeof(t_def));
    t_def *d_Z; cudaMalloc((void **)&d_Z, s*s*sizeof(t_def));
    t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
    t_def *d_J; cudaMalloc((void **)&d_J, s*s*sizeof(t_def));
    t_def *d_Y; cudaMalloc((void **)&d_Y, s*s*sizeof(t_def));       
    t_def *d_dx; cudaMalloc((void **)&d_dx, s*sizeof(t_def));
    t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
    t_def *d_dz_solve; cudaMalloc((void **)&d_dz_solve, s*sizeof(t_def));
    t_def *d_dz_old; cudaMalloc((void **)&d_dz_old, s*sizeof(t_def));
    t_def *d_abs_dz; cudaMalloc((void **)&d_abs_dz, s*sizeof(t_def));
    t_def *d_dy; cudaMalloc((void **)&d_dy, s*sizeof(t_def));

    t_def *d_S; cudaMalloc((void **)&d_S, s*s*sizeof(t_def));
    t_def *d_c; cudaMalloc((void **)&d_c, s*sizeof(t_def));

    utils::fillRandVector(h_a, s,-1,1, seed, utils::VALUEOP::REAL);
    utils::fillRandVector(h_b, s, -1,1, seed, utils::VALUEOP::REAL);
    utils::fillRandVector(h_Z, s*s,-1,1,seed, utils::VALUEOP::REAL);
    utils::fillRandVector(h_Y, s*s,-1,1,seed, utils::VALUEOP::REAL);
    utils::fillRandVector(h_dx, s,-1,1,seed, utils::VALUEOP::REAL);
    utils::fillRandMatrix(h_J, s,s,-1,1,seed, utils::MATRIXOPT::INVERTIBLE, utils::VALUEOP::REAL);
    utils::fillVector(h_L, s*s,0.0);


    utils::printf_vector(h_a, s, "a");
    utils::printf_vector(h_b, s, "b");
    utils::printf_matrix(h_Z, s, s, "Z");
    utils::printf_matrix(h_L, s, s, "L");
    utils::printf_matrix(h_J, s, s, "J");
    utils::printf_matrix(h_Y, s, s, "Y");
    utils::printf_vector(h_dx, s, "dx");

    utils::rowColConversion(h_Z, s, s, true);
    utils::rowColConversion(h_Y, s, s, true);
    utils::rowColConversion(h_J, s, s, true);

    cudaMemcpy(d_a, h_a,  s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b,  s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, h_J,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, h_dx, s*sizeof(t_def), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz_solve, h_a, s*sizeof(t_def), cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    int gridsize, blocksize;
    cuutils::getGridBlockSize(&gridsize, &blocksize);

    // EVALUATE ABSNF
    absnf::eval_core(cublas_handle, d_a, d_b,
                         d_Z, d_L,
                         d_J, d_Y,
                         d_dx,
                         s, s, s,
                         d_dz, d_dy,
                         d_abs_dz);

    cuutils::printf_vector(d_dz, s, "d_dz_eval");
    // SOLVE ABSNF
    // ADJUST b
    cuutils::vvSub<<<gridsize, blocksize>>>(d_b, d_dy, d_b, s);
    cuutils::printf_vector(d_b, s, "d_b");
    
    absnf::calculate_S_and_c(cublas_handle, solver_handle,
                      d_a, d_b, d_Z, d_L, d_J, d_Y, 
                      s, s,
                      d_c, d_S);

    cuutils::printf_vector(d_S, s*s, "d_S");
    cuutils::printf_vector(d_c, s, "d_c");

    absnf::modulus(cublas_handle, d_S, d_c, d_dz_solve,
            d_abs_dz, d_dz_old, s, s,blocksize, gridsize,
            10000, 1e-4, true);

    cuutils::printf_vector(d_dz, s, "d_dz");
    cuutils::printf_vector(d_dz_solve, s, "d_dz_solve");

    free(h_a);
    free(h_b);
    free(h_Z);
    free(h_L);
    free(h_J);
    free(h_Y);
    free(h_dx);

    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_Z);
    cudaFree(d_L);
    cudaFree(d_S);
    cudaFree(d_c);
    cudaFree(d_J);
    cudaFree(d_Y);
    cudaFree(d_dx);
    cudaFree(d_dz);
    cudaFree(d_dz_solve);
    cudaFree(d_dz_old);
    cudaFree(d_abs_dz);
    cudaFree(d_dy);

    cublasDestroy(cublas_handle);   

	return 0;
}