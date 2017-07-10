#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuutils.h"
#include <iostream>
#include "absnf.h"
#include "utils.hpp"
#include <vector>
#include <chrono>
#define t_def double

typedef std::chrono::high_resolution_clock::time_point TimeVar;

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
void modulus_singular(int s, int maxiter, int seed, double tolerance)
{
    t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_b = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_Z = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_J = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_Y = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_dx = (t_def *)malloc(s*s*sizeof(t_def));
    t_def *h_dz = (t_def *)malloc(s*sizeof(t_def));
    t_def *h_dz_solve = (t_def *)malloc(s*sizeof(t_def));
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

    // SOLVE ABSNF
    // ADJUST b
    cuutils::vvSub<<<gridsize, blocksize>>>(d_b, d_dy, d_b, s);
    
    TimeVar t_0 = std::chrono::high_resolution_clock::now();
    absnf::calculate_S_and_c(cublas_handle, solver_handle,
                      d_a, d_b, d_Z, d_L, d_J, d_Y, 
                      s, s,
                      d_c, d_S);


    absnf::modulus(cublas_handle, d_S, d_c, d_dz_solve,
            d_abs_dz, d_dz_old, s, s,blocksize, gridsize,
            maxiter, tolerance, false);
    cudaDeviceSynchronize();
    TimeVar t_1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_dz_solve, d_dz_solve, s*sizeof(t_def), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dz, d_dz, s*sizeof(t_def), cudaMemcpyDeviceToHost);
    if(!utils::vectors_almost_equal(h_dz, h_dz_solve, s, 1e-4, false))
        throw "ERROR";

    auto int_time = std::chrono::duration_cast<std::chrono::milliseconds>( t_1 - t_0 ).count();
    std::cout << s << ": " << int_time << std::endl;
    free(h_a);
    free(h_b);
    free(h_Z);
    free(h_L);
    free(h_J);
    free(h_Y);
    free(h_dx);
    free(h_dz_solve);

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
}
int main()
{
    int seed = 2;
    // int s = 5; // works seed=1
    int maxiter = 1000;
    double tolerance = 1e-5;
    for(int i = 100; i<= 1200; i+=100)
        modulus_singular(i,maxiter,seed, tolerance);



	return 0;
}