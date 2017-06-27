#include <cublas_v2.h>
#include <iostream>
#include <cusolverDn.h>
#include <sstream>
#include "cuutils.h"
#include "absnf.h"
#include <vector>
#define t_def double

template <class T>
class CudaData
{
public:
	int size;
};

template <class T>
class CudaMatrix
{
public:
	enum MemOrder {ROW_MAJOR, COL_MAJOR};
	enum Format {NORM, LU, QR};
	Format m_form_device;
	MemOrder m_mem_device;
	int m_rows;
	int m_cols;
	int m_elements;
	int m_size;
	T *m_d_mem;
	T *m_h_mem;
	// functions
	T *d_mem();
	T *h_mem();
	T *h_buffer();
	T *h_T();
	T *d_T();
	void upload();
	void download();
	void buffer();
};
template <class T>
class CudaVector
{

};

template <class T>
class Absnf {
public:
	int m_size_m;
	int m_size_n;
	int m_size_s;
	int m_blocksize;
	int m_gridsize;
	// Host memory
	T *m_h_Z = NULL;
	T *m_h_L = NULL;
	T *m_h_J = NULL;
	T *m_h_Y = NULL;
	T *m_h_a = NULL;
	T *m_h_b = NULL;
	// Device Memory
	T *m_d_Z = NULL;
	T *m_d_L = NULL;
	T *m_d_J = NULL;
	T *m_d_Y = NULL;
	T *m_d_a = NULL;
	T *m_d_b = NULL;

	cublasHandle_t m_cublas_handle = NULL;
	cusolverDnHandle_t m_solver_handle = NULL;

	void prepare_cublas();
	void prepare_cusolve();
	void prepare_device();
	void calculate_gradient_workload(int m, int n, int s, int &workload, bool max);
	void calculate_solve_workload();
	T * cuda_allocate(int size);
	T * cuda_allocate_and_copy(T *h_source, int size);
	void cuda_download_and_free(T* h_dest, T *d_source, int size);
	// void setDevice();
	Absnf(){};
	Absnf(T *Z, T *L, T *J, T *Y, T *a, T *b, int m, int n, int s) 
		:m_h_Z(Z), m_h_L(L), m_h_J(J), m_h_Y(Y),  m_h_a(a), m_h_b(b), m_size_m(m), m_size_n(n), m_size_s(s)
	{};
	~Absnf();
	void calculate_eval_workload(int m, int n, int s, int &workload, bool max);
	void prepare_memory();
	void eval(T *h_dx, T *h_dz, T *h_dy);
	void gradient(T *h_dz, T *h_gamma, T *h_Gamma);
	void solve();
};
template <class T>
void Absnf<T>::prepare_device()
{
	cuutils::getGridBlockSize(&m_gridsize, &m_blocksize);
}
template <class T>
void Absnf<T>::prepare_cublas()
{
	if(m_cublas_handle == NULL)
		cuutils::check(cublasCreate(&m_cublas_handle));
}
template <class T>
void Absnf<T>::prepare_cusolve()
{
	if(m_solver_handle == NULL)
		cuutils::check(cusolverDnCreate(&m_solver_handle));
}

template <class T>
void Absnf<T>::prepare_memory()
{
	if(m_h_Z && m_d_Z == NULL)
		m_d_Z = cuda_allocate_and_copy(m_h_Z, m_size_s*m_size_n);
	else
		throw "cannot allocate device memory for Z";
	if(m_h_L && m_d_L == NULL)
		m_d_L = cuda_allocate_and_copy(m_h_L, m_size_s*m_size_s);
	else
		throw "cannot allocate device memory for L";
	if(m_h_J && m_d_J == NULL)
		m_d_J = cuda_allocate_and_copy(m_h_J, m_size_m*m_size_n);
	else
		throw "cannot allocate device memory for J";
	if(m_h_Y && m_d_Y == NULL)
		m_d_Y = cuda_allocate_and_copy(m_h_Y, m_size_m*m_size_s);
	else
		throw "cannot allocate device memory for Y";
	if(m_h_a && m_d_a == NULL)
		m_d_a = cuda_allocate_and_copy(m_h_a, m_size_s);
	else
		throw "cannot allocate device memory for a";
	if(m_h_b && m_d_b == NULL)
		m_d_b = cuda_allocate_and_copy(m_h_b, m_size_m);
	else
		throw "cannot allocate device memory for b";
}
template <class T>
void Absnf<T>::cuda_download_and_free(T* h_dest, T* d_source, int size)
{
	cuutils::check(cudaMemcpy(h_dest, d_source, size*sizeof(T), cudaMemcpyDeviceToHost));
	cuutils::check(cudaFree(d_source));
}
template <class T>
T* Absnf<T>::cuda_allocate_and_copy(T *h_source, int size)
{
	T *d_p = cuda_allocate(size);
	cuutils::check(cudaMemcpy(d_p, h_source,  size*sizeof(T), cudaMemcpyHostToDevice));
	return d_p;
}
template <class T>
T* Absnf<T>::cuda_allocate(int size)
{
	T *d_p;
	cuutils::check(cudaMalloc((void **)&d_p, size*sizeof(T)));
	return d_p;
}
template <class T>
Absnf<T>::~Absnf()
{
	if(m_solver_handle != NULL)
		cuutils::check(cusolverDnDestroy(m_solver_handle));
	if(m_cublas_handle != NULL)
    	cuutils::check(cublasDestroy(m_cublas_handle));
	cuutils::check(cudaThreadExit());
}

template <class T>
void Absnf<T>::calculate_eval_workload(int m, int n, int s,
									   int &workload, bool max=true)
{
	if(max)
		workload = 4*s+2*m+n+(s*n)+(s*s)+(m*n)+(m*s)*sizeof(T);
	else
		throw "not implemented";

};
template <class T>
void Absnf<T>::calculate_gradient_workload(int m, int n, int s,
										   int &workload, bool max=true)
{
	if(max)
		workload = s+m+(s*n)+(s*s)+(m*n)+(m*s)+n+s+(s*s)+(s*s)+(m*s)+m+(m*n)*sizeof(T);
	else
		throw "not implemented";
}

template <class T>
void Absnf<T>::eval(T *h_dx, T *h_dz, T *h_dy)
{
	prepare_memory();
	prepare_cublas();
	prepare_device();
	T *d_dy = cuda_allocate(m_size_m);
	T *d_dz = cuda_allocate(m_size_s);
	T *d_abs_dz = cuda_allocate(m_size_s);
	T *d_dx = cuda_allocate_and_copy(h_dx, m_size_n);

	absnf::eval_core(m_cublas_handle,
					 m_d_a, m_d_b, m_d_Z, m_d_L, m_d_J, m_d_Y,
					 d_dx, m_size_m, m_size_n, m_size_s,
					 d_dz, d_dy, d_abs_dz);

	cuda_download_and_free(h_dy, d_dy, m_size_m);
	cuda_download_and_free(h_dz, d_dz, m_size_s);

	cuutils::check(cudaFree(d_dx));
	cuutils::check(cudaFree(d_abs_dz));
}
template <class T>
void Absnf<T>::gradient(T *h_dz, T *h_gamma, T *h_Gamma)
{
	prepare_memory();
	prepare_cublas();
	prepare_device();

	T *d_dz = cuda_allocate_and_copy(h_dz, m_size_s);
	T *d_gamma = cuda_allocate(m_size_m);
	T *d_Gamma = cuda_allocate(m_size_m * m_size_n);
	T *d_Tss = cuda_allocate(m_size_s * m_size_s);
	T *d_I = cuda_allocate(m_size_s * m_size_s);
	T *d_K = cuda_allocate(m_size_m * m_size_s);

	absnf::gradient_core(m_cublas_handle,
						 m_d_a, m_d_b, m_d_Z, m_d_L, m_d_J, m_d_Y,
						 d_dz, d_Tss, d_I, d_K,
						 m_size_m, m_size_n, m_size_s,
						 m_gridsize, m_blocksize,
						 d_gamma, d_Gamma);

	cuda_download_and_free(h_gamma, d_gamma, m_size_m);
	cuda_download_and_free(h_Gamma, d_Gamma, m_size_m * m_size_n);
	cuutils::check(cudaFree(d_dz));
	cuutils::check(cudaFree(d_Tss));
	cuutils::check(cudaFree(d_I));
	cuutils::check(cudaFree(d_K));
}
int main()
{
	int n=4;
	int s=3;
	int m=2;
	// n
	std::vector<t_def> h_a = {4, 4,-3};
	// m
	std::vector<t_def> h_b = {4, 4};
	// s * n
	std::vector<t_def> h_Z = {-4,  0, -4,  1,
							 3,  0, -2, -3,
							-3, -4, -4,  0};
	// s * s
	std::vector<t_def> h_L = {0, 0, 0,
						    4, 0, 0,
						    0, 4, 0};
	// m * n
	std::vector<t_def> h_J = {0, 0, 2, 0,
						    4, 2, 0, 1};
	// m * s
	std::vector<t_def> h_Y = {0, 0, 2,
							4, 2, 0};

	utils::rowColConversion(&h_Z[0], s, n, true);
	utils::rowColConversion(&h_J[0], m, n, true);
	utils::rowColConversion(&h_Y[0], m, s, true);

	t_def *h_dz =(t_def *)malloc(s*sizeof(t_def));
	t_def *h_dy = (t_def *)malloc(m*sizeof(t_def));
	std::vector<t_def> h_dx = {-3, 4, 4, 0};

	Absnf<t_def> absnf(&h_Z[0], &h_L[0], &h_J[0], &h_Y[0], &h_a[0], &h_b[0], m, n, s);
	absnf.eval(&h_dx[0], h_dz, h_dy);
	utils::printf_vector(h_dy, m, "dy");
	utils::printf_vector(h_dz, s, "dz");
	// absnf.eval(&h_dx[0], h_dy, h_dz);

	// try
 //  	{
 //    // do something crazy
 //    	absnf.cuutils::check(cudaSetDevice(-1));
 //  	}
 //  	catch(std::runtime_error &e)
 //  	{
 //    	std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
 //    // oops, recover
 //    cudaSetDevice(0);
 //  	}
	free(h_dz);
	free(h_dy);
	return 0;
}