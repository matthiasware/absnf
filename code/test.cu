#include <cublas_v2.h>
#include <iostream>
#include <cusolverDn.h>
#include <sstream>
#include "cuutils.h"
#include <vector>
#define t_def double

template <class T>
class Absnf {
public:
	int m_size_m;
	int m_size_n;
	int m_size_s;
	// Host memory
	T *m_h_Z = NULL;
	T *m_h_L = NULL;
	T *m_h_J = NULL;
	T *m_h_Y = NULL;
	T *m_h_a = NULL;
	T *m_h_b = NULL;

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
	void calculate_gradient_workload();
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
	void eval_core(T *d_dx, T *d_dz, T *d_abs_dz, T *d_dy);
	void gradient(T *h_dz, T *h_gamma, T *h_Gamma);
	void gradient_core(T *d_dz, T* d_Tss, T* d_I, T* d_K, T *d_gamma, T *d_Gamma);
	void solve();
	void check(cudaError_t code);
	void check(cublasStatus_t code);
	void check(cusolverStatus_t code);
	void cuda_err_handler(const char *msg, const char *file, int line, bool abort=true);
};
template <class T>
void Absnf<T>::prepare_cublas()
{
	if(m_cublas_handle == NULL)
		check(cublasCreate(&m_cublas_handle));
}
template <class T>
void Absnf<T>::prepare_cusolve()
{
	if(m_solver_handle == NULL)
		check(cusolverDnCreate(&m_solver_handle));
}
template <class T>
void Absnf<T>::check(cudaError_t code)
{
	if(code != cudaSuccess)
		cuda_err_handler(cudaGetErrorString(code), __FILE__, __LINE__);
}
template <class T>
void Absnf<T>::check(cublasStatus_t code)
{
	if(code != CUBLAS_STATUS_SUCCESS)
		cuda_err_handler(cuutils::cublasGetErrorString(code), __FILE__, __LINE__);
}
template <class T>
void Absnf<T>::check(cusolverStatus_t code)
{
	if(code != CUSOLVER_STATUS_SUCCESS)
		cuda_err_handler(cuutils::cusolverGetErrorString(code), __FILE__, __LINE__);
}

template <class T>
void Absnf<T>::cuda_err_handler(const char *msg, const char *file, int line, bool abort)
{

	std::stringstream ss;
	ss << file << "(" << line << ")" << " : " << msg;
	std::string file_and_line;
	ss >> file_and_line;
	if (abort)
		throw std::runtime_error(file_and_line);
	else
		std::cout << "GPUassert: " << file_and_line << std::endl;
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
	check(cudaMemcpy(h_dest, d_source, size*sizeof(T), cudaMemcpyDeviceToHost));
	check(cudaFree(d_source));
}
template <class T>
T* Absnf<T>::cuda_allocate_and_copy(T *h_source, int size)
{
	T *d_p = cuda_allocate(size);
	check(cudaMemcpy(d_p, h_source,  size*sizeof(T), cudaMemcpyHostToDevice));
	return d_p;
}
template <class T>
T* Absnf<T>::cuda_allocate(int size)
{
	T *d_p;
	check(cudaMalloc((void **)&d_p, size*sizeof(T)));
	return d_p;
}
template <class T>
Absnf<T>::~Absnf()
{
	if(m_solver_handle != NULL)
		check(cusolverDnDestroy(m_solver_handle));
	if(m_cublas_handle != NULL)
    	check(cublasDestroy(m_cublas_handle));
	check(cudaThreadExit());
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
void Absnf<T>::eval_core(T *d_dx, T *d_dz, T *d_abs_dz, T *d_dy)
{
	//  ----------------------------------
	// dz = a
	//  ----------------------------------
	check(cudaMemcpy(d_dz, m_d_a, m_size_s*sizeof(T), cudaMemcpyDeviceToDevice));
	//  ----------------------------------		
	// dz = Z * dx + dz
	// dz = alpha * (Z * dx) + beta * dz
	//  ----------------------------------
	double alpha = 1;
	double beta = 1;
	check(cublasDgemv(m_cublas_handle, CUBLAS_OP_N, 
					  m_size_s, m_size_n,
					  &alpha,
					  m_d_Z, m_size_s,
					  d_dx, 1,
					  &beta,
					  d_dz, 1));
	//  ----------------------------------
	// dz = dz + a
	// dz = dz + L * |dz|
	//  ----------------------------------
	for(int i=0; i<m_size_s; i++)
	{
		check(cublasDgemv(m_cublas_handle, CUBLAS_OP_N,
						  1, i,
						  &alpha,
						  &m_d_L[i * m_size_s], 1,
						  d_abs_dz, 1,
						  &beta,
						  &d_dz[i], 1));
		cuutils::abs <<<1,1>>>(&d_dz[i], &d_abs_dz[i], 1);
	}
	//  ----------------------------------
	// dy = b
	//  ----------------------------------
	check(cudaMemcpy(d_dy, m_d_b, m_size_m*sizeof(T), cudaMemcpyDeviceToDevice));
	//  ----------------------------------
	// dy = J * dx
	//  ----------------------------------
	check(cublasDgemv(m_cublas_handle, CUBLAS_OP_N,
					  m_size_m, m_size_n,
					  &alpha,
					  m_d_J, m_size_m,
					  d_dx, 1,
					  &beta, d_dy, 1));
	//  ----------------------------------
	// dy = dy + Y * |dz|
	// dy = beta * dy + alpha(Y*abs_dz)
	//  ----------------------------------
	check(cublasDgemv(m_cublas_handle, CUBLAS_OP_N,
					  m_size_m, m_size_s,
					  &alpha,
					  m_d_Y, m_size_m,
					  d_abs_dz, 1,
					  &beta,
					  d_dy, 1));
}

template <class T>
void Absnf<T>::eval(T *h_dx, T *h_dz, T *h_dy)
{
	prepare_memory();
	prepare_cublas();
	T *d_dy = cuda_allocate(m_size_m);
	T *d_dz = cuda_allocate(m_size_s);
	T *d_abs_dz = cuda_allocate(m_size_s);
	T *d_dx = cuda_allocate_and_copy(h_dx, m_size_n);

	eval_core(d_dx, d_dz, d_abs_dz, d_dy);

	cuda_download_and_free(h_dy, d_dy, m_size_m);
	cuda_download_and_free(h_dz, d_dz, m_size_s);

	check(cudaFree(d_dx));
	check(cudaFree(d_abs_dz));
}
template <class T>
void Absnf<T>::gradient_core(T *d_dz, T* d_Tss, T* d_I, T* d_K, T *d_gamma, T *d_Gamma)
{

}
template <class T>
void Absnf<T>::gradient(T *h_dz, T *h_gamma, T *h_Gamma)
{
	prepare_memory();
	prepare_cublas();
	T *d_dz = cuda_allocate_and_copy(h_dz, m_size_s);
	T *d_gamma = cuda_allocate(m_size_m);
	T *d_Gamma = cuda_allocate(m_size_m * m_size_n);
	T *d_Tss = cuda_allocate(m_size_s * m_size_s);
	T *d_I = cuda_allocate(m_size_s * m_size_s);
	T *d_K = cuda_allocate(m_size_m * m_size_s);

	gradient_core(d_dz, d_Tss, d_I, d_K, d_gamma, d_Gamma);

	cuda_download_and_free(h_gamma, d_gamma, m_size_m);
	cuda_download_and_free(h_Gamma, d_Gamma, m_size_m * m_size_n);
	check(cudaFree(d_dz));
	check(cudaFree(d_Tss));
	check(cudaFree(d_I));
	check(cudaFree(d_K));
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
 //    	absnf.check(cudaSetDevice(-1));
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