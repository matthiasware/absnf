#include <cublas_v2.h>
#include "absnf.h"
#include "utils.hpp"
#define t_def double

void test_eval(t_def *h_a, t_def *h_b, 
			   t_def *h_Z, t_def *h_L, 
			   t_def *h_J, t_def *h_Y,
			   t_def *h_dx,
			   int m, int n, int s,
			   t_def *h_dz_expected, t_def *h_dy_expected)
{
	// convert to column major storage
	t_def *h_cm_Z = (t_def *)malloc(s*n*sizeof(t_def));
	t_def *h_cm_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_cm_J = (t_def *)malloc(m*n*sizeof(t_def));
	t_def *h_cm_Y = (t_def *)malloc(m*s*sizeof(t_def));
	utils::rowColConversion(h_Z, h_cm_Z, s, n);
	utils::rowColConversion(h_L, h_cm_L, s, s);
	utils::rowColConversion(h_J, h_cm_J, m, n);
	utils::rowColConversion(h_Y, h_cm_Y, m, s);
	// results
	t_def *h_dz =(t_def *)malloc(s*sizeof(t_def));
	t_def *h_dy = (t_def *)malloc(m*sizeof(t_def));

	absnf::eval(h_a, h_b, h_cm_Z, h_cm_L, h_cm_J, h_cm_Y, h_dx, m, n, s, h_dz, h_dy);

	// calculate results
	// compare results
}

int main()
{	
	// int a[] = {1,2,3,4,5,6};
	// utils::printf_vector(a, 6, "a");
	// utils::rowColConversion(a, 2, 3);
	// utils::printf_vector(a, 6, "a");
	// utils::rowColConversion(a, 2, 3, false);
	// utils::printf_vector(a, 6, "a");

	int rows = 2;
	int cols = 6;
	int size = rows*cols;
	int a[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int a_cm[] = {1,7,2,8,3,9,4,10,5,11,6,12};
    int a_rm[] = {1,2,3,4,5,6,7,8,9,10,11,12};
	// utils::printf_vector(a, size, "a");
	utils::rowColConversion(a, rows, cols);
	// utils::printf_vector(a, size, "a");
	utils::rowColConversion(a, rows, cols, false);
	utils::printf_vector(a, size, "a");
	return 0;
}