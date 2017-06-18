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
	// utils::rowColConversion(h_L, s, s, true);
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
	// utils::vectors_equals(h_dz, h_dz_expected, s, true);
	// utils::vectors_equals(h_dy, h_dy_expected, m, true);
	return true;
}
int test_eval()
{
	int errors = 0;
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

	test_eval_Singular(&a[0], &b[0], &Z[0], &L[0], &J[0], &Y[0], 
			           &dx[0], m, n, s, &dz_expected[0], &dy_expected[0]);

	return errors;
}


int main()
{	
	test_eval();
	return 0;
}