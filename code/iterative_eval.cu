#include <cublas_v2.h>
#include "utils.hpp"

#define t_def double

int main()
{
	int n=4, s=3;

	t_def *h_a = (t_def *)malloc(s*sizeof(t_def));
	t_def *h_Z = (t_def *)malloc(s*n*sizeof(t_def));
	t_def *h_L = (t_def *)malloc(s*s*sizeof(t_def));
	t_def *h_dx =(t_def *)malloc(n*sizeof(t_def));
	t_def *h_dz =(t_def *)malloc(s*sizeof(t_def));

	utils::fillRandVector(h_a, s, 0, 5, 1, utils::VALUEOP::INT);
	utils::fillRandVector(h_dx, n, 0, 5, 2, utils::VALUEOP::INT);
	utils::fillRandMatrix(h_Z, s, n, 0, 5, 3, utils::MATRIXOPT::NONE, utils::VALUEOP::INT);
	utils::fillRandMatrix(h_L, s, s, 1, 5, 4, utils::MATRIXOPT::LOWER, utils::VALUEOP::INT);


	utils::printf_vector(h_a, s);
	utils::printf_vector(h_dx, n);
	utils::printf_matrix(h_Z, s, n);
	utils::printf_matrix(h_L, s, s);
	free(h_a); free(h_L); free(h_Z);
	free(h_dz); free(h_dx);

	return 0;
}