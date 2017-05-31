#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "utils.hpp"

#define t_def double

int main()
{
	int m = 2;
	int n = 2;
	t_def *v =  (t_def *) malloc(m*sizeof(t_def));
	t_def *M =  (t_def *) malloc(m*n*sizeof(t_def));
	utils::fillRandVector<t_def>(v, m, 0, 10, 1);
	utils::fillRandMatrix<t_def>(M, m, n, 0, 10, 1, utils::MATRIXOPT::LOWER);
	utils::printf_vector(v, m);
	utils::printf_matrix(M, m, n);
	free(v);
	free(M);
	return 0;
}