#include <iostream>
#include <random>
namespace utils
{
	void pp()
	{
		std::cout << "UTILS" << std::endl;
	}
	void printf_matrix(double *A, int cols, int rows)
	{
	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<cols; j++)
		{
			int idx = i*cols + j;
			printf(" %f", A[idx]);
		}
		printf("\n");
	}
}
void printf_vector(double *A, int cols)
{
	for(int i=0; i<cols; i++)
	{
		printf(" %f", A[i]);
	}
	printf("\n");
}

}