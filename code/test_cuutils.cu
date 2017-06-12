#include <iostream>
#include "cuutils.h"
#include "utils.hpp"

#define t_def double

struct test_result {
    int tests;
    int failed;
};

void test_vvAdd()
{

}
template <typename T>
int vectors_equals(T* a, T* b, int size, bool verbose=false)
{
	if(verbose)
	{
		std::cout << "COMPARING: " << std::endl;
		utils::printf_vector(a, size);
		utils::printf_vector(b, size);

	}
	for(int i=0; i<size; i++)
	{
		if(a[i] != b[i])
		{	
			std::cout << a[i] << " == " << b[i] << std::endl;
			return false;
		}
	}
	return true;
}

template <typename T>
bool test_singleMakeSignumVector(T *vector, T *expected, int size, bool verbose=false)
{
	bool correct = false;
	// ALLOCATE HOST
	t_def *h_actual = (t_def *) malloc(size*sizeof(t_def));

	// ALLOCATE DEVICE
	t_def *d_vec; cudaMalloc((void **)&d_vec, size*sizeof(t_def));
	t_def *d_actual; cudaMalloc((void **)&d_actual, size*sizeof(t_def));
	
	// HOST -> DEVICE
	cudaMemcpy(d_vec, vector, size*sizeof(t_def), cudaMemcpyHostToDevice);

	// OPERATION
	cuutils::makeSignumVector <<<size, size>>> (d_vec, d_actual, size);

	//DEVICE -> HOST
	cudaMemcpy(h_actual, d_actual, size*sizeof(t_def), cudaMemcpyDeviceToHost);	
	
	//COMPARE
	if (vectors_equals(expected, h_actual, size, verbose))
	{
		correct = true;
	}

	//FREE
	free(h_actual);
	cudaFree(d_vec);
	cudaFree(d_actual);

	return correct;
}
test_result test_makeSignumVector()
{
	std::cout << "test_makeSignumVector()" << std::endl;
	int size=5;
	int err = 0;
	int tests = 0;
	t_def data[8][5] = 
	{
		{1,2,3,4,5},
		{1,1,1,1,1},

		{-1,-2,-3,-4,-5},
		{-1,-1,-1,-1, -1},
		
		{0,0,0,0,0},
		{0,0,0,0,0},

		{-5, 1, -1, 0, 100},
		{-1, 1, -1, 0, 1}
	};
	for(int i=0; i<8; i=i+2)
	{
		t_def *vector = &data[i][0];
		t_def *expected = &data[i+1][0];
		if(!test_singleMakeSignumVector(vector, expected, size, false))
		{
			err++;
			std::cout << "ERROR WITH: makeSignumVector()\n";
		}
		tests++;
	}
	test_result result;
	result.tests = tests;
	result.failed = err;
	return result;
}

int main()
{
	test_result result = test_makeSignumVector();
	
	std::cout << "TESTS: " << result.tests << std::endl;
	std::cout << "FAILED: " << result.failed << std::endl;
}