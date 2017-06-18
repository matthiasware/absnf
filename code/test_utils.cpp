#include "utils.hpp"
#include <iostream>
#include <string.h>
#include <vector>

template <typename T>
bool test_rowColConversion_Singular(T *rm, T *cm, int rows, int cols)
{
	int size=rows*cols;
	T a[size];
	memcpy(a, rm, size*sizeof(T));
	utils::rowColConversion(a, rows, cols, true);
	if(!utils::vectors_equals(a, cm, size, false))
		return false;
	utils::rowColConversion(a, rows, cols, false);
	if(!utils::vectors_equals(a, rm, size, false))
		return false;
	return true;

}
int test_rowColConversion()
{
	int errors=0;
	std::vector<int> a_rm = {0};
	std::vector<int> a_cm = {0};
	if(!test_rowColConversion_Singular(&a_rm[0], &a_cm[0], 1, 1))
		errors++;

	a_rm = {1,2};
	a_cm = {1,2};
	if(!test_rowColConversion_Singular(&a_rm[0], &a_cm[0], 1, 2))
		errors++;

	a_rm = {1,2,3,4};
	a_cm = {1,3,2,4};
	if(!test_rowColConversion_Singular(&a_rm[0], &a_cm[0], 2, 2))
		errors++;
	
	a_rm = {1,2,3,4,5,6,7,8,9,10,11,12};
	a_cm = {1,7,2,8,3,9,4,10,5,11,6,12};
	if(!test_rowColConversion_Singular(&a_rm[0], &a_cm[0], 2, 6))
		errors++;
	return errors;
}

int main()
{
	// test_rowColConversion();
	int errors = 0;
	errors += test_rowColConversion();
	std::cout << "utils::test_rowColConversion()" << std::endl;
	std::cout << "failed: " << errors << std::endl;
	return 0;
}