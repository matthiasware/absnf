#include <iostream>
#include <cstdlib> 
#include <cmath> 


void doS(int *a, int *b)
{
	std::cout << a << std::endl;
	std::cout << *a << std::endl;
	std::cout << b << std::endl;
	std::cout << *b << std::endl;
	std::cout << std::abs(-1) << std::endl;
	*b = abs(*a);
}

int main()
{
	std::cout << "HI" << std::endl;
	int *a = (int *) std::malloc(3*sizeof(int));
	// int *b = (int *) std::malloc(3*sizeof(int));
	// a[0] = -1; a[1] = -2, a[2] = -3;
	// doS(&a[1], &b[2]);
	// std::cout << a[1] << std::endl;
	// std::cout << b[2] << std::endl;
	for(int i=0; i< 3; i++)
	{
		std::cout << a+i << std::endl;
	}
	return 0;
}