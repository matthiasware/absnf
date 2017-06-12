#include <cublas_v2.h>
#include "utils.hpp"

#define t_def double
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void makeSignumVector(t_def *v_source, t_def *v_target, int size)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id < size)
	{
		v_target[id] = v_target[]
	}
}

int main()
{
	return 0;
}