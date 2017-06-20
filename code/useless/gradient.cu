#include <cublas_v2.h>
#include "utils.hpp"
#include "cuutils.h"

#define t_def double
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
// #define BLOCKSIZE 1024

void __global__ initTss(t_def *Tss, t_def *L, t_def *dz, int s, int size)
{
	int j = blockIdx.x;
	int i = threadIdx.x;
	int id = j*s + i;
	while(id < size)
	{
		if(i < s)
		{
			if(i == j)
			{
				Tss[id] = 1;
			}
			else if(j > i)
			{
				Tss[id] = 0;
			}
			else
			{
				Tss[id] = L[id] * (double(0) < dz[j]) - (dz[j] < double(0));
				printf("L[%i, %i],%d\n",i,j,L[id]);
			}
			i += blockDim.x;
		}
		else
		{
			i = threadIdx.x;
			j = j + gridDim.x;
		}
		id = j*s + i;
	}
}
int main()
{
	// int n=3, s=4, m=2;
	// t_def h_L[] = {0,0,0,0,
	// 			   1,0,0,0,
	// 			   2,3,0,0,
	// 			   4,5,6,0};
	// t_def h_dz[] = {-4,0,3,-2};
	// t_def *h_Tss = (t_def *)malloc(s*s*sizeof(t_def));

	// t_def *d_dz; cudaMalloc((void **)&d_dz, s*sizeof(t_def));
	// t_def *d_L; cudaMalloc((void **)&d_L, s*s*sizeof(t_def));
	// t_def *d_Tss; cudaMalloc((void **)&d_Tss, s*s*sizeof(t_def));

	// cudaMemcpy(d_dz, h_dz, s*sizeof(t_def), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_L, h_L,  s*s*sizeof(t_def), cudaMemcpyHostToDevice);

	// int gridsize, blocksize;
	// cuutils::getGridBlockSize(&gridsize, &blocksize);
	// initTss <<<gridsize, blocksize >>>(d_Tss,d_L, d_dz, s, s*s);
	// cudaMemcpy(h_Tss, d_Tss, s*s*sizeof(t_def), cudaMemcpyDeviceToHost);
	// utils::printf_matrix_C2R(h_Tss,s, s, "Tss");

	// free(h_Tss);
	// cudaFree(d_dz);
	// cudaFree(d_L);
	// cudaFree(d_Tss);

	return 0;
}