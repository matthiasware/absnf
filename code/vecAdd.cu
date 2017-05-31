#include <iostream>

// every function that runs on the device and is called from the host
// is a kernel
// a divive function runs at and is called from the device
// blockIdx.x vd threadIdx.x
// grid = collection of blocks
// threads belong to groups
// they can synchronize, cooperate, share stuff
// this group is called block
// block = group of threads that cooperate (communicate, synchronize, share data)
// threads of different blocks cannot synchronize, communicate ect. 
__global__ void add(int *a, int *b, int *c, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) // protects us from access beyond end of array
	{
		c[index] = a[index] + b[index];
	}
}

#define N (512*512)
#define M 512

int main(void)
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	// allocate memory
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	// setup data
	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);

	for (int i=0;i<N;i++)
	{
		a[i] = 1;
		b[i] = 2;
		c[i] = 0;
	}

	// copy data
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch Kernel
	// <<< Blocks, Threads >>>
	add <<<(N + M-1)/M, M>>>(d_a, d_b, d_c, N);

	// Copy result back
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Results
	printf("Result: \n");
	for (int i=0; i<10; i++)
	{
		printf("%i ", c[i]);
	}
	printf("\n");
	
	// Clean
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}