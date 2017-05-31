#include <iostream>

// every function that runs on the device and is called from the host
// is a kernel
// a divive function runs at and is called from the device
__global__ void add(int *a, int *b, int *c)
{
	*c = *a + *b;
}

int main(void)
{
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	// allocate memory
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	# setup data
	a = 2;
	b = 7;

	// copy data
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// Launch Kernel
	add <<<1, 1>>>(d_a, d_b, d_c);

	// Copy result back
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	// clean
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}