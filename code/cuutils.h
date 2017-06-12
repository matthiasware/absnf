namespace cuutils
{
	template <typename T>
	__global__ void vvAdd(T *u, T *v, T *z, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		if(id < size)
		{
			z[id] = u[id] + v[id];
		}
	}
}