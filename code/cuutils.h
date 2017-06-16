namespace cuutils
{
	template <typename T>
	__global__ void vvAdd(T *u, T *v, T *z, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			z[id] = u[id] + v[id];
			// increment by the total number of threads running
			// such that we can handle structures of arbitrary size
			id += blockDim.x * gridDim.x;
		}
	}
	template <typename T>
	__global__ void makeSignumVector(T *v_source, T *v_target, int size)
	{	
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			v_target[id] = (T(0)  < v_source[id]) - (v_source[id] < T(0));
			id += blockDim.x * gridDim.x;
		}
	}
	template <typename T>
	__global__ void abs(T *v_source, T *v_target, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		while(id < size)
		{
			v_target[id] = (T) fabs(v_source[id]);
			id += blockDim.x * gridDim.x;
		}	
	}
	void getGridBlockSize(int *gridsize, int *blocksize);
}