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
	template <typename T>
	__global__ void makeSignumVector(T *v_source, T *v_target, int size)
	{	
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		if(id < size)
		{
			if(v_source[id] == 0)
			{
				v_target[id] = (T) 0;
			}
			else if(v_source[id] > - v_source[id])
			{
				v_target[id] = (T) 1;
			}
			else
			{
				v_target[id] = (T) -1;
			}
		}
	}
	template <typename T>
	__global__ void makeDiagMatrixFromVector(T *matrix, T *vector, int size)
	{
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		if(true)
		{
			matrix[id] = 0;
		}
	}
}