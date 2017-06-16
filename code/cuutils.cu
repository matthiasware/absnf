namespace cuutils
{
	void getGridBlockSize(int *gridsize, int *blocksize)
	{
		/*
			We want to be able to work with structures of 
			arbitrary sizes. Therefore we chose the gridsize,
			depending on the amout of MPUs.
			the blocksize is the maximum amout of threads, 
			that can be executed within a thread.
		*/
		cudaDeviceProp prop;
		int devcount;
		cudaGetDeviceCount(&devcount);
		// Take first device, 
		// TODO: room for improvements
		cudaGetDeviceProperties(&prop, 0);
		// we decided to run 8 blocks / MPU
		// TODO: room for improvements
		*gridsize = prop.multiProcessorCount * 8;
		*blocksize = prop.maxThreadsPerBlock;
	}
}