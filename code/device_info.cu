#include <iostream>
int main()
{
	cudaDeviceProp prop;
	int devcount;
	cudaGetDeviceCount(&devcount);
	for(int i=0; i<devcount; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		std::cout << "------------------" << std::endl;
		std::cout << "Device: " << i << std::endl;
		std::cout << "------------------" << std::endl;
		std::cout << "Name:\t\t\t" << prop.name << std::endl;
		std::cout << "GlobalMemory:\t\t" << prop.totalGlobalMem << std::endl;
		std::cout << "WarpSize:\t\t" << prop.warpSize << std::endl;
		std::cout << "MaxThreadsPerBlock:\t" << prop.maxThreadsPerBlock << std::endl;
		std::cout << "MaxThreadsDim:\t\t" << prop.maxThreadsDim[0] << " : " << prop.maxThreadsDim[1] << " : " << prop.maxThreadsDim[2] << std::endl;
		std::cout << "MaxGridSize:\t\t" << prop.maxGridSize[0] << " : " << prop.maxGridSize[1] << " : " << prop.maxGridSize[2] << std::endl;
		std::cout << "MultiProcessorCount:\t" << prop.multiProcessorCount << std::endl;
	}
	return 0;
}