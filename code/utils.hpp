#include <random>
#include <iostream>

namespace utils
{
	void pp();
	enum MATRIXOPT
	{
		NONE,
		PDF,
		PSD,
		LOWER,
		UPER,
		SYMM
	};
	enum VALUEOP
	{
		REAL,
		INT
	};
	template <typename T>
	void fillRandMatrixNone(T *matrix, int rows, int cols,
							std::mt19937 gen,
						    int first, int last,
						    VALUEOP vo=REAL)
	{
		std::uniform_real_distribution<> dis(first, last);
		if(vo == REAL)
		{
			for(int i=0; i<rows; i++)
			{
				for(int j=0; j<cols; j++)
				{
					int idx = i * cols + j;
					matrix[idx] = (T) dis(gen);
				}
			}
		}
		else if(vo == INT)
		{
			for(int i=0; i<rows; i++)
			{
				for(int j=0; j<cols; j++)
				{
					int idx = i * cols + j;
					matrix[idx] = (T) (int) dis(gen);
				}
			}
		}
	}
	template <typename T>
	void fillRandMatrixLower(T *matrix, int rows, int cols,
							 std::mt19937 gen,
							 int first, int last,
							 VALUEOP vo=REAL)
	{
		std::uniform_real_distribution<> dis(first, last);
		if(vo==REAL)
		{
			for(int i=0; i<rows; i++)
			{
				for(int j=0; j<i; j++)
				{
					int idx = i * cols + j;
					matrix[idx] = (T) dis(gen);
				}
			}
		}
		else if(vo == INT)
		{
			for(int i=0; i<rows; i++)
			{
				for(int j=0; j<i; j++)
				{
					int idx = i * cols + j;
					matrix[idx] = (T) (int) dis(gen);
				}
			}			
		}
	}
	template <typename T>
	void fillRandMatrix(T *matrix, int rows, int cols, 
						int first=0, int last=10,
						int seed=0,
						MATRIXOPT mo=NONE,
						VALUEOP vo=REAL)
	{
		std::mt19937 gen;
		if(seed > 0)
			gen.seed(seed);
		else
		{
			std::random_device rd;
			gen.seed(rd());
		}
		switch(mo)
		{
			case MATRIXOPT::NONE : 
				fillRandMatrixNone(matrix, rows, cols, gen, first, last, vo);
				break;
			case MATRIXOPT::LOWER : 
				fillRandMatrixLower(matrix, rows, cols, gen, first, last, vo);
				break;
			default: 
				throw "NOT IMPLEMENTED";
		}

	}
	template <typename T>
	void fillRandVector(T *vector, int size, 
					    int first=0, int last=10,
					    int seed=0,
					    VALUEOP vo=REAL)
	{
		std::mt19937 gen;
		if(seed > 0)
			gen.seed(seed);
		else
		{
			std::random_device rd;
			gen.seed(rd());
		}
		std::uniform_real_distribution<> dis(first, last);
		if(vo == REAL)
		{
			for (int i=0; i<size; i++)
			{
				vector[i] = (T) dis(gen);
			}
		}
		else if(vo == INT)
		{
			for (int i=0; i<size; i++)
			{
				vector[i] = (T) (int) dis(gen);
			}
		}

	}
	template <typename T>
	void printf_matrix(T *A, int cols, int rows)
	{
		std::cout << "[\n";
		for(int i=0; i<rows; i++)
		{
			std::cout << "[";
			for(int j=0; j<cols; j++)
			{
				int idx = i*cols + j;
				std::cout << A[idx] << ",";
			}
			std::cout << "]";	
			std::cout << "\n";
		}
		std::cout << "]\n";
	}
	template <typename T>
	void printf_vector(T *A, int cols)
	{
		std::cout << "[";
		for(int i=0; i<cols; i++)
		{
			std::cout << A[i] << ",";
		}
		std::cout << "]\n";
	}
};