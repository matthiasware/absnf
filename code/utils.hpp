#ifndef __UTILS_H_INCLUDED__
#define __UTILS_H_INCLUDED__
#include <random>
#include <iostream>
#include <string.h>
#include <math.h> 
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define DEBUG_UTILS false

namespace utils
{

	enum MATRIXOPT
	{
		NONE,
		PDF,
		PSD,
		LOWER,
		UPER,
		SYMM,
		INVERTIBLE
	};
	enum VALUEOP
	{
		REAL,
		INT
	};
	// ---------------------------------------------------
	//  DECLARATIONS
	// ---------------------------------------------------
	template <typename T>
	void printf_vector(T *A, int cols, const std::string& name = "");

	template <typename T>
	void printf_matrix(T *A, int rows, int cols, const std::string& name = "");

	template <typename T>
	void printf_matrix_C2R(T *A, int rows, int cols, const std::string& name = "");

	template <typename T>
	void fillRandMatrixNone(T *matrix, int rows, int cols, std::mt19937 gen,
						    int first, int last, VALUEOP vo=REAL);

	template <typename T>
	void fillRandMatrixLower(T *matrix, int rows, int cols,
							 std::mt19937 gen, int first, int last, VALUEOP vo=REAL);

	template <typename T>
	void fillRandMatrixLowerCM(T *matrix, int rows, int cols,
							 std::mt19937 gen, int first, int last, VALUEOP vo=REAL);

	template <typename T>
	void fillRandMatrixInvertible(T *matrix, int rows, int cols,
							 std::mt19937 gen,
							 int first, int last,
							 VALUEOP vo=REAL);
	template <typename T>
	void fillRandMatrix(T *matrix, int rows, int cols, 
						int first=0, int last=10,
						int seed=0,
						MATRIXOPT mo=NONE,
						VALUEOP vo=REAL);

	template <typename T>
	void fillRandMatrixCM(T *matrix, int rows, int cols, 
						int first=0, int last=10,
						int seed=0,
						MATRIXOPT mo=NONE,
						VALUEOP vo=REAL);

	template <typename T>
	void fillRandVector(T *vector, int size, 
					    int first=0, int last=10,
					    int seed=0,
					    VALUEOP vo=REAL);

	template <typename T>
	void fillVector(T *vector, int size, T value);

	template <typename T>
	void rowColConversion(T *m_source, T *m_target, int rows, int cols);

	template <typename T>
	void rowColConversion(T *matrix, int rows, int cols, bool rm=true);

	template <typename T>
	bool vectors_equals(T* a, T* b, int size, bool verbose=DEBUG_UTILS);

	template <typename T>
	bool vectors_almost_equal(T* a, T* b, int size, double tol=1e-6, bool verbose=DEBUG_UTILS);

	// ---------------------------------------------------
	//  IMPLEMENTATION
	// ---------------------------------------------------
	template <typename T>
	bool vectors_almost_equal(T* a, T* b, int size, double tol, bool verbose)
	{
		if(verbose)
		{
			std::cout << "COMPARING : " << std::endl;
			utils::printf_vector(a, size);
			utils::printf_vector(b, size);

		}
		for(int i=0; i<size; i++)
		{
			if(fabs(a[i] - b[i]) > tol)
			{	
				std::cout << a[i] << " != " << b[i] << std::endl;
				return false;
			}
		}
		return true;
	}
	template <typename T>
	void printf_vector(T *A, int cols, const std::string& name)
	{
		if (name.size() > 0)
		{
			std::cout << name << std::endl;
		}
		std::cout << "[";
		for(int i=0; i<cols; i++)
		{
			std::cout << A[i] << ",";
		}
		std::cout << "]\n";
	}
	template <typename T>
	void fillRandMatrixNone(T *matrix, int rows, int cols,
							std::mt19937 gen,
						    int first, int last,
						    VALUEOP vo)
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
							 VALUEOP vo)
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
	void fillRandMatrixInvertible(T *matrix, int rows, int cols,
							 std::mt19937 gen,
							 int first, int last,
							 VALUEOP vo)
	{
		std::uniform_real_distribution<> dis(first, last);
		if(vo==REAL)
		{
			for(int i=0; i<rows; i++)
			{
				for(int j=0; j<cols; j++)
				{
					int idx = i * cols + j;
					matrix[idx] = (T) dis(gen);
					if (i == j)
					{
						matrix[idx] = (T) 1000;
					}
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
					if (i == j)
					{
						matrix[idx] = (T) 1000;
					}
				}
			}			
		}
		else
		{
			throw "NOT IMPLEMENTED";
		}
	}
	template <typename T>
	void fillRandMatrixLowerCM(T *matrix, int rows, int cols,
							 std::mt19937 gen,
							 int first, int last,
							 VALUEOP vo)
	{
		std::uniform_real_distribution<> dis(first, last);
		if(vo==REAL)
		{
			for(int i=0; i<rows; i++)
			{
				for(int j=0; j<i; j++)
				{
					int idx = IDX2C(i,j, rows);
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
					int idx = IDX2C(i,j, rows);
					matrix[idx] = (T) (int) dis(gen);
				}
			}			
		}
	}
	template <typename T>
	void fillRandMatrix(T *matrix, int rows, int cols, 
						int first, int last,
						int seed,
						MATRIXOPT mo,
						VALUEOP vo)
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
			case MATRIXOPT::INVERTIBLE:
				fillRandMatrixInvertible(matrix, rows, cols, gen, first, last, vo);
				break;
			default: 
				throw "NOT IMPLEMENTED";
		}

	}
	// FILLS MATRIX IN COLUMN MAJOR
	template <typename T>
	void fillRandMatrixCM(T *matrix, int rows, int cols, 
						int first, int last,
						int seed,
						MATRIXOPT mo,
						VALUEOP vo)
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
				fillRandMatrixLowerCM(matrix, rows, cols, gen, first, last, vo);
				break;
			default: 
				throw "NOT IMPLEMENTED";
		}

	}
	template <typename T>
	void fillRandVector(T *vector, int size, 
					    int first, int last,
					    int seed,
					    VALUEOP vo)
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
	void rowColConversion(T *m_source, T *m_target, int rows, int cols)
	{
		for(int i=0; i<rows; i++)
		{
			for(int j=i; j<cols; j++)
			{
				m_target[j*rows+i] = m_source[i*cols+j];
			}
		}
	}
	template <typename T>
	void rowColConversion(T *matrix, int rows, int cols, bool rm)
	{
		if(!rm)
		{
			cols ^= rows;
			rows ^= cols;
			cols ^= rows;
		}
		int size = rows*cols;
		bool bitmap[size];
		memset(bitmap, false, size*sizeof(bool));
		bitmap[0] = bitmap[size-1] = true;
		int i_start;
		int i_now;
		int i_next;
		T data_now;
		T data_temp;
		for(int k=1; k<size; k++)
		{
			if(!bitmap[k])
			{
				i_start = k;
				i_now = i_start;
				data_now = matrix[i_start];
				do
				{
					i_next = (i_now*rows)%(size-1);
					data_temp = matrix[i_next];
					matrix[i_next] = data_now;
					bitmap[i_next] = true;
					data_now = data_temp;
					i_now = i_next;
				}
				while(i_next != i_start);
			}
		}
	}
	template <typename T>
	void fillVector(T *vector, int size, T value)
	{
		for(int i=0; i<size; i++)
		{
			vector[i] = value;
		}
	}
	template <typename T>
	void printf_matrix(T *A, int rows, int cols, const std::string& name)
	{
		if (name.size() > 0)
		{
			std::cout << name << std::endl;
		}
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
	// PRINTS COLUMN MAJOR MATRIX
	template <typename T>
	void printf_matrix_C2R(T *A, int rows, int cols, const std::string& name)
	{
		if (name.size() > 0)
		{
			std::cout << name << std::endl;
		}
		std::cout << "[\n";
		for(int i=0; i<rows; i++)
		{
			std::cout << "[";
			for(int j=0; j<cols; j++)
			{
				int idx = IDX2C(i,j,rows);
				std::cout << A[idx] << ",";
			}
			std::cout << "]";	
			std::cout << "\n";
		}
		std::cout << "]\n";
	}
	template <typename T>
	bool vectors_equals(T* a, T* b, int size, bool verbose)
	{
		if(verbose)
		{
			std::cout << "COMPARING: " << std::endl;
			utils::printf_vector(a, size);
			utils::printf_vector(b, size);

		}
		for(int i=0; i<size; i++)
		{
			if(a[i] != b[i])
			{	
				std::cout << a[i] << " != " << b[i] << std::endl;
				return false;
			}
		}
		return true;
	}
};

#endif // __UTILS_H_INCLUDED