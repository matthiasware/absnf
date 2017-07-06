# all : utils.o cuutils.o absnf.o test_cuutils eval devinfo gradient
all : test_absnf devinfo test_utils solve test num_eval num_grad num_blocksize

# test_cuutils: test_cuutils.cu utils.h
# 	nvcc -std=c++11 test_cuutils.cu -o tests/test_cuutils

test_absnf: test_absnf.cu absnf.h utils.hpp cuutils.h
	nvcc -std=c++11 -O0 test_absnf.cu -lcublas -lcusolver -o test_absnf

test_utils: test_utils.cpp utils.hpp
	g++ -std=c++11 -O0 test_utils.cpp -o test_utils

devinfo: device_info.cu
	nvcc -std=c++11 -O0 device_info.cu -o devinfo

solve: solve.cu
	nvcc -std=c++11 -O0 solve.cu -lcublas -lcusolver -o solve 

test: test.cu absnf.h utils.hpp cuutils.h
	nvcc -std=c++11 -O0 test.cu -lcublas -lcusolver -o test

performance_eval: num_eval.cu absnf.h utils.hpp cuutils.h
	nvcc -std=c++11 num_eval.cu -lcublas -lcusolver -o num_eval

num_grad: num_grad.cu absnf.h utils.hpp cuutils.h
	nvcc -std=c++11 num_grad.cu -lcublas -lcusolver -o num_grad	

num_blocksize: num_blocksize.cu absnf.h utils.hpp cuutils.h
	nvcc -std=c++11 num_blocksize.cu -lcublas -lcusolver -o num_blocksize	
