# all : utils.o cuutils.o absnf.o test_cuutils eval devinfo gradient
all : test_absnf devinfo test_utils solve test

# test_cuutils: test_cuutils.cu utils.h
# 	nvcc -std=c++11 test_cuutils.cu -o tests/test_cuutils

test_absnf: test_absnf.cu absnf.h utils.hpp
	nvcc -std=c++11 test_absnf.cu -lcublas -o test_absnf

test_utils: test_utils.cpp utils.hpp
	g++ -std=c++11 test_utils.cpp -o test_utils

devinfo: device_info.cu
	nvcc -std=c++11 device_info.cu -o devinfo

solve: solve.cu
	nvcc -std=c++11 solve.cu -lcublas -lcusolver -o solve 

test: test.cu
	nvcc -std=c++11 test.cu -lcublas -lcusolver -o test