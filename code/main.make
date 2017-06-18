# all : utils.o cuutils.o absnf.o test_cuutils eval devinfo gradient
all : eval test_eval devinfo test_utils

# test_cuutils: test_cuutils.cu utils.h
# 	nvcc -std=c++11 test_cuutils.cu -o tests/test_cuutils

eval: eval.cu utils.hpp absnf.h
	nvcc -std=c++11 eval.cu -lcublas -o eval

test_eval: test_eval.cu absnf.h utils.hpp
	nvcc -std=c++11 test_eval.cu -lcublas -o test_eval

test_utils: test_utils.cpp utils.hpp
	g++ -std=c++11 test_utils.cpp -o test_utils

devinfo: device_info.cu
	nvcc -std=c++11 device_info.cu -o devinfo

# gradient: gradient.cu utils.o cuutils.o
# 	nvcc -std=c++11 gradient.cu utils.o cuutils.o -lcublas -o gradient