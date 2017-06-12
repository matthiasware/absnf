all : utils.o cuutils.o test_cuutils eval

utils.o: utils.cpp
	g++ -std=c++11 -c utils.cpp -o utils.o

cuutils.o: cuutils.cu
	nvcc -std=c++11  -c cuutils.cu -o cuutils.o

test_cuutils: test_cuutils.cu utils.o
	nvcc -std=c++11 test_cuutils.cu cuutils.o utils.o -o tests/test_cuutils

eval: eval.cu utils.o cuutils.o
	nvcc -std=c++11 eval.cu utils.o cuutils.o -lcublas -o eval