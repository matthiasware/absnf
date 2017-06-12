all : utils.o cuutils.o eval

utils.o: utils.cpp
	g++ -std=c++11 -c utils.cpp -o utils.o

cuutils.o: cuutils.cu
	nvcc -std=c++11  -c cuutils.cu -o cuutils.o

eval: eval.cu utils.o cuutils.o
	nvcc -std=c++11 eval.cu utils.o cuutils.o -lcublas -o eval