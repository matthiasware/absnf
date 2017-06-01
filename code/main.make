all:
	nvcc -std=c++11 iterative_eval.cu utils.cpp -lcublas -o main