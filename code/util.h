#ifndef UTIL_H
#define UTIL_H

void printf_matrix(double *A, int cols, int rows);
void printf_vector(double *A, int cols);
void Ay(double *A, double *y, int s, int i);
void AyPLUSb(double *A, double *y, double *b, int s);
void initData(double *x, int s);
void initData(double *x, int s, double val);
void matrixMvector(double *A, double *x, double *y, int rows, int cols);
void vecAvec(double *a, double *b, double *c, int s);
void initLowerTriangular(double *A, int rows, int cols);

#endif