# ABS-NF
This project contrains a Python and CUDA/C++ implementation of the following operations to deal with picewise linear functions in abs-normal form:

 * Evaluate
 * Gradient
 * Solve

We were specifically interested, to what extend performance boost through massive parallelization could be achieved.

A theoretical workout can be found [here](/script/script.pdf).


## Result overview.

### Gradient: CPU vs GPU 
![gradient](presentation/img/grad-100.png)

### Eval CPU vs GPU 
![eval](presentation/img/eval-100.png)