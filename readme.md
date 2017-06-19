### TODO
- Write MEMORY calculation, before starting the functions
- CONSIDER MAX-RAM!!!
- handle case, that blocksize > max_blocksize!!!
- handle case, that we allocate too much memory
- chose block and thread size depending on s,m,n
- block size
- row column major storage converter
- row colum major print
- unittests
- Device choser
- write optimizer for blocksize, gridsize
- add support for multiple gpus

### QA
Programming
- header vs cpp
- speed vs memory
- python3 libraries numpy ect.
- abs of one value,transfer to cpu?
- implement on my own or use cublas?
- synchrone?
- how to handle headers?

Math
- Gradient, how does it work?
- Signum function? at 0?


# Threads and Blocks
- why not using max thread per block?