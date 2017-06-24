### TODO
- function for cudamalloc + copy
- function for copy host to device
- function for copy device to host
- use classes for core functions
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

# Vortrag
- Content
- Aufgabenbeschreibung
- ABSNF
- Aufgabe 1)
- Aufgabe 2)
- Aufgabe 3)
- Application
- Numerical tests
- Cool Snippets
- Common Problems: grids vs blocks
- Room for improvements
- Lesson learned: prototyping

https://stackoverflow.com/questions/28794010/solving-linear-systems-ax-b-with-cuda?noredirect=1&lq=1
https://stackoverflow.com/questions/22399794/qr-decomposition-to-solve-linear-systems-in-cuda
http://docs.nvidia.com/cuda/cublas/index.html#axzz4kNclpKOl
https://git.inf-ra.uni-jena.de/sa26lat/absnf/blob/master/code/absnf.h