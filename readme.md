### TODO
- chose block and thread size depending on s,m,n
- block size
- row column major storage converter
- row colum major print
- unittests
- Device choser

### QA
Programming
- python3 libraries numpy ect.
- abs of one value,transfer to cpu?
- implement on my own or use cublas?
- synchrone?
- how to handle headers?

Math
- Gradient, how does it work?
- Signum function? at 0?


# Threads and Blocks
vector a,b |a| = 1000
gpumpus: 4, 100 threads each

add <<<1, 1000>>> (a,b)
add <<<1000, 1>>> (a,b)
add <<<4, 250>>> (a,b)

get device information generically


-> get blocksize, devide tasks, s.t. they are qually devided on blocks