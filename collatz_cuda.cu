/*
Collatz code for CS 4380 / CS 5351

Copyright (c) 2020 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void collatz(const long bound, int* const maxlen)
{
  // compute sequence lengths
  const long i = threadIdx.x + blockIdx.x * (long)blockDim.x + 1;
  if( i < bound) {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
  // atomicMax( maxlen, max( maxlen, len));
  if( len > *maxlen)
    atomicMax( maxlen, len); 
  }
  return;
}


static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.3\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  const long bound = atol(argv[1]);
  if (bound < 2) {fprintf(stderr, "ERROR: upper_bound must be at least 2\n"); exit(-1);}
  printf("upper bound: %ld\n", bound);
  
  int maxlen = 0;
  int *d_maxlen;
  int size = sizeof(int);
  if (cudaSuccess != cudaMalloc((void**) &d_maxlen, size)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_maxlen, &maxlen, size, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  collatz<<<(bound + ThreadsPerBlock - 2)/ThreadsPerBlock, ThreadsPerBlock>>>(bound, d_maxlen); 
  // I use -2 instead of -1 because of more precise. For instance, if bound==11 and Threads==5, we only use 10 threads to conputer bound from 1 to 10, so we need only 2 blocks instead of 3
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  CheckCuda();
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.5f s\n", runtime);

  // print result
  if (cudaSuccess != cudaMemcpy(&maxlen, d_maxlen, size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}
  printf("longest sequence: %d elements\n", maxlen);
  cudaFree( d_maxlen);

  return 0;
}
