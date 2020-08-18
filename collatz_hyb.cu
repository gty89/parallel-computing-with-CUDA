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
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static int* d_maxlen;

static __global__ void collatz(const long start, const long stop, int* const maxlen)
{
  // todo: process values from start to stop (excluding stop) with one thread per value based on code from previous project
  const long i = start + threadIdx.x + blockIdx.x * (long)blockDim.x;
  if( i < stop)
  {
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
  if( len > *maxlen)
    atomicMax( maxlen, len);
  }
  return;
}

void GPU_Init(void)
{
  int maxlen = 0;
  if (cudaSuccess != cudaMalloc((void **)&d_maxlen, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_maxlen, &maxlen, sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
}

void GPU_Exec(const long start, const long stop)
{
  if (start < stop) {
    // todo: launch the kernel with just the right number of blocks and ThreadsPerBlock threads per block and do nothing else
    
    collatz<<< ( stop - start + ThreadsPerBlock - 1)/ ThreadsPerBlock, ThreadsPerBlock>>>(start, stop, d_maxlen);
    
  }
}

int GPU_Fini(void)
{
  int maxlen;

  // todo: copy the result from the device to the host and free the device memory
  if (cudaSuccess != cudaMemcpy(&maxlen, d_maxlen, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}
  cudaFree( d_maxlen);
  return maxlen;
}
