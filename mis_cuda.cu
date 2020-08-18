/*
Maximal independent set code for CS 4380 / CS 5351

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

#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include "ECLgraph.h"
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static const unsigned char in = 2;
static const unsigned char out = 1;
static const unsigned char undecided = 0;

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int hash (unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static __global__ void init( const ECLgraph g, unsigned char* const status, unsigned int* const random)
{
  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if( v< g.nodes)
  {
    status[v] = undecided;
    random[v] = hash(v + 712459897);
  }
  return;
}

static __global__ void mis(const ECLgraph g, volatile unsigned char* const status, unsigned int* const random, volatile bool *missing)
{
  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if( v < g.nodes && status[v] == undecided)
  {
        // go over all the nodes
  
        int i = g.nindex[v];
        // try to find a neighbor whose random number is lower
        while ((i < g.nindex[v + 1]) && ((status[g.nlist[i]] == out) || (random[v] < random[g.nlist[i]]) || ((random[v] == random[g.nlist[i]]) && (v < g.nlist[i])))) {
          i++;
        }
        if (i < g.nindex[v + 1]) {
          // found such a neighbor -> status still unknown
          *missing = true;
        } else {
          // no such neighbor -> all neighbors are "out" and my status is "in"
          for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
            status[g.nlist[i]] = out;
          }
          status[v] = in;
        }
      
    
  }
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

int main(int argc, char* argv[])
{
  printf("Maximal Independent Set v1.4\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // allocate arrays
  unsigned char* const status = new unsigned char [g.nodes];
  unsigned char* d_status;
  unsigned int* d_random;
  bool *d_missing;
  bool missing;
  if (cudaSuccess != cudaMalloc((void **)&d_status, sizeof(unsigned char) * g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_random, sizeof(int) * g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_missing, sizeof(bool))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  init<<< ( g.nodes + ThreadsPerBlock -1)/ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_random);
  // execute timed code
  do
  {
    missing = false;
    cudaMemcpy(d_missing, &missing, sizeof(bool), cudaMemcpyHostToDevice);
    mis<<< ( g.nodes + ThreadsPerBlock -1)/ThreadsPerBlock, ThreadsPerBlock>>>( d_g, d_status, d_random, d_missing); 
    cudaMemcpy(&missing, d_missing, sizeof(bool), cudaMemcpyDeviceToHost);
  }
  while( missing);
  cudaDeviceSynchronize();
  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.5f s\n", runtime);
  CheckCuda();
  cudaMemcpy(status, d_status, sizeof(unsigned char) * g.nodes, cudaMemcpyDeviceToHost);
  // determine and print set size
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == in) {
      count++;
    }
  }
  printf("elements in set: %d (%.1f%%)\n", count, 100.0 * count / g.nodes);

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != in) && (status[v] != out)) {fprintf(stderr, "ERROR: found unprocessed node\n"); exit(-1);}
    if (status[v] == in) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }
  printf("verification passed\n");

  // clean up
  freeECLgraph(g);
  delete [] status;
  
  cudaFree(d_status);
  cudaFree(d_random);
  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);
  cudaFree(d_missing);
  //freeECLgraph(d_g);
  return 0;
}
