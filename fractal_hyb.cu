/*
Fractal code for CS 4380 / CS 5351

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
#include <cmath>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void fractal(const int width, const int start_frame, const int gpu_frames, unsigned char* const pic)
{
  // todo: use the GPU to compute the requested frames (base the code on the previous project)
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if( i < gpu_frames * width * width){
    const double Delta = 0.01;
    const double xMid = 0.28017435;
    const double yMid = 0.60538436;
  
    const int col = i % width;
    const int row = (i / width) % width;
    const int frame = i / (width * width) + start_frame;
    // compute pixels of each frame
    const double delta = Delta * pow( 0.981, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    const double cy = yMin + row * dw;
    const double cx = xMin + col * dw;
    double x = cx;
    double y = cy;
    double x2, y2;
    int depth = 256;
    do {
      x2 = x * x;
      y2 = y * y;
      y = 2.0 * x * y + cy;
      x = x2 - y2 + cx;
      depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0));
        pic[(frame - start_frame) * width * width + row * width + col] = (unsigned char)depth;
  }
}

unsigned char* GPU_Init(const int gpu_frames, const int width)
{
  unsigned char* d_pic;
  if (cudaSuccess != cudaMalloc((void **)&d_pic, gpu_frames * width * width * sizeof(unsigned char))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  return d_pic;
}

void GPU_Exec(const int start_frame, const int gpu_frames, const int width, unsigned char* d_pic)
{
  // todo: launch the kernel with just the right number of blocks and ThreadsPerBlock threads per block and do nothing else
  fractal<<< ( gpu_frames * width * width + ThreadsPerBlock - 1)/ThreadsPerBlock, ThreadsPerBlock>>>(width, start_frame, gpu_frames, d_pic);
}

void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* d_pic)
{
  // todo: copy the result from the device to the host and free the device memory
  if (cudaSuccess != cudaMemcpy( pic, d_pic, sizeof(unsigned char) * gpu_frames * width * width, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}
  cudaFree(d_pic);
}
