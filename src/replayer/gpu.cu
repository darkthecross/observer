#include "gpu.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

namespace gpu_util {

namespace {

#define IMG_W 848
#define IMG_H 480

__global__ void FilterPixel(unsigned char* img, unsigned char thres)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= IMG_W * IMG_H) return;
  if (img[i] < thres) {
    img[i] = 0;
  } else {
    img[i] = 255;
  }
}

// TODO(darkthecross): Use shared memory.
__global__ void MeanShiftIteration(unsigned char* img, unsigned char* out_img, int kernel_size, float diff_thresh) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int img_y = i % IMG_W;
  int img_x = i / IMG_W;
  int max_offset = kernel_size / 2;
  unsigned int num_valid_pixels = 0;
  int avg_pixel = 0;
  for(int xx = -max_offset; xx <= max_offset; ++xx) {
    for(int yy = -max_offset; yy <= max_offset; ++yy) {
      int xp = img_x + xx;
      int yp = img_y + yy;
      if(xp >= 0 && xp < IMG_H && yp >= 0 && yp < IMG_W) {
        unsigned char cur_pixel = *(img + xp * IMG_W + yp);
        if( cur_pixel > 0 ) {
          ++num_valid_pixels;
          avg_pixel += int(cur_pixel);
        }
      }
    }
  }
  avg_pixel = float(avg_pixel) / float(num_valid_pixels);
  if( (float)abs( int(*(img+i)) - avg_pixel ) / (float)*(img+i) < diff_thresh ) {
    *(out_img + i) = avg_pixel;
  } else {
    *(out_img + i) = *(img+i);
  }
}

__global__ void CopyDeviceMemory(unsigned char* from, unsigned char* to, int* diff_counter) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= IMG_W * IMG_H) return;
  if(*(to + i) != *(from + i)) {
    atomicAdd(diff_counter, 1);
  }
  *(to + i) = *(from + i);
}



}  // namespace

void ThresholdImage(unsigned char* in_mat, unsigned char* out_mat, unsigned char thresh) {
  unsigned char *device_mat;

  cudaMalloc(&device_mat, IMG_W*IMG_H*sizeof(unsigned char)); 
  cudaMemcpy(device_mat, in_mat, IMG_W*IMG_H*sizeof(unsigned char), cudaMemcpyHostToDevice);

  FilterPixel<<<(IMG_W*IMG_H+255)/256, 256>>>(device_mat, thresh);

  cudaMemcpy(out_mat, device_mat, IMG_W*IMG_H*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void MeanShiftSegmentation(unsigned char* in_mat, unsigned char* out_mat, int kernel_size, int num_max_iter, float diff_thresh) {
  unsigned char *device_mat, *iter_mat;

  cudaMalloc(&device_mat, IMG_W*IMG_H*sizeof(unsigned char)); 
  cudaMemcpy(device_mat, in_mat, IMG_W*IMG_H*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMalloc(&iter_mat, IMG_W*IMG_H*sizeof(unsigned char)); 

  int* device_diff_counter;
  cudaMalloc(&device_diff_counter, sizeof(int)); 
  int host_diff;
  
  for(int i = 0; i<num_max_iter; ++i) {
    cudaMemset(device_diff_counter, 0, sizeof(int));
    MeanShiftIteration<<<(IMG_W*IMG_H+255)/256, 256>>>(device_mat, iter_mat, kernel_size, diff_thresh);
    CopyDeviceMemory<<<(IMG_W*IMG_H+255)/256, 256>>>(iter_mat, device_mat, device_diff_counter);
    cudaMemcpy(&host_diff, device_diff_counter, sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaMemcpy(out_mat, device_mat, IMG_W*IMG_H*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}


}  // namespace gpu_util