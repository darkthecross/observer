#ifndef GPU_H
#define GPU_H

namespace gpu_util {

void ThresholdImage(unsigned char* in_mat, unsigned char* out_mat, unsigned char thresh);

void MeanShiftSegmentation(unsigned char* in_mat, unsigned char* out_mat, int kernel_size, int num_max_iter, float diff_thresh);

}  // namespace gpu_util

#endif