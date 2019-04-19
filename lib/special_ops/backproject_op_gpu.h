
#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_ROIPOOLING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_ROIPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {


bool BackProjectForwardLauncher(const float* input, const float* coords,
  const int dim[6], float *top, const Eigen::GpuDevice& d);

bool BackProjectBackwardLauncher(const float *grad, 
  const float* input, const float* coords, const int dim[6],
  float* inputs_diff, float* coords_diff, const Eigen::GpuDevice& d);


}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
