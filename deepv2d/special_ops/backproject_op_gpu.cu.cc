#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "backproject_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void BackProjectForward(const int nthreads, const Dtype* input,
    const Dtype* coords, int b, int h, int w, int s, int f, int c, Dtype *top_data)
{
  int dims[6];
  dims[0] = b;
  dims[1] = h;
  dims[2] = w;
  dims[3] = s;
  dims[4] = f;
  dims[5] = c;

  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
    //
    // dims: [B[0], H[1], W[2], S[3], F[4], C]
    // input: [B[0], H[1], W[2], F[4], C[5]]
    // coords: [B[0], H[1], W, S, F, 2]

    int n = index;
    int f = n % dims[4]; n /= dims[4];
    int k = n % dims[3]; n /= dims[3];
    int w = n % dims[2]; n /= dims[2];
    int h = n % dims[1]; n /= dims[1];

    Dtype x = coords[2*index];
    Dtype y = coords[2*index+1];

    if (x>0 && y>0 && x<dims[2]-1 && y<dims[1]-1) {
      int x0 = static_cast<int>(floor(x));
      int x1 = static_cast<int>(ceil(x));
      int y0 = static_cast<int>(floor(y));
      int y1 = static_cast<int>(ceil(y));

      Dtype dx = x - static_cast<Dtype>(x0);
      Dtype dy = y - static_cast<Dtype>(y0);

      Dtype w00 = (1-dy)*(1-dx);
      Dtype w01 = (1-dy)*dx;
      Dtype w10 = dy*(1-dx);
      Dtype w11 = dy*dx;

      int offset = (n*dims[1]*dims[2]*dims[4]+f)*dims[5];
      int idx00 = offset + dims[4]*dims[5]*(y0*dims[2] + x0);
      int idx01 = offset + dims[4]*dims[5]*(y0*dims[2] + x1);
      int idx10 = offset + dims[4]*dims[5]*(y1*dims[2] + x0);
      int idx11 = offset + dims[4]*dims[5]*(y1*dims[2] + x1);

      const Dtype *im00 = input + idx00;
      const Dtype *im01 = input + idx01;
      const Dtype *im10 = input + idx10;
      const Dtype *im11 = input + idx11;

      Dtype *top = top_data + index*dims[5];
      for (int c=0; c<dims[5]; c++) {
        *top = (*im00)*w00+(*im01)*w01+(*im10)*w10+(*im11)*w11;
        im00++; im01++; im10++; im11++; top++;
      }
    }
  }
}



bool BackProjectForwardLauncher(const float* input, const float* coords,
  const int dim[6], float *top, const Eigen::GpuDevice& d)
{

  cudaError_t err;
  const int kblock = 512;
  const int nthreads = dim[0]*dim[1]*dim[2]*dim[3]*dim[4];

  cudaMemset((void*) top, 0, nthreads*dim[5]*sizeof(float));
  BackProjectForward<<<(nthreads+kblock-1)/kblock, kblock, 0, d.stream()>>>(
    nthreads, input, coords, dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], top);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit(-1);
  }

  return d.ok();
}




template <typename Dtype>
__global__ void BackProjectBackward(const int nthreads,
    const Dtype* top_diff, const Dtype* input, const Dtype* coords,
    int b, int h, int w, int s, int f, int c, Dtype* input_diff,
    Dtype* coords_diff)
{
  int dims[6];
  dims[0] = b;
  dims[1] = h;
  dims[2] = w;
  dims[3] = s;
  dims[4] = f;
  dims[5] = c;

  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {

    // dims: [B[0], H[1], W[2], S[3], F[4], C]
    // input: [B[0], H[1], W[2], F[4], C[5]]
    // coords: [B[0], H[1], W, S, F, 2]

    int n = index;
    int f = n % dims[4]; n /= dims[4];
    int k = n % dims[3]; n /= dims[3];
    int w = n % dims[2]; n /= dims[2];
    int h = n % dims[1]; n /= dims[1];

    Dtype x = coords[2*index];
    Dtype y = coords[2*index+1];

    if (x>0 && y>0 && x<dims[2]-1 && y<dims[1]-1) {
      int x0 = static_cast<int>(floor(x));
      int x1 = static_cast<int>(ceil(x));
      int y0 = static_cast<int>(floor(y));
      int y1 = static_cast<int>(ceil(y));

      Dtype dx = x - static_cast<Dtype>(x0);
      Dtype dy = y - static_cast<Dtype>(y0);

      Dtype wx0 = 1-dx;
      Dtype wx1 = dx;
      Dtype wy0 = 1-dy;
      Dtype wy1 = dy;

      Dtype w00 = (1-dy)*(1-dx);
      Dtype w01 = (1-dy)*dx;
      Dtype w10 = dy*(1-dx);
      Dtype w11 = dy*dx;

      int offset = (n*dims[1]*dims[2]*dims[4]+f)*dims[5];
      int idx00 = offset + dims[4]*dims[5]*(y0*dims[2] + x0);
      int idx01 = offset + dims[4]*dims[5]*(y0*dims[2] + x1);
      int idx10 = offset + dims[4]*dims[5]*(y1*dims[2] + x0);
      int idx11 = offset + dims[4]*dims[5]*(y1*dims[2] + x1);

      const Dtype *im00 = input + idx00;
      const Dtype *im01 = input + idx01;
      const Dtype *im10 = input + idx10;
      const Dtype *im11 = input + idx11;

      const Dtype *grad = top_diff+index*dims[5];
      Dtype *im00_grad = input_diff + idx00;
      Dtype *im01_grad = input_diff + idx01;
      Dtype *im10_grad = input_diff + idx10;
      Dtype *im11_grad = input_diff + idx11;

      Dtype gx = 0;
      Dtype gy = 0;
      for (int c=0; c<dims[5]; c++) {
        Dtype g = *grad;
        atomicAdd(im00_grad, g*w00);
        atomicAdd(im01_grad, g*w01);
        atomicAdd(im10_grad, g*w10);
        atomicAdd(im11_grad, g*w11);

        gx += g*(wy0*(*im01 - *im00) + wy1*(*im11 - *im10));
        gy += g*(wx0*(*im10 - *im00) + wx1*(*im11 - *im01));

        grad++;
        im00++; im00_grad++;
        im01++; im01_grad++;
        im10++; im10_grad++;
        im11++; im11_grad++;
      }

      coords_diff[2*index] = gx;
      coords_diff[2*index+1] = gy;
    }
  }
}



bool BackProjectBackwardLauncher(const float *grad, const float* input,
  const float* coords, const int dim[6], float* inputs_diff,
  float* coords_diff, const Eigen::GpuDevice& d)
{
  cudaError_t err;
  const int kblock = 512;
  const int nthreads = dim[0]*dim[1]*dim[2]*dim[3]*dim[4];

  cudaMemset((void*) inputs_diff, 0, dim[0]*dim[1]*dim[2]*dim[4]*dim[5]*sizeof(float));
  cudaMemset((void*) coords_diff, 0, 2*dim[0]*dim[1]*dim[2]*dim[3]*dim[4]*sizeof(float));

  BackProjectBackward<<<(nthreads+kblock-1)/kblock, kblock, 0, d.stream()>>>(
    nthreads, grad, input, coords, dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], inputs_diff, coords_diff);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit(-1);
  }

  return d.ok();
}




#endif // GOOGLE_CUDA
