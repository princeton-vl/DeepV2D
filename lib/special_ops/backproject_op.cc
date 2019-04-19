/* Copyright 2015 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include <stdio.h>
#include <cfloat>
#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "work_sharder.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;
using GPUDevice = Eigen::GpuDevice;


typedef float T;

REGISTER_OP("BackProject")
    .Input("inputs: float32")
    .Input("coords: float32")
    .Output("output: float32");

REGISTER_OP("BackProjectGrad")
    .Input("input: float32")
    .Input("coords: float32")
    .Input("grad: float32")
    .Output("inputs_grad: float32")
    .Output("coords_grad: float32");


bool BackProjectForwardLauncher(const float* input, const float* coords,
  const int dim[6], float *top, const Eigen::GpuDevice& d);

bool BackProjectBackwardLauncher(const float *grad,
  const float* input, const float* coords, const int dim[6],
  float* inputs_diff, float* coords_diff, const Eigen::GpuDevice& d);


class BackProjectOp: public OpKernel {
 public:

  typedef Eigen::GpuDevice Device;
  explicit BackProjectOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& inputs = context->input(0);
    const Tensor& coords = context->input(1);
    const T* inputs_data = inputs.flat<T>().data();
    const T* coords_data = coords.flat<T>().data();

    int b = coords.dim_size(0);
    int h = coords.dim_size(1);
    int w = coords.dim_size(2);
    int s = coords.dim_size(3);
    int f = coords.dim_size(4);
    int c = inputs.dim_size(4);

    int dims[6];
    dims[0] = b;
    dims[1] = h;
    dims[2] = w;
    dims[3] = s;
    dims[4] = f;
    dims[5] = c;

    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 6, &output_shape);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    T* top_data = output_tensor->template flat<T>().data();

    BackProjectForwardLauncher(inputs_data, coords_data, dims, top_data,
      context->eigen_device<Eigen::GpuDevice>());
    }
};



class BackProjectGradOp: public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit BackProjectGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override
  {

    // Grab the input tensor
    const Tensor& inputs = context->input(0);
    const Tensor& coords = context->input(1);
    const Tensor& top_diff = context->input(2);

    const T* inputs_data = inputs.flat<T>().data();
    const T* coords_data = coords.flat<T>().data();
    const T* top_diff_data = top_diff.flat<T>().data();

    int b = coords.dim_size(0);
    int h = coords.dim_size(1);
    int w = coords.dim_size(2);
    int s = coords.dim_size(3);
    int f = coords.dim_size(4);
    int c = inputs.dim_size(4);

    int inputs_grad_dims[5];
    inputs_grad_dims[0] = b;
    inputs_grad_dims[1] = h;
    inputs_grad_dims[2] = w;
    inputs_grad_dims[3] = f;
    inputs_grad_dims[4] = c;

    int coords_grad_dims[6];
    coords_grad_dims[0] = b;
    coords_grad_dims[1] = h;
    coords_grad_dims[2] = w;
    coords_grad_dims[3] = s;
    coords_grad_dims[4] = f;
    coords_grad_dims[5] = 2;

    int dims[6];
    dims[0] = b;
    dims[1] = h;
    dims[2] = w;
    dims[3] = s;
    dims[4] = f;
    dims[5] = c;

    TensorShape coords_grad_shape, inputs_grad_shape;
    TensorShapeUtils::MakeShape(inputs_grad_dims, 5, &inputs_grad_shape);
    TensorShapeUtils::MakeShape(coords_grad_dims, 6, &coords_grad_shape);

    Tensor* inputs_grad = NULL;
    Tensor* coords_grad = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, inputs_grad_shape, &inputs_grad));
    OP_REQUIRES_OK(context, context->allocate_output(1, coords_grad_shape, &coords_grad));

    T* inputs_grad_data = inputs_grad->flat<T>().data();
    T* coords_grad_data = coords_grad->flat<T>().data();

    BackProjectBackwardLauncher(top_diff_data, inputs_data, coords_data, dims,
      inputs_grad_data, coords_grad_data, context->eigen_device<Eigen::GpuDevice>());
  }
};

// compute gradient
REGISTER_KERNEL_BUILDER(Name("BackProject").Device(DEVICE_GPU), BackProjectOp);
REGISTER_KERNEL_BUILDER(Name("BackProjectGrad").Device(DEVICE_GPU), BackProjectGradOp);
