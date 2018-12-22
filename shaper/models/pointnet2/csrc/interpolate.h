#ifndef _INTERPOLATE
#define _INTERPOLATE

#include <torch/extension.h>

//CUDA declarations
std::vector<at::Tensor> PointSearch(
    const int64_t npoint,
    const at::Tensor old_xyz,       // i.e. xyz2, the output of SA layers
    const at::Tensor new_xyz);      // i.e. xyz1, the input of SA layers

at::Tensor interpolate(
    const at::Tensor point_features,
    const at::Tensor id,
    const at::Tensor weight);

at::Tensor interpolateBackward(
    const int64_t m,
    const at::Tensor grad_out,
    const at::Tensor weight,
    const at::Tensor id);

#endif
