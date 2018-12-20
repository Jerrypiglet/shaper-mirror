#ifndef _INTERPOLATE
#define _INTERPOLATE

#include <torch/extension.h>

//CUDA declarations
void PointSearch(
    const int64_t m, // old map
    const int64_t n,  // new map
    const int64_t npoint,
    const at::Tensor old_xyz,
    const at::Tensor new_xyz,
    at::Tensor& id,
    at::Tensor& dist);

at::Tensor interpolate(
    const int64_t m, // number of points in old map
    const int64_t n,  // number of points in new map
    const int64_t c, // channels
    const at::Tensor id,
    const at::Tensor point_features,
    const at::Tensor weight);

at::Tensor interpolate_grad(
    const int64_t c,
    const int64_t m,
    const int64_t n,
    const at::Tensor grad_out,
    const at::Tensor weight,
    const at::Tensor id);

#endif
