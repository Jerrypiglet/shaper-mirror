#ifndef _GROUP_POINTS
#define _GROUP_POINTS

#include <torch/torch.h>

// CUDA declarations
at::Tensor GroupPointsForward(
    at::Tensor input,
    at::Tensor index);

at::Tensor GroupPointsBackward(
    at::Tensor grad_output,
    at::Tensor index,
    int64_t num_points);

#endif
