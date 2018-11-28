#ifndef _GATHER_KNN
#define _GATHER_KNN

#include <torch/torch.h>

// CUDA declarations
at::Tensor GatherKNNForward(
    at::Tensor input,
    at::Tensor index);

at::Tensor GatherKNNBackward(
    at::Tensor grad_output,
    at::Tensor index);

#endif
