#ifndef _SAMPLING
#define _SAMPLING

#include <torch/torch.h>

// CUDA declarations
at::Tensor FarthestPointSample(
    at::Tensor point,
    int64_t num_centroids);

#endif
