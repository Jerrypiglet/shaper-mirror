#ifndef _SAMPLING
#define _SAMPLING

#include <torch/extension.h>

// CUDA declarations
at::Tensor FarthestPointSample(
    const at::Tensor mdist,
    const at::Tensor pos,
    const at::Tensor distance,
    const at::Tensor point,
    const int64_t num_centroids);

#endif
