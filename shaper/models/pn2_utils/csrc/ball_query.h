#ifndef _BALL_QUERY
#define _BALL_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> BallQuery(
    const at::Tensor point,
    const at::Tensor centroid,
    const float radius,
    const int64_t num_neighbours);

#endif
