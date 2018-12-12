#ifndef _BALL_QUERY
#define _BALL_QUERY

#include <vector>
#include <torch/torch.h>

std::vector<at::Tensor> BallQuery(
    at::Tensor point,
    at::Tensor centroid,
    float radius,
    int64_t num_neighbours);

#endif
