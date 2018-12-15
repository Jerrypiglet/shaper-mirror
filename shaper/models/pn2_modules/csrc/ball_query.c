#include <THC/THC.h>

#include "ball_query_gpu.h"

extern THCState *state;

int ball_query_wrapper(int b, int n, int m, float radius, int nsample,
		       THCudaTensor *new_xyz_tensor, THCudaTensor *xyz_tensor,
		       THCudaIntTensor *idx_tensor, THCudaIntTensor *pts_cnt_tensor) {

    const float *new_xyz = THCudaTensor_data(state, new_xyz_tensor);
    const float *xyz = THCudaTensor_data(state, xyz_tensor);
    int *idx = THCudaIntTensor_data(state, idx_tensor);
    int *pts_cnt = THCudaIntTensor_data(state, pts_cnt_tensor);


    cudaStream_t stream = THCState_getCurrentStream(state);

    query_ball_point_kernel_wrapper(b, n, m, radius, nsample, new_xyz, xyz, idx, pts_cnt,
				    stream);
    return 1;
}
