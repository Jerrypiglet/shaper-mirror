// CUDA Implementation for interpolating 

#ifndef _INTERPOLATING_KERNEL
#define _INTERPOLATING_KERNEL

#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <vector>
#include <typeinfo>
#define MAX_THREADS uint64_t(512)

/*****************************
*****get the size of block*****
******************************/
inline uint64_t get_block(int64_t x) {
  int cnt = 0;
  x -= 1;
  while (x > 0) {
    x = x >> 1;
    cnt += 1;
  }
  return std::min(uint64_t(1) << cnt, MAX_THREADS);
}

/**********************************
*****kernel for searching point*****
***********************************/
/*
*Input:
*  n : int  The feature channels in output map
*  m : int  The feature channels in input map 
*  old_xyz : (B, N1, 3)
*  new_xyz : (B, N2, 3)
*
*Output
*  distance : (B, N2, 3)
*  id : (B, N2, 3)
*/

template <typename scalar_t, typename index_t>
__global__ void point_search_kernel(
    const int64_t n,  
    const int64_t m,
    const scalar_t *__restrict__ old_xyz,
    const scalar_t *__restrict__ new_xyz,
    scalar_t *__restrict__ distance,
    index_t *__restrict__ id){

    const int batch_index = blockIdx.x;
    old_xyz += batch_index * m * 3;
    new_xyz += batch_index * n * 3;
    id += batch_index * n * 3;
    distance += batch_index * n * 3;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index; i < n; i += stride){
        scalar_t x1 = new_xyz[i*3+0];
        scalar_t y1 = new_xyz[i*3+1];
        scalar_t z1 = new_xyz[i*3+2];

        scalar_t best1 = 1e40, best2 = 1e41, best3=1e40;
        int besti1 = 0, besti2 = 0, besti3 = 0;
        for (int k = 0; k < m; ++k) {
            scalar_t x2 = old_xyz[k*3+0];
            scalar_t y2 = old_xyz[k*3+1];
            scalar_t z2 = old_xyz[k*3+2];
            float d = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
            if(d < best1) {
                best3 = best2;
                besti3 = besti2;
                best2 = best1;
                besti2 = besti1;
                best1 = d;
                besti1 = k;
            } else if (d < best2) {
                best3 = best2;
                besti3 = besti2;
                best2 = d;
                besti2 = k;
            } else if (d < best3) {
                best3 = d;
                besti3 = k;
            }
        }
        distance[i * 3 + 0] = best1;
        distance[i * 3 + 1] = best2;
        distance[i * 3 + 2] = best3;

        id[i * 3 + 0] = besti1;
        id[i * 3 + 1] = besti2;
        id[i * 3 + 2] = besti3;
    }
}

/*
Input:
    npoint: k number of neighbors
    old_xyz: (B, 3, N2)
    new_xyz: (B, 3, N1)
Output:
    distance: (B, k, N1)
    id: (B, k, N1)
*/
std::vector<at::Tensor> PointSearch(
    const int64_t npoint,
    const at::Tensor old_xyz,
    const at::Tensor new_xyz) {

    const auto batch_size = new_xyz.size(0);
    const auto num_point_new = new_xyz.size(2);  // feature channels in output
    const auto num_point_old = old_xyz.size(2);
    const auto block = get_block(num_point_new);

    // Convert the Tensor Dimension
    auto old_xyz_trans = old_xyz.transpose(1, 2).contiguous();      // (B, N2, 3)
    auto new_xyz_trans = new_xyz.transpose(1, 2).contiguous();      // (B, N1, 3)
    auto id_trans = at::zeros({batch_size, num_point_new, 3}, new_xyz.type().toScalarType(at::kLong));
    auto distance_trans = at::zeros({batch_size, num_point_new, 3}, new_xyz.type());
    // std::cout << typeid(new_xyz_trans).name() << std::endl;

    AT_DISPATCH_FLOATING_TYPES(new_xyz.type(), "PointSearch", ([&] {
    point_search_kernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
        num_point_new,
        num_point_old,
        old_xyz_trans.data<scalar_t>(),
        new_xyz_trans.data<scalar_t>(),
        distance_trans.data<scalar_t>(),
        id_trans.data<int64_t>());
    }));

    auto distance = distance_trans.transpose(1, 2).contiguous();
    auto id = id_trans.transpose(1, 2).contiguous();

    return std::vector<at::Tensor>({distance, id});
}


/*********************************
***** Kernel for interpolate *****
*********************************/
/*
* Input 
*   c: num of channels in input
*   m: num of points in input
*   n: num of points in output 
*   point_feature: (B, c, m)
*   id: (B, n, 3)
*   weight: (B, n, 3)
*
* Output
*    out: (B, c, n)
*/
template<typename scalar_t, typename index_t>
__global__ void interpolate_kernel(
    const int64_t c,    // channels
    const int64_t m,    // old num of points
    const int64_t n,    // new num of points
    const scalar_t *__restrict__ point_feature,
    const index_t *__restrict__ id,
    const scalar_t *__restrict__ weight,
    scalar_t *__restrict__ out){
   
    const int batch_index = blockIdx.x;

    point_feature += batch_index * m * c;
    id += batch_index * n * 3;
    weight += batch_index * n * 3;
    out += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;

    for(int i = index; i < c * n; i += stride){
        const int l = i / n;
        const int j = i % n ;

        scalar_t w1 = weight[j*3+0];
        scalar_t w2 = weight[j*3+1];
        scalar_t w3 = weight[j*3+2];

        index_t i1 = id[j*3+0];
        index_t i2 = id[j*3+1];
        index_t i3 = id[j*3+2];

        out[i] = point_feature[l*m + i1]*w1 + point_feature[l*m + i2]*w2 + point_feature[l*m + i3]*w3;
    }
}

/*
Input:
    point_features: (B, c, m)
    id: (B, k, n)       k is the number of neighbor in PointSearch
    weight: (B, k, n)
*/
at::Tensor interpolate(
    const at::Tensor point_features,
    const at::Tensor id,
    const at::Tensor weight){

    const auto batch_size = id.size(0);
    const auto c = point_features.size(1);
    const auto m = point_features.size(2);
    const auto n = weight.size(2);
    const auto block = get_block(n);

    // Somehow the size of idx, and weight is not defined in the same way as point_features
    // Maybe in the future we need to uniform them
    auto out = at::zeros({batch_size, c, n}, point_features.type());
    auto id_trans = id.transpose(1, 2).contiguous();
    auto weight_trans = weight.transpose(1, 2).contiguous();

    AT_DISPATCH_FLOATING_TYPES(point_features.type(), "Interpolate", ([&] {
    interpolate_kernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
        c,
        m,
        n,
        point_features.data<scalar_t>(),
        id_trans.data<int64_t>(),
        weight_trans.data<scalar_t>(),
        out.data<scalar_t>());
    }));

    return out;
}  


/**********************************************************
***** Kernel for backpropagation in interpolate layer *****
**********************************************************/
/*
* Input
*    c: channels in input 
*    n: num of point in ouput
*    m: num of point in input 
*    grad_out: (B, c ,n)
*    id: (B, n, 3)
*    weight: (B, n, 3)
*
*  Ouput:
*    back_grad: (B, c, m)
*/
template<typename scalar_t, typename index_t>
__global__ void interpolate_backward_kernel(
   int64_t c,
   int64_t n,
   int64_t m,
   const scalar_t *__restrict__ grad_out,
   const index_t *__restrict__ id,
   const scalar_t *__restrict__ weight,
   scalar_t *__restrict__ back_grad) {
   
    const auto batch_index = blockIdx.x;
    grad_out += batch_index * n * c;
    id += batch_index * n * 3;
    weight += batch_index * n * 3;
    back_grad += batch_index * m * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;

    for (int i = index; i < c * n; i += stride) {
        const int l = i / n;
        const int j = i % n;

        scalar_t w1 = weight[j * 3 + 0];
        scalar_t w2 = weight[j * 3 + 1];
        scalar_t w3 = weight[j * 3 + 2];

        index_t i1 = id[j * 3 + 0];
        index_t i2 = id[j * 3 + 1];
        index_t i3 = id[j * 3 + 2];

        atomicAdd(back_grad + l*m + i1, grad_out[i]*w1);
        atomicAdd(back_grad + l*m + i2, grad_out[i]*w2);
        atomicAdd(back_grad + l*m + i3, grad_out[i]*w3);
    }
}

/*
Input:
    m: number of features in SA output layer
    grad_out:
    weight:
    id
Output:
    back_grad

*/
at::Tensor interpolateBackward(
    const int64_t m,
    const at::Tensor grad_out,
    const at::Tensor weight,
    const at::Tensor id){

    const auto batch_size = id.size(0);
    const auto n = id.size(1);
    const auto c = grad_out.size(1);
    const auto block = get_block(n);

    auto back_grad = at::zeros({batch_size, c, m}, grad_out.type());
    auto id_trans = id.transpose(1, 2).contiguous();
    auto weight_trans = weight.transpose(1, 2).contiguous();

    AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "Interpolate_back", ([&] {
    interpolate_backward_kernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
         c,
         n,
         m,
         grad_out.data<scalar_t>(),
         id_trans.data<int64_t>(),
         weight_trans.data<scalar_t>(),
         back_grad.data<scalar_t>());
    }));
    return back_grad;
}


#endif
