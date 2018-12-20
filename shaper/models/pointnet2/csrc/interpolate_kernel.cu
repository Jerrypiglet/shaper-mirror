// CUDA Implementation for interpolating 

#ifndef _INTERPOLATING_KERNEL
#define _INTERPOLATING_KERNEL

#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <vector>
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
template <typename scalar_t, typename index_t>
__global__ void point_search_kernel(
    const int64_t n,  // new num_point
    const int64_t m, // old num_point
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

void PointSearch(
    const int64_t m, // old map
    const int64_t n,  // new map
    const int64_t npoint,
    const at::Tensor old_xyz,
    const at::Tensor new_xyz,
    at::Tensor& distance,
    at::Tensor& id) {
   const auto batch_size = new_xyz.size(0);
   const auto num_point = new_xyz.size(2);

   const auto block = get_block(num_point);
   
   AT_DISPATCH_FLOATING_TYPES(new_xyz.type(), "PointSearch", ([&] {
    point_search_kernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
         n,
	 m,
	 old_xyz.data<scalar_t>(),
         new_xyz.data<scalar_t>(),
         distance.data<scalar_t>(),
         id.data<int64_t>());
   }));
}


template<typename scalar_t, typename index_t>
__global__ void interpolate_kernel(
    const int64_t c, // channels
    const int64_t m, // old num of points
    const int64_t n,  // new num of points 
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

at::Tensor interpolate(
    const int64_t m, // number of points in old map
    const int64_t n,  // number of points in new map
    const int64_t c, // channels
    const at::Tensor id,
    const at::Tensor point_features,
    const at::Tensor weight){
   const auto batch_size = id.size(0);
   const auto num_point = id.size(2);

   const auto block = get_block(num_point);
   auto out = at::zeros({batch_size, c, num_point}, id.type().toScalarType(at::kLong));
   AT_DISPATCH_FLOATING_TYPES(id.type(), "Interpolate", ([&] {
    interpolate_kernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
         c,
         m,
	 n,
	 point_features.data<scalar_t>(),
         id.data<int64_t>(),
         weight.data<scalar_t>(),
         out.data<scalar_t>());
   }));

   return out;
}  

template<typename scalar_t, typename index_t>
__global__ void interpolate_grad_kernel(
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

at::Tensor interpolate_grad(
    const int64_t c,
    const int64_t m,
    const int64_t n,
    const at::Tensor grad_out,
    const at::Tensor weight,
    const at::Tensor id){
   const auto batch_size = id.size(0);
   const auto num_point = id.size(2);
   
   auto back_grad = at::zeros({batch_size, c, m}, id.type().toScalarType(at::kLong));
   
   const auto block = get_block(num_point);
   auto out = at::zeros({batch_size, c, num_point}, id.type().toScalarType(at::kLong));
   AT_DISPATCH_FLOATING_TYPES(id.type(), "Interpolate_back", ([&] {
    interpolate_grad_kernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
         c,
         n,
         m,
         grad_out.data<scalar_t>(),
         id.data<int64_t>(),
         weight.data<scalar_t>(),
         back_grad.data<scalar_t>());
   }));
   return back_grad;
}


#endif
