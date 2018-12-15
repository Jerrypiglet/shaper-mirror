#include <ATen/ATen.h>
// #include <THC/THC.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_grad_output(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_EQ(x, y) AT_CHECK(x == y, #x " does not equal to " #y)

#define DIM 4

using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;

/* 
Forward interface
input: (B, C, N)
index: (B, N, K)
output: (B, N, K, C)
*/
at::Tensor gather_cuda_forward(
    const at::Tensor input,
    const at::Tensor index) { 
  const auto batch_size = input.size(0);
  const auto channels = input.size(1);
  const auto num_inst = input.size(2);
  const auto k = index.size(2);

  // Sanity check
  CHECK_CUDA(input);
  CHECK_CUDA(index);
  CHECK_EQ(input.dim(), DIM - 1);
  CHECK_EQ(index.dim(), DIM - 1);
  CHECK_EQ(index.size(0), batch_size);
  CHECK_EQ(index.size(1), num_inst);

  // Expand input and index
  auto input_trans = input.transpose(1, 2).contiguous();  // (B, N, C)
  auto input_expand = input_trans.unsqueeze(1).expand({batch_size, num_inst, num_inst, channels});  // (B, N, N, C)
  auto index_expand = index.unsqueeze(DIM - 1).expand({batch_size, num_inst, k, channels});  // (B, N, K, C)

  auto output_tmp = input_expand.gather(2, index_expand);  // (B, N, K, C)
  auto output = output_tmp.permute({0, 3, 1, 2}).contiguous(); // (B, C, N, K)
  
  return output;
}

/* Backward Kernel */
template <typename scalar_t, typename IndexType>
__global__ void gather_cuda_backward_kernel(
    const TensorInfo<scalar_t, IndexType> grad_input,
    const TensorInfo<scalar_t, IndexType> grad_output,
    const TensorInfo<int64_t, IndexType> index,
    const IndexType totalElements) {
  // IndexType batch_size = grad_output.sizes[0];
  IndexType channels = grad_output.sizes[1];
  IndexType num_inst = grad_output.sizes[2];
  IndexType k = grad_output.sizes[3];
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    // Compute offsets
    IndexType linearId_tmp = linearId;
    IndexType k_offset = linearId_tmp % k;
    linearId_tmp /= k;
    IndexType inst_offset = linearId_tmp % num_inst;
    linearId_tmp /= num_inst;
    IndexType channel_offset = linearId_tmp % channels;
    IndexType batch_offset = linearId_tmp / channels;
    
    IndexType srcOffset = k_offset * grad_output.strides[3]
      + inst_offset * grad_output.strides[2]
      + channel_offset * grad_output.strides[1]
      + batch_offset * grad_output.strides[0];

    IndexType tensorOffset = channel_offset * grad_input.strides[1]
      + batch_offset * grad_input.strides[0];
    
    IndexType indexOffset = k_offset * index.strides[2]
      + inst_offset * index.strides[1]
      + batch_offset * index.strides[0];

    int64_t indexValue = index.data[indexOffset] - TH_INDEX_BASE;
    assert(indexValue >= 0 && indexValue < num_inst);
    tensorOffset += indexValue * grad_input.strides[2];
    atomicAdd(&grad_input.data[tensorOffset], grad_output.data[srcOffset]);
  }
}

/* 
Backward interface
grad_output: (B, C, N, K)
index: (B, N, K)
grad_input: (B, C, N)
*/
at::Tensor gather_cuda_backward(
    const at::Tensor grad_output,
    const at::Tensor index) {
  const auto batch_size = grad_output.size(0);
  const auto channels = grad_output.size(1);
  const auto num_inst = grad_output.size(2);
  const auto k = grad_output.size(3);

  // Sanity check
  CHECK_CUDA(grad_output);
  CHECK_CUDA(index);
  CHECK_EQ(grad_output.dim(), DIM);
  CHECK_EQ(index.dim(), DIM - 1);
  CHECK_EQ(index.size(0), batch_size);
  CHECK_EQ(index.size(1), num_inst);
  CHECK_EQ(index.size(2), k);

  // Allocate new space for output
  auto grad_input = at::zeros({batch_size, channels, num_inst}, grad_output.type());
  CHECK_CUDA(grad_input);
  CHECK_CONTIGUOUS(grad_input);

  // Calculate grids and blocks for kernels 
  const auto totalElements = grad_output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  int curDevice = at::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");
  // printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
  // printf("Block: %d, %d, %d\n", block.x, block.y, block.z);

  // printf("grad_input.strides(): %ld, %ld, %ld\n", grad_input.stride(0), grad_input.stride(1), grad_input.stride(2));
  // printf("grad_output.strides(): %ld, %ld, %ld, %ld\n", grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3));
  // printf("grad_output.sizes(): %ld, %ld, %ld, %ld\n", grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3));
  // printf("totalElements: %ld\n", totalElements);

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "gather_cuda_backward", ([&] {
    auto gradInputInfo = getTensorInfo<scalar_t, uint64_t>(grad_input);
    auto gradOutputInfo = getTensorInfo<scalar_t, uint64_t>(grad_output);
    auto IndexInfo = getTensorInfo<int64_t, uint64_t>(index);
    gather_cuda_backward_kernel<scalar_t><<<grid, block>>>(
        gradInputInfo,
        gradOutputInfo,
        IndexInfo,
        (uint64_t)totalElements);
  }));

  return grad_input;
}
