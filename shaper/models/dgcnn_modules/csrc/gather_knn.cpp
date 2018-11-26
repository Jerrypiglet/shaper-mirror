/* CUDA Implementation for efficient scatter*/
#include <torch/torch.h>

// CUDA declarations
at::Tensor gather_cuda_forward(
    at::Tensor input,
    at::Tensor index);

at::Tensor gather_cuda_backward(
    at::Tensor grad_output,
    at::Tensor index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gather_cuda_forward, "Gather forward (CUDA)");
  m.def("backward", &gather_cuda_backward, "Gather backward (CUDA)");
}
