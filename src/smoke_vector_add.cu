// smoke_vector_add.cu — minimal CUDA kernel for sm capability smoke test
// Purpose: verify PyTorch cpp_extension can build + run on the current GPU
//          (common to Spark sm_121 / B200 sm_100) before writing the FP8/FP4 kernels.

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.numel() == b.numel(), "shape mismatch");

    auto c = torch::empty_like(a);
    int n = a.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    vector_add_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                       c.data_ptr<float>(), n);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector add (CUDA)");
}
