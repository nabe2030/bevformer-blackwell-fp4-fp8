// smoke_fp4_cvt.cu — FP4 (e2m1) round-trip smoke for sm_121 / sm_121a / sm_120a / sm_120.
//
// Purpose: confirm cvt instruction (float ↔ __nv_fp4_e2m1) compiles and runs.
// NO tensor core (tcgen05 / block-scaled MMA). Pure cvt path only.
// Vector variant __nv_fp4x4_e2m1 also tested (4 floats → 4 FP4 packed → 4 floats).

#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// scalar round-trip: float → __nv_fp4_e2m1 → float
__global__ void fp4_roundtrip_scalar_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_fp4_e2m1 q(in[i]);             // float → FP4 e2m1 (quantize, RN saturate)
        out[i] = static_cast<float>(q);      // FP4 → float (exact dequant of the FP4 grid value)
    }
}

// vec4 round-trip: float4 → __nv_fp4x4_e2m1 (4 packed) → float4
__global__ void fp4_roundtrip_vec4_kernel(const float4 *in, float4 *out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        __nv_fp4x4_e2m1 q(in[i]);            // float4 → 4 packed FP4 (8 bits total)
        out[i] = static_cast<float4>(q);     // 4 packed FP4 → float4
    }
}

torch::Tensor fp4_roundtrip_scalar(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(),
                "x must be CUDA float32 contiguous");
    auto y = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    fp4_roundtrip_scalar_kernel<<<grid, block>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scalar kernel failed: ", cudaGetErrorString(err));
    return y;
}

torch::Tensor fp4_roundtrip_vec4(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(),
                "x must be CUDA float32 contiguous");
    TORCH_CHECK(x.numel() % 4 == 0, "x.numel() must be multiple of 4");
    auto y = torch::empty_like(x);
    int n4 = x.numel() / 4;
    int block = 256;
    int grid = (n4 + block - 1) / block;
    fp4_roundtrip_vec4_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n4);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "vec4 kernel failed: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4_roundtrip_scalar", &fp4_roundtrip_scalar, "FP4 e2m1 round-trip (scalar)");
    m.def("fp4_roundtrip_vec4",   &fp4_roundtrip_vec4,   "FP4 e2m1 round-trip (vec4)");
}
