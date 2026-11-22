// ms_deform_attn_fp4_kernel.cu
// SPDX-License-Identifier: Apache-2.0
//
// Derived from mmcv 1.7.2 (https://github.com/open-mmlab/mmcv),
//   mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
//   Copyright (c) OpenMMLab. All rights reserved. Licensed under the Apache License 2.0.
// which carries the notice:
//   Deformable DETR. Copyright (c) 2020 SenseTime. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0.
// Modifications (2026, the bevformer-blackwell-fp4-fp8 authors): value read as packed FP4 e2m1; FP4 quantize+pack kernel.
//
// Stage 1-B: forward inference, FP4 (e2m1) value load.
//
// Build flag (settled by the smoke test): -gencode=arch=compute_121,code=sm_121
//
// Storage decision:
//   PyTorch's `torch.float4_e2m1fn_x2` is declared but has no usable ops
//   (zeros/.to() raise NotImplementedError on cuda).
//   → We use `torch::kUInt8` tensor of shape (bs, ss, h, channels/2) and
//     reinterpret_cast to `__nv_fp4x2_e2m1*` in the kernel. Each byte packs
//     2 FP4 values: low nibble = channel 2k, high nibble = channel 2k+1.
//
// Design fidelity (mirrors FP8 path):
//   - value loaded as packed __nv_fp4x2_e2m1*, nibble extracted per thread
//   - 1 thread = 1 (b, q, m, c) — same layout as FP8
//   - per-load dequant via __nv_fp4x2 → __half2 → float (intrinsics, no
//     __half operators since __CUDA_NO_HALF_OPERATORS__ is set by PyTorch)
//   - value_scale applied at output (linearity, mirrors FP8)
//
// Also provides a quantize+pack helper kernel so Python can produce the
// packed uint8 tensor without relying on PyTorch's FP4 conversion.

#include <cuda_fp4.h>
#include <cuda_fp16.h>      // for __half2 intrinsics
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// Quantize + pack: float32 → packed FP4 e2m1 (2-per-byte uint8)
//
// Each thread handles ONE pair (= 2 channels), producing 1 byte:
//   low nibble (bits 0-3) = FP4(in[2k])
//   high nibble (bits 4-7) = FP4(in[2k+1])
// Uses `__nv_fp4x2_e2m1(float2)` constructor (the float4 version worked in the
// smoke test; the vec2 version is assumed to work the same way).
// ----------------------------------------------------------------------------
__global__ void quantize_pack_fp4_kernel(
    const float *in,
    uint8_t *out,
    float inv_scale,
    int n_pairs)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx < n_pairs) {
        const float v0 = in[pair_idx * 2]     * inv_scale;
        const float v1 = in[pair_idx * 2 + 1] * inv_scale;
        const float2 f2 = make_float2(v0, v1);
        const __nv_fp4x2_e2m1 packed(f2);
        // __nv_fp4x2_e2m1 is 1 byte aligned (storage = uint8_t containing 2 nibbles).
        out[pair_idx] = *reinterpret_cast<const uint8_t*>(&packed);
    }
}

torch::Tensor quantize_pack_fp4(torch::Tensor in, double value_scale) {
    TORCH_CHECK(in.is_cuda() && in.scalar_type() == torch::kFloat32 && in.is_contiguous(),
                "in must be CUDA float32 contiguous");
    TORCH_CHECK(in.numel() % 2 == 0, "in.numel() must be even (packed FP4 needs pairs)");

    auto sizes = in.sizes().vec();
    sizes.back() /= 2;
    auto out = torch::empty(sizes, torch::dtype(torch::kUInt8).device(in.device()));

    const int n_pairs = in.numel() / 2;
    const int block = 256;
    const int grid = (n_pairs + block - 1) / block;
    const float inv_scale = 1.0f / static_cast<float>(value_scale);
    quantize_pack_fp4_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<uint8_t>(),
        inv_scale,
        n_pairs);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "quantize_pack kernel failed: ", cudaGetErrorString(err));
    return out;
}

// ----------------------------------------------------------------------------
// BLI helper (packed FP4 4-neighbor → float dequant → float interp)
//
// `bottom_data_pair` is a pointer to packed pair-bytes. Stride is in PAIRS:
//   w_stride = nheads * channels_pairs,  channels_pairs = channels/2
// For a given (m, c), the byte holding our nibble is at pair_idx = m*channels_pairs + c/2,
// and within that byte, low nibble (c%2==0) or high nibble (c%2==1).
//
// We decode via __nv_fp4x2_e2m1 → __half2 (operator) → __half (intrinsic) → float.
// __half operators are disabled by PyTorch defines, but __low2half/__high2half/
// __half2float are intrinsics (functions), so they work without operator support.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float load_fp4_nibble(
    const __nv_fp4x2_e2m1 *pair_data,
    int pair_idx,
    int c_half)
{
    const __nv_fp4x2_e2m1 pair = pair_data[pair_idx];
    const __half2 hp = static_cast<__half2>(pair);
    const __half h = c_half ? __high2half(hp) : __low2half(hp);
    return __half2float(h);
}

__device__ __forceinline__ float ms_deform_attn_im2col_bilinear_fp4(
    const __nv_fp4x2_e2m1 *bottom_data_pair,
    const int height, const int width,
    const int nheads, const int channels,
    const float h, const float w,
    const int m, const int c)
{
    const int h_low = floorf(h);
    const int w_low = floorf(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h - (float)h_low;
    const float lw = w - (float)w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;

    const int channels_pairs = channels >> 1;            // channels/2
    const int w_stride = nheads * channels_pairs;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels_pairs + (c >> 1);  // pair index for (m, c)
    const int c_half = c & 1;                            // nibble selector

    float v1 = 0.0f;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = load_fp4_nibble(bottom_data_pair, ptr1, c_half);
    }
    float v2 = 0.0f;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = load_fp4_nibble(bottom_data_pair, ptr2, c_half);
    }
    float v3 = 0.0f;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = load_fp4_nibble(bottom_data_pair, ptr3, c_half);
    }
    float v4 = 0.0f;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = load_fp4_nibble(bottom_data_pair, ptr4, c_half);
    }

    const float w1 = hh * hw;
    const float w2 = hh * lw;
    const float w3 = lh * hw;
    const float w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// ----------------------------------------------------------------------------
// forward kernel — 1 thread = 1 (b, q, m, c) (same layout as FP8)
// ----------------------------------------------------------------------------
__global__ void ms_deformable_im2col_gpu_kernel_fp4(
    const int n,
    const __nv_fp4x2_e2m1 *data_value_pair,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const float *data_sampling_loc,
    const float *data_attn_weight,
    const int batch_size, const int spatial_size,
    const int num_heads, const int channels,
    const int num_levels, const int num_query,
    const int num_point,
    float *data_col)
{
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < n; index += stride) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        _temp /= num_query;
        const int b_col = _temp;

        float *data_col_ptr = data_col + index;
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int channels_pairs = channels >> 1;
        const int qid_stride_pairs = num_heads * channels_pairs;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride_pairs;
        float col = 0.0f;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const __nv_fp4x2_e2m1 *data_value_ptr =
                data_value_pair + (data_value_ptr_init_offset + level_start_id * qid_stride_pairs);
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const float loc_w = data_sampling_loc[data_loc_w_ptr];
                const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const float weight = data_attn_weight[data_weight_ptr];

                const float h_im = loc_h * spatial_h - 0.5f;
                const float w_im = loc_w * spatial_w - 0.5f;

                if (h_im > -1.0f && w_im > -1.0f &&
                    h_im < (float)spatial_h && w_im < (float)spatial_w) {
                    col += ms_deform_attn_im2col_bilinear_fp4(
                        data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                        h_im, w_im, m_col, c_col) * weight;
                }
                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
            }
        }
        *data_col_ptr = col;
    }
}

// ----------------------------------------------------------------------------
// Binding — ms_deform_attn_fp4_forward
//
// value_packed: uint8 tensor, shape (bs, sum_l H_l*W_l, num_heads, channels/2)
//               (each byte = 2 packed FP4: low nibble = ch 2k, high nibble = ch 2k+1)
// channels  : python int (effective channel count = 2 × value_packed.size(3))
// value_scale: python float (per-tensor scale; applied at output via linearity)
// Returns:    (bs, num_query, num_heads * channels) float32
// ----------------------------------------------------------------------------
torch::Tensor ms_deform_attn_fp4_forward(
    const torch::Tensor &value_packed,
    int64_t channels,                  // effective channels (= 2 × value_packed.size(3))
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    double value_scale)
{
    TORCH_CHECK(value_packed.is_cuda(), "value_packed must be CUDA");
    TORCH_CHECK(value_packed.scalar_type() == torch::kUInt8,
                "value_packed must be uint8 (storage for packed FP4)");
    TORCH_CHECK(value_packed.is_contiguous(), "value_packed must be contiguous");
    TORCH_CHECK(value_packed.dim() == 4, "value_packed must be 4-D (bs, ss, h, c/2)");
    TORCH_CHECK(channels % 2 == 0, "channels must be even (packed FP4)");
    TORCH_CHECK(value_packed.size(3) * 2 == channels,
                "value_packed.size(3) must equal channels/2");

    TORCH_CHECK(spatial_shapes.is_cuda() && spatial_shapes.scalar_type() == torch::kInt64);
    TORCH_CHECK(spatial_shapes.is_contiguous());
    TORCH_CHECK(level_start_index.is_cuda() && level_start_index.scalar_type() == torch::kInt64);
    TORCH_CHECK(level_start_index.is_contiguous());
    TORCH_CHECK(sampling_loc.is_cuda() && sampling_loc.scalar_type() == torch::kFloat32);
    TORCH_CHECK(sampling_loc.is_contiguous());
    TORCH_CHECK(attn_weight.is_cuda() && attn_weight.scalar_type() == torch::kFloat32);
    TORCH_CHECK(attn_weight.is_contiguous());

    const int bs           = value_packed.size(0);
    const int spatial_size = value_packed.size(1);
    const int num_heads    = value_packed.size(2);
    const int num_levels   = spatial_shapes.size(0);
    const int num_query    = sampling_loc.size(1);
    const int num_point    = sampling_loc.size(4);

    auto output = torch::zeros({bs, num_query, num_heads, (int64_t)channels},
                               torch::dtype(torch::kFloat32).device(value_packed.device()));

    const int num_kernels = bs * num_query * num_heads * channels;
    const int block = 256;
    const int grid  = (num_kernels + block - 1) / block;

    ms_deformable_im2col_gpu_kernel_fp4<<<grid, block>>>(
        num_kernels,
        reinterpret_cast<const __nv_fp4x2_e2m1*>(value_packed.data_ptr<uint8_t>()),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        bs, spatial_size, num_heads, (int)channels,
        num_levels, num_query, num_point,
        output.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));

    output.mul_(static_cast<float>(value_scale));

    return output.view({bs, num_query, num_heads * (int64_t)channels});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_pack_fp4", &quantize_pack_fp4,
          "FP32 → packed FP4 e2m1 (uint8 storage, 2-per-byte)");
    m.def("ms_deform_attn_fp4_forward", &ms_deform_attn_fp4_forward,
          "MS Deform Attn forward (packed FP4 e2m1 value, per-tensor scale, float32 output)");
}
