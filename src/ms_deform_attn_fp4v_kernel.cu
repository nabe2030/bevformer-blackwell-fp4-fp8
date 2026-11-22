// ms_deform_attn_fp4v_kernel.cu
// SPDX-License-Identifier: Apache-2.0
//
// Derived from mmcv 1.7.2 (https://github.com/open-mmlab/mmcv),
//   mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
//   Copyright (c) OpenMMLab. All rights reserved. Licensed under the Apache License 2.0.
// which carries the notice:
//   Deformable DETR. Copyright (c) 2020 SenseTime. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0.
// Modifications (2026, the bevformer-blackwell-fp4-fp8 authors): value read as packed FP4 e2m1 with vector decode.
//
// Stage 1-B'-a: FP4 forward, vector decode + 1 thread = 2 channels.
//
// Motivation: 1-B (scalar) lost to FP8 because per-load dequant was 4×
// instructions (cvt + nibble select + half→float) + branch divergence.
// 1-B'-a fix: each thread reads 1 byte (= 2 packed FP4) and USES BOTH halves
// — no select, no branch, 2 outputs per thread.
//
// Build flag: -gencode=arch=compute_121,code=sm_121 (same as 1-A/1-B)
//
// Storage (same as 1-B):
//   value_packed : uint8 tensor, shape (bs, ss, num_heads, channels/2)
//                  each byte = 2 packed FP4 (low nibble = ch 2k, high = ch 2k+1)
//
// Thread layout:
//   1 thread = 1 (b, q, m, c_pair)  where c_pair indexes 2 consecutive channels
//   num_kernels = bs * num_query * num_heads * (channels / 2)
//   each thread writes 2 outputs: (..., 2*c_pair) and (..., 2*c_pair + 1)
//
// Decode:
//   per byte read: __nv_fp4x2_e2m1 → __half2 (via existing operator) →
//                  __low2half/__high2half intrinsics → __half2float
//   Note: same cvt cost as 1-B per byte, but USE BOTH halves (no select).
//   Per thread (2 outputs): 4 neighbor bytes × 1 cvt + 2 half-extract + 2 conv
//                          = 4 cvt + 8 half→float = 12 ops for both channels
//   Per output: ~6 ops (vs 1-B's ~12 ops/output), and no branch divergence.

#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// Quantize + pack: float32 → packed FP4 e2m1 (2-per-byte uint8)
// (Identical to 1-B's helper; copied here so the fp4v module is self-contained)
// ----------------------------------------------------------------------------
__global__ void quantize_pack_fp4v_kernel(
    const float *in, uint8_t *out, float inv_scale, int n_pairs)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx < n_pairs) {
        const float v0 = in[pair_idx * 2]     * inv_scale;
        const float v1 = in[pair_idx * 2 + 1] * inv_scale;
        const float2 f2 = make_float2(v0, v1);
        const __nv_fp4x2_e2m1 packed(f2);
        out[pair_idx] = *reinterpret_cast<const uint8_t*>(&packed);
    }
}

torch::Tensor quantize_pack_fp4v(torch::Tensor in, double value_scale) {
    TORCH_CHECK(in.is_cuda() && in.scalar_type() == torch::kFloat32 && in.is_contiguous(),
                "in must be CUDA float32 contiguous");
    TORCH_CHECK(in.numel() % 2 == 0, "in.numel() must be even");
    auto sizes = in.sizes().vec();
    sizes.back() /= 2;
    auto out = torch::empty(sizes, torch::dtype(torch::kUInt8).device(in.device()));
    const int n_pairs = in.numel() / 2;
    const int block = 256;
    const int grid = (n_pairs + block - 1) / block;
    const float inv_scale = 1.0f / static_cast<float>(value_scale);
    quantize_pack_fp4v_kernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<uint8_t>(), inv_scale, n_pairs);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "quantize_pack kernel failed: ", cudaGetErrorString(err));
    return out;
}

// ----------------------------------------------------------------------------
// decode_fp4_pair: 1 packed byte → 2 floats (vlo, vhi)
//   USES BOTH halves of the half2 — no select, no branch.
//   Same cvt cost as 1-B per byte, but produces 2 useful outputs.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void decode_fp4_pair(
    const __nv_fp4x2_e2m1 pair, float &vlo, float &vhi)
{
    const __half2 h2 = static_cast<__half2>(pair);
    vlo = __half2float(__low2half(h2));
    vhi = __half2float(__high2half(h2));
}

// ----------------------------------------------------------------------------
// BLI helper (vector): 4 neighbor packed bytes → 2 interpolated floats
//   c_pair: pair index (= original channel index / 2)
//   Outputs: val_lo (for ch = 2*c_pair), val_hi (for ch = 2*c_pair + 1)
// Same boundary check / interp weights as 1-B; key change is per-byte decode
// produces BOTH channels' values via decode_fp4_pair.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void ms_deform_attn_im2col_bilinear_fp4v(
    const __nv_fp4x2_e2m1 *bottom_data_pair,
    const int height, const int width,
    const int nheads, const int channels,
    const float h, const float w,
    const int m, const int c_pair,
    float &val_lo, float &val_hi)
{
    const int h_low = floorf(h);
    const int w_low = floorf(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h - (float)h_low;
    const float lw = w - (float)w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;

    const int channels_pairs = channels >> 1;
    const int w_stride = nheads * channels_pairs;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels_pairs + c_pair;

    float v1_lo = 0.0f, v1_hi = 0.0f;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        decode_fp4_pair(bottom_data_pair[ptr1], v1_lo, v1_hi);
    }
    float v2_lo = 0.0f, v2_hi = 0.0f;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        decode_fp4_pair(bottom_data_pair[ptr2], v2_lo, v2_hi);
    }
    float v3_lo = 0.0f, v3_hi = 0.0f;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        decode_fp4_pair(bottom_data_pair[ptr3], v3_lo, v3_hi);
    }
    float v4_lo = 0.0f, v4_hi = 0.0f;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        decode_fp4_pair(bottom_data_pair[ptr4], v4_lo, v4_hi);
    }

    const float w1 = hh * hw;
    const float w2 = hh * lw;
    const float w3 = lh * hw;
    const float w4 = lh * lw;
    val_lo = w1 * v1_lo + w2 * v2_lo + w3 * v3_lo + w4 * v4_lo;
    val_hi = w1 * v1_hi + w2 * v2_hi + w3 * v3_hi + w4 * v4_hi;
}

// ----------------------------------------------------------------------------
// forward kernel — 1 thread = 1 (b, q, m, c_pair), writes 2 outputs
// num_kernels = bs * q * m * (channels/2)  — half of 1-B's
// ----------------------------------------------------------------------------
__global__ void ms_deformable_im2col_gpu_kernel_fp4v(
    const int n_pairs,
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
    const int channels_pairs = channels >> 1;
    const int stride = blockDim.x * gridDim.x;
    for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
         pair_idx < n_pairs; pair_idx += stride) {
        int _temp = pair_idx;
        const int c_pair = _temp % channels_pairs;
        _temp /= channels_pairs;
        const int sampling_index = _temp;     // (b*q*m + q*m + m)
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        _temp /= num_query;
        const int b_col = _temp;

        // Output positions: 2 consecutive channels (2*c_pair, 2*c_pair+1)
        // Linear output index in (b, q, m, c) layout: same as 1-B but doubled
        const int out_idx_lo = pair_idx * 2;       // c = 2*c_pair (even channel)
        const int out_idx_hi = pair_idx * 2 + 1;   // c = 2*c_pair + 1
        float *out_lo_ptr = data_col + out_idx_lo;
        float *out_hi_ptr = data_col + out_idx_hi;

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int qid_stride_pairs = num_heads * channels_pairs;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride_pairs;
        float col_lo = 0.0f;
        float col_hi = 0.0f;

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
                    float vl, vh;
                    ms_deform_attn_im2col_bilinear_fp4v(
                        data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                        h_im, w_im, m_col, c_pair, vl, vh);
                    col_lo += vl * weight;
                    col_hi += vh * weight;
                }
                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
            }
        }
        *out_lo_ptr = col_lo;
        *out_hi_ptr = col_hi;
    }
}

// ----------------------------------------------------------------------------
// Binding — ms_deform_attn_fp4v_forward
// API identical to 1-B's ms_deform_attn_fp4_forward (drop-in replacement)
// ----------------------------------------------------------------------------
torch::Tensor ms_deform_attn_fp4v_forward(
    const torch::Tensor &value_packed,
    int64_t channels,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    double value_scale)
{
    TORCH_CHECK(value_packed.is_cuda());
    TORCH_CHECK(value_packed.scalar_type() == torch::kUInt8,
                "value_packed must be uint8 (packed FP4 storage)");
    TORCH_CHECK(value_packed.is_contiguous());
    TORCH_CHECK(value_packed.dim() == 4, "value_packed must be 4-D (bs, ss, h, c/2)");
    TORCH_CHECK(channels % 2 == 0, "channels must be even");
    TORCH_CHECK(value_packed.size(3) * 2 == channels,
                "value_packed.size(3) must equal channels/2");

    TORCH_CHECK(spatial_shapes.is_cuda() && spatial_shapes.scalar_type() == torch::kInt64
                && spatial_shapes.is_contiguous());
    TORCH_CHECK(level_start_index.is_cuda() && level_start_index.scalar_type() == torch::kInt64
                && level_start_index.is_contiguous());
    TORCH_CHECK(sampling_loc.is_cuda() && sampling_loc.scalar_type() == torch::kFloat32
                && sampling_loc.is_contiguous());
    TORCH_CHECK(attn_weight.is_cuda() && attn_weight.scalar_type() == torch::kFloat32
                && attn_weight.is_contiguous());

    const int bs           = value_packed.size(0);
    const int spatial_size = value_packed.size(1);
    const int num_heads    = value_packed.size(2);
    const int num_levels   = spatial_shapes.size(0);
    const int num_query    = sampling_loc.size(1);
    const int num_point    = sampling_loc.size(4);

    auto output = torch::zeros({bs, num_query, num_heads, (int64_t)channels},
                               torch::dtype(torch::kFloat32).device(value_packed.device()));

    // num_kernels = bs * q * m * (channels/2)  — HALF of 1-B's count
    const int channels_pairs = (int)channels >> 1;
    const int n_pairs = bs * num_query * num_heads * channels_pairs;
    const int block = 256;
    const int grid = (n_pairs + block - 1) / block;

    ms_deformable_im2col_gpu_kernel_fp4v<<<grid, block>>>(
        n_pairs,
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
    m.def("quantize_pack_fp4v", &quantize_pack_fp4v,
          "FP32 → packed FP4 e2m1 (uint8, 2-per-byte) — same as fp4 module");
    m.def("ms_deform_attn_fp4v_forward", &ms_deform_attn_fp4v_forward,
          "MS Deform Attn forward (FP4 e2m1, vector decode, 1 thread = 2 channels)");
}
