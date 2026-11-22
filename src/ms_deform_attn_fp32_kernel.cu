// ms_deform_attn_fp32_kernel.cu
// SPDX-License-Identifier: Apache-2.0
//
// Derived from mmcv 1.7.2 (https://github.com/open-mmlab/mmcv),
//   mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
//   Copyright (c) OpenMMLab. All rights reserved. Licensed under the Apache License 2.0.
// which carries the notice:
//   Deformable DETR. Copyright (c) 2020 SenseTime. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0.
// Modifications (2026, the bevformer-blackwell-fp4-fp8 authors): FP32 twin of the FP8 kernel (microbenchmark baseline).
//
// Stage 1-A-6 microbench baseline — FP32 twin of ms_deform_attn_fp8_kernel.cu.
//
// PURPOSE: provide a fair baseline for FP8 vs FP32 comparison.
// Only difference from FP8 twin: value dtype is float (not __nv_fp8_e4m3).
// Everything else — thread layout, BLI structure, accumulator type, boundary
// checks, loop structure, launch config — is *byte-for-byte identical* so that
// the timing difference reflects purely the data-width (memory bandwidth) change.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// BLI helper — twin of ms_deform_attn_im2col_bilinear_fp8 (value is float)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float ms_deform_attn_im2col_bilinear_fp32(
    const float *bottom_data,
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

    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    float v1 = 0.0f;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];  // ← FP8: static_cast<float>(bottom_data[ptr1])
    }
    float v2 = 0.0f;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    float v3 = 0.0f;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    float v4 = 0.0f;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }

    const float w1 = hh * hw;
    const float w2 = hh * lw;
    const float w3 = lh * hw;
    const float w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// ----------------------------------------------------------------------------
// forward kernel — twin of ms_deformable_im2col_gpu_kernel_fp8 (value is float)
// ----------------------------------------------------------------------------
__global__ void ms_deformable_im2col_gpu_kernel_fp32(
    const int n,
    const float *data_value,
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
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        float col = 0.0f;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const float *data_value_ptr =
                data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const float loc_w = data_sampling_loc[data_loc_w_ptr];
                const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const float weight = data_attn_weight[data_weight_ptr];

                const float h_im = loc_h * spatial_h - 0.5f;
                const float w_im = loc_w * spatial_w - 0.5f;

                if (h_im > -1.0f && w_im > -1.0f &&
                    h_im < (float)spatial_h && w_im < (float)spatial_w) {
                    col += ms_deform_attn_im2col_bilinear_fp32(
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
// Binding — ms_deform_attn_fp32_forward
// Same as FP8 twin minus value_scale (float input → no quantize).
// ----------------------------------------------------------------------------
torch::Tensor ms_deform_attn_fp32_forward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight)
{
    TORCH_CHECK(value.is_cuda() && value.scalar_type() == torch::kFloat32,
                "value must be CUDA float32");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(spatial_shapes.is_cuda() && spatial_shapes.scalar_type() == torch::kInt64,
                "spatial_shapes must be CUDA int64");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
    TORCH_CHECK(level_start_index.is_cuda() && level_start_index.scalar_type() == torch::kInt64,
                "level_start_index must be CUDA int64");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index must be contiguous");
    TORCH_CHECK(sampling_loc.is_cuda() && sampling_loc.scalar_type() == torch::kFloat32,
                "sampling_loc must be CUDA float32");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
    TORCH_CHECK(attn_weight.is_cuda() && attn_weight.scalar_type() == torch::kFloat32,
                "attn_weight must be CUDA float32");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight must be contiguous");

    const int bs            = value.size(0);
    const int spatial_size  = value.size(1);
    const int num_heads     = value.size(2);
    const int channels      = value.size(3);
    const int num_levels    = spatial_shapes.size(0);
    const int num_query     = sampling_loc.size(1);
    const int num_point     = sampling_loc.size(4);

    auto output = torch::zeros({bs, num_query, num_heads, channels},
                               torch::dtype(torch::kFloat32).device(value.device()));

    const int num_kernels = bs * num_query * num_heads * channels;
    const int block = 256;
    const int grid  = (num_kernels + block - 1) / block;

    ms_deformable_im2col_gpu_kernel_fp32<<<grid, block>>>(
        num_kernels,
        value.data_ptr<float>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        bs, spatial_size, num_heads, channels,
        num_levels, num_query, num_point,
        output.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));

    // No value_scale: FP32 value, no quantize step, no mul_.
    return output.view({bs, num_query, num_heads * channels});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_fp32_forward", &ms_deform_attn_fp32_forward,
          "MS Deform Attn forward (FP32 value, baseline twin of FP8 kernel)");
}
