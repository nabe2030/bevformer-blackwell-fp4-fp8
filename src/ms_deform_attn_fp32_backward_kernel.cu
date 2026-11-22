// ms_deform_attn_fp32_backward_kernel.cu
// SPDX-License-Identifier: Apache-2.0
//
// Derived from mmcv 1.7.2 (https://github.com/open-mmlab/mmcv),
//   mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
//   Copyright (c) OpenMMLab. All rights reserved. Licensed under the Apache License 2.0.
// which carries the notice:
//   Deformable DETR. Copyright (c) 2020 SenseTime. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0.
// Modifications (2026, the bevformer-blackwell-fp4-fp8 authors): FP32 backward with direct atomics (reference).
//
// Stage 1-C debug ref: FP32 backward (port of mmcv's own kernel).
//
// Phase 1-C-3 debug step (c): as the correct reference, provide an FP32 backward with the
// same kernel structure as the FP8 backward, changing only the value type to FP32. Comparing
// it with the autograd grid_sample ref decides whether the observed 88-90x gap in grad_sloc/aw is
//   - common to the kernel (also in FP32 → convention gap between the autograd ref and mmcv's hand-written grad)
//   - specific to the FP8 path (not in FP32 → a bug in the FP8 path).
//
// Ports mmcv's own kernel (ms_deform_attn_cuda_kernel.cuh L67-128 col2im_bilinear +
// L636-736 _shm_reduce_v2_multi_blocks) in the same simplified form as the FP8 backward
// (each thread writes with direct atomics).

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// BLI backward helper — FP32 4-neighbor (fully equivalent to mmcv L67-128, direct FP32 load)
__device__ __forceinline__ void ms_deform_attn_col2im_bilinear_fp32(
    const float *bottom_data,
    const int height, const int width,
    const int nheads, const int channels,
    const float h, const float w,
    const int m, const int c,
    const float top_grad,
    const float attn_w,
    float *grad_value,
    float *grad_loc_x, float *grad_loc_y, float *grad_w)
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

    const float w1 = hh * hw;
    const float w2 = hh * lw;
    const float w3 = lh * hw;
    const float w4 = lh * lw;

    const float top_grad_value = top_grad * attn_w;

    float grad_h_weight = 0.0f;
    float grad_w_weight = 0.0f;
    float val = 0.0f;

    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        const float v1 = bottom_data[ptr1];
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        val           += w1 * v1;
        atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    }
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        const float v2 = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        val           += w2 * v2;
        atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    }
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        const float v3 = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        val           += w3 * v3;
        atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        const float v4 = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        val           += w4 * v4;
        atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    }

    *grad_w     = top_grad * val;
    *grad_loc_x = width  * grad_w_weight * top_grad_value;
    *grad_loc_y = height * grad_h_weight * top_grad_value;
}


__global__ void ms_deformable_col2im_gpu_kernel_fp32(
    const int n,
    const float *grad_col,
    const float *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const float *data_sampling_loc,
    const float *data_attn_weight,
    const int batch_size, const int spatial_size,
    const int num_heads, const int channels,
    const int num_levels, const int num_query,
    const int num_point,
    float *grad_value,
    float *grad_sampling_loc,
    float *grad_attn_weight)
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

        const float top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        float *grad_sampling_loc_out = grad_sampling_loc + (grad_sampling_ptr << 1);
        float *grad_attn_weight_out  = grad_attn_weight  +  grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride    = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset =
                data_value_ptr_init_offset + level_start_id * qid_stride;
            const float *data_value_ptr = data_value + value_ptr_offset;
            float *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const float loc_w = data_sampling_loc[data_loc_w_ptr];
                const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const float weight = data_attn_weight[data_weight_ptr];

                const float h_im = loc_h * spatial_h - 0.5f;
                const float w_im = loc_w * spatial_w - 0.5f;

                if (h_im > -1.0f && w_im > -1.0f &&
                    h_im < (float)spatial_h && w_im < (float)spatial_w) {
                    float local_grad_x = 0.0f, local_grad_y = 0.0f, local_grad_w = 0.0f;
                    ms_deform_attn_col2im_bilinear_fp32(
                        data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                        h_im, w_im, m_col, c_col, top_grad, weight,
                        grad_value_ptr,
                        &local_grad_x, &local_grad_y, &local_grad_w);
                    atomicAdd(grad_sampling_loc_out,     local_grad_x);
                    atomicAdd(grad_sampling_loc_out + 1, local_grad_y);
                    atomicAdd(grad_attn_weight_out,      local_grad_w);
                }

                data_weight_ptr      += 1;
                data_loc_w_ptr       += 2;
                grad_attn_weight_out += grad_weight_stride;
                grad_sampling_loc_out += grad_loc_stride;
            }
        }
    }
}


void ms_deform_attn_fp32_backward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const torch::Tensor &grad_output,
    torch::Tensor &grad_value,
    torch::Tensor &grad_sampling_loc,
    torch::Tensor &grad_attn_weight)
{
    TORCH_CHECK(value.is_cuda() && value.scalar_type() == torch::kFloat32, "value FP32 CUDA");
    TORCH_CHECK(value.is_contiguous(), "value contiguous");
    TORCH_CHECK(grad_value.scalar_type() == torch::kFloat32, "grad_value FP32");

    const int bs            = value.size(0);
    const int spatial_size  = value.size(1);
    const int num_heads     = value.size(2);
    const int channels      = value.size(3);
    const int num_levels    = spatial_shapes.size(0);
    const int num_query     = sampling_loc.size(1);
    const int num_point     = sampling_loc.size(4);

    const int num_kernels = bs * num_query * num_heads * channels;
    const int block = (channels > 1024) ? 1024 : channels;
    const int grid  = (num_kernels + block - 1) / block;

    ms_deformable_col2im_gpu_kernel_fp32<<<grid, block>>>(
        num_kernels,
        grad_output.data_ptr<float>(),
        value.data_ptr<float>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        bs, spatial_size, num_heads, channels,
        num_levels, num_query, num_point,
        grad_value.data_ptr<float>(),
        grad_sampling_loc.data_ptr<float>(),
        grad_attn_weight.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch: ", cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_fp32_backward", &ms_deform_attn_fp32_backward,
          "MS Deform Attn backward FP32 (port of mmcv, used as ref)");
}
