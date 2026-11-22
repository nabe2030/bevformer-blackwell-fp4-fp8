// ms_deform_attn_fp4_backward_kernel.cu
// SPDX-License-Identifier: Apache-2.0
//
// Derived from mmcv 1.7.2 (https://github.com/open-mmlab/mmcv),
//   mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
//   Copyright (c) OpenMMLab. All rights reserved. Licensed under the Apache License 2.0.
// which carries the notice:
//   Deformable DETR. Copyright (c) 2020 SenseTime. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0.
// Modifications (2026, the bevformer-blackwell-fp4-fp8 authors): packed FP4 e2m1 value in the backward, FP32 grad_value staging.
//
// Stage 1-C-b: backward (training), packed FP4 e2m1 value load + FP32 grad_value staging.
//
// Design: follows 1-C-a (FP8 backward); only the value type is replaced by packed FP4 (uint8 storage).
//   - value read: packed __nv_fp4x2_e2m1* (1 byte = 2 nibbles), decoded by load_fp4_nibble
//   - grad_value write: FP32 staging buffer (unpacked shape, 1 atomic per c)
//   - two strides for read/write (packed for read, unpacked for the FP32 grad atomic)
//   - grad_attn_weight / grad_sampling_loc: stay float (same as mmcv L195-197)
//   - scale handling: the kernel computes without the scale; the Python wrapper
//     restores grad_sloc / grad_aw with mul_(scale) (same STE pattern as 1-C-a)
//
// Independent extension — no mmcv include. Helpers mirror FP8 backward (Phase 1-C-a)
// with packed FP4 dequant via load_fp4_nibble (same decode path as the forward 1-B implementation).

#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>


// ----------------------------------------------------------------------------
// FP4 nibble decode (same as forward 1-B kernel L91-100, independent re-impl)
//   __nv_fp4x2_e2m1 → __half2 (operator) → __half (intrinsic) → float
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


// ----------------------------------------------------------------------------
// BLI backward helper — packed FP4 4-neighbor (read) + FP32 staging (write)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void ms_deform_attn_col2im_bilinear_fp4_fp32(
    const __nv_fp4x2_e2m1 *bottom_data_pair,  // packed FP4 (uint8 reinterpret)
    const int height, const int width,
    const int nheads, const int channels,
    const float h, const float w,
    const int m, const int c,
    const float top_grad,
    const float attn_w,
    float *grad_value,                          // FP32 unpacked staging
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

    // Packed stride (for FP4 read): channels_pairs = channels/2
    const int channels_pairs = channels >> 1;
    const int w_stride_p = nheads * channels_pairs;
    const int h_stride_p = width * w_stride_p;
    const int h_low_ptr_offset_p = h_low * h_stride_p;
    const int h_high_ptr_offset_p = h_low_ptr_offset_p + h_stride_p;
    const int w_low_ptr_offset_p = w_low * w_stride_p;
    const int w_high_ptr_offset_p = w_low_ptr_offset_p + w_stride_p;
    const int base_ptr_p = m * channels_pairs + (c >> 1);
    const int c_half = c & 1;

    // Unpacked stride (for FP32 grad_value atomic write)
    const int w_stride_u = nheads * channels;
    const int h_stride_u = width * w_stride_u;
    const int h_low_ptr_offset_u = h_low * h_stride_u;
    const int h_high_ptr_offset_u = h_low_ptr_offset_u + h_stride_u;
    const int w_low_ptr_offset_u = w_low * w_stride_u;
    const int w_high_ptr_offset_u = w_low_ptr_offset_u + w_stride_u;
    const int base_ptr_u = m * channels + c;

    const float w1 = hh * hw;
    const float w2 = hh * lw;
    const float w3 = lh * hw;
    const float w4 = lh * lw;

    const float top_grad_value = top_grad * attn_w;

    float grad_h_weight = 0.0f;
    float grad_w_weight = 0.0f;
    float val = 0.0f;

    if (h_low >= 0 && w_low >= 0) {
        const int ptr1_p = h_low_ptr_offset_p + w_low_ptr_offset_p + base_ptr_p;
        const int ptr1_u = h_low_ptr_offset_u + w_low_ptr_offset_u + base_ptr_u;
        const float v1 = load_fp4_nibble(bottom_data_pair, ptr1_p, c_half);
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        val           += w1 * v1;
        atomicAdd(grad_value + ptr1_u, w1 * top_grad_value);
    }
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2_p = h_low_ptr_offset_p + w_high_ptr_offset_p + base_ptr_p;
        const int ptr2_u = h_low_ptr_offset_u + w_high_ptr_offset_u + base_ptr_u;
        const float v2 = load_fp4_nibble(bottom_data_pair, ptr2_p, c_half);
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        val           += w2 * v2;
        atomicAdd(grad_value + ptr2_u, w2 * top_grad_value);
    }
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3_p = h_high_ptr_offset_p + w_low_ptr_offset_p + base_ptr_p;
        const int ptr3_u = h_high_ptr_offset_u + w_low_ptr_offset_u + base_ptr_u;
        const float v3 = load_fp4_nibble(bottom_data_pair, ptr3_p, c_half);
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        val           += w3 * v3;
        atomicAdd(grad_value + ptr3_u, w3 * top_grad_value);
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4_p = h_high_ptr_offset_p + w_high_ptr_offset_p + base_ptr_p;
        const int ptr4_u = h_high_ptr_offset_u + w_high_ptr_offset_u + base_ptr_u;
        const float v4 = load_fp4_nibble(bottom_data_pair, ptr4_p, c_half);
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        val           += w4 * v4;
        atomicAdd(grad_value + ptr4_u, w4 * top_grad_value);
    }

    *grad_w     = top_grad * val;
    *grad_loc_x = width  * grad_w_weight * top_grad_value;
    *grad_loc_y = height * grad_h_weight * top_grad_value;
}


// ----------------------------------------------------------------------------
// backward kernel — 1 thread = 1 (b, q, m, c) (= same layout as the FP4 forward)
// Simplified: each thread writes grad_sloc / grad_aw with direct atomics (shm reduce bypass).
// Bisected in Phase 1-C-a: the shm reduce is not needed; direct atomics work correctly.
// ----------------------------------------------------------------------------
__global__ void ms_deformable_col2im_gpu_kernel_fp4(
    const int n,
    const float *grad_col,                       // top_grad
    const __nv_fp4x2_e2m1 *data_value_pair,      // packed FP4
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const float *data_sampling_loc,
    const float *data_attn_weight,
    const int batch_size, const int spatial_size,
    const int num_heads, const int channels,
    const int num_levels, const int num_query,
    const int num_point,
    float *grad_value,                            // FP32 unpacked staging
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

        // Packed value ptr offset (1 byte = 2 nibbles → channels_pairs = channels/2)
        const int channels_pairs = channels >> 1;
        const int qid_stride_p = num_heads * channels_pairs;
        const int data_value_ptr_init_offset_p = b_col * spatial_size * qid_stride_p;
        // Unpacked grad_value ptr offset (FP32 staging)
        const int qid_stride_u = num_heads * channels;
        const int data_value_ptr_init_offset_u = b_col * spatial_size * qid_stride_u;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset_p =
                data_value_ptr_init_offset_p + level_start_id * qid_stride_p;
            const int value_ptr_offset_u =
                data_value_ptr_init_offset_u + level_start_id * qid_stride_u;
            const __nv_fp4x2_e2m1 *data_value_ptr = data_value_pair + value_ptr_offset_p;
            float *grad_value_ptr = grad_value + value_ptr_offset_u;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const float loc_w = data_sampling_loc[data_loc_w_ptr];
                const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const float weight = data_attn_weight[data_weight_ptr];

                const float h_im = loc_h * spatial_h - 0.5f;
                const float w_im = loc_w * spatial_w - 0.5f;

                if (h_im > -1.0f && w_im > -1.0f &&
                    h_im < (float)spatial_h && w_im < (float)spatial_w) {
                    float local_gx = 0.0f, local_gy = 0.0f, local_gw = 0.0f;
                    ms_deform_attn_col2im_bilinear_fp4_fp32(
                        data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                        h_im, w_im, m_col, c_col, top_grad, weight,
                        grad_value_ptr,
                        &local_gx, &local_gy, &local_gw);
                    atomicAdd(grad_sampling_loc_out,     local_gx);
                    atomicAdd(grad_sampling_loc_out + 1, local_gy);
                    atomicAdd(grad_attn_weight_out,      local_gw);
                }

                data_weight_ptr      += 1;
                data_loc_w_ptr       += 2;
                grad_attn_weight_out += grad_weight_stride;
                grad_sampling_loc_out += grad_loc_stride;
            }
        }
    }
}


// ----------------------------------------------------------------------------
// Binding — ms_deform_attn_fp4_backward
// value_packed: uint8 storage (bs, ss, h, c/2)
// channels    : effective channel count (= 2 × value_packed.size(3))
// grad_value  : FP32 staging (bs, ss, h, c)   ★ unpacked shape!
//
// Scale handling (caller responsibility, mirroring 1-C-a):
//   kernel computes grad_sloc / grad_aw with scale-UNAPPLIED v values; Python
//   wrapper must `grad_sampling_loc.mul_(scale); grad_attn_weight.mul_(scale)`
//   to restore. grad_value uses STE chain rule (no scale multiply, mmcv-equivalent).
// ----------------------------------------------------------------------------
void ms_deform_attn_fp4_backward(
    const torch::Tensor &value_packed,
    int64_t channels,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const torch::Tensor &grad_output,
    torch::Tensor &grad_value,
    torch::Tensor &grad_sampling_loc,
    torch::Tensor &grad_attn_weight)
{
    TORCH_CHECK(value_packed.scalar_type() == torch::kUInt8, "value_packed uint8");
    TORCH_CHECK(value_packed.is_contiguous(), "value_packed contiguous");
    TORCH_CHECK(value_packed.dim() == 4, "value_packed 4-D");
    TORCH_CHECK(channels % 2 == 0, "channels even");
    TORCH_CHECK(value_packed.size(3) * 2 == channels, "size(3) = channels/2");
    TORCH_CHECK(grad_value.scalar_type() == torch::kFloat32, "grad_value FP32 staging");
    TORCH_CHECK(grad_value.size(3) == channels,
                "grad_value last dim must be unpacked channels");

    const int bs           = value_packed.size(0);
    const int spatial_size = value_packed.size(1);
    const int num_heads    = value_packed.size(2);
    const int num_levels   = spatial_shapes.size(0);
    const int num_query    = sampling_loc.size(1);
    const int num_point    = sampling_loc.size(4);

    const int num_kernels = bs * num_query * num_heads * (int)channels;
    const int block = ((int)channels > 1024) ? 1024 : (int)channels;
    const int grid  = (num_kernels + block - 1) / block;

    ms_deformable_col2im_gpu_kernel_fp4<<<grid, block>>>(
        num_kernels,
        grad_output.data_ptr<float>(),
        reinterpret_cast<const __nv_fp4x2_e2m1*>(value_packed.data_ptr<uint8_t>()),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        bs, spatial_size, num_heads, (int)channels,
        num_levels, num_query, num_point,
        grad_value.data_ptr<float>(),
        grad_sampling_loc.data_ptr<float>(),
        grad_attn_weight.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch: ", cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_fp4_backward", &ms_deform_attn_fp4_backward,
          "MS Deform Attn backward FP4 (packed uint8 value, FP32 grad_value staging, "
          "scale applied at Python wrapper for linearity)");
}
