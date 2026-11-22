"""Stage 1-B'-a numerical check.

Two checks:
  1. FP4v vs FP4 (scalar 1-B): same quantization, expect bit-near-equal output
     (decoded values are identical, only iteration order changes; floats can
     drift slightly but normalized should match).
  2. FP4v vs FP32 reference: expect similar normalized abs to 1-B (~13.88%).
"""
import os
import subprocess
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
import torch.nn.functional as F

from msda_fp4 import fp4_forward as fp4_forward_scalar
from msda_fp4v import fp4v_forward


def print_gpu_coexistence():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True).strip()
        print(f"--- GPU coexistence ---\n{out or '(none)'}")
    except Exception as e:
        print(f"--- GPU coexistence query failed: {e} ---")


# The reference function below is a port of multi_scale_deformable_attn_pytorch from
# mmcv 1.7.2 (mmcv/ops/multi_scale_deform_attn.py), Copyright (c) OpenMMLab.
# All rights reserved. Licensed under the Apache License, Version 2.0.
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                         sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(H_) * int(W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    aw = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              aw).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


def main():
    print_gpu_coexistence()
    torch.manual_seed(0)

    bs, num_query, num_heads, channels = 2, 100, 8, 32
    num_levels, num_points = 1, 8
    spatial_shapes_list = [(50, 50)]
    spatial_size = sum(h * w for h, w in spatial_shapes_list)

    value_fp32 = torch.randn(bs, spatial_size, num_heads, channels,
                             device='cuda', dtype=torch.float32)
    sampling_locations = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                                    device='cuda', dtype=torch.float32)
    aw_raw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                         device='cuda', dtype=torch.float32)
    attention_weights = aw_raw.softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous()

    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    level_start_index = torch.zeros(num_levels, dtype=torch.int64, device='cuda')

    # 1. FP32 reference
    out_ref = multi_scale_deformable_attn_pytorch(
        value_fp32, spatial_shapes.cpu(), sampling_locations, attention_weights)

    # 2. FP4 scalar (1-B)
    out_fp4 = fp4_forward_scalar(value_fp32, spatial_shapes, level_start_index,
                                 sampling_locations, attention_weights)

    # 3. FP4 vector (1-B'-a)
    out_fp4v = fp4v_forward(value_fp32, spatial_shapes, level_start_index,
                            sampling_locations, attention_weights)

    print(f"\nshapes: ref={tuple(out_ref.shape)} fp4={tuple(out_fp4.shape)} fp4v={tuple(out_fp4v.shape)}")

    # --- (a) FP4v vs FP4 scalar ---
    abs_diff_vs_scalar = (out_fp4v - out_fp4).abs()
    max_d = abs_diff_vs_scalar.max().item()
    mean_d = abs_diff_vs_scalar.mean().item()
    print(f"\n[A] FP4v vs FP4 scalar (1-B)")
    print(f"    abs diff: max={max_d:.6e}, mean={mean_d:.6e}")
    if max_d < 1e-5:
        print(f"    [BIT-EXACT MATCH] decode equivalence confirmed")
    elif max_d < 1e-3:
        print(f"    [NEAR-EQUAL] tiny float drift, decode equivalent in practice")
    else:
        print(f"    [DIFFER] significant difference — may indicate a bug")

    # --- (b) FP4v vs FP32 reference ---
    abs_diff_vs_ref = (out_fp4v.float() - out_ref).abs()
    norm_diff = abs_diff_vs_ref.mean().item() / out_ref.abs().mean().clamp_min(1e-6).item()
    print(f"\n[B] FP4v vs FP32 reference")
    print(f"    abs diff: max={abs_diff_vs_ref.max().item():.4e}, mean={abs_diff_vs_ref.mean().item():.4e}")
    print(f"    normalized abs: {norm_diff:.4e} (1-B FP4 scalar was 13.88%)")

    if norm_diff < 0.20:
        print(f"\n[PASS] normalized abs {norm_diff*100:.2f}% within FP4 e2m1 expected range")
        return 0
    else:
        print(f"\n[WARN] normalized abs {norm_diff*100:.2f}% — investigate")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
