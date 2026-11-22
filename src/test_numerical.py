"""Stage 1-A-5 numerical verification.

Compare FP8 kernel vs FP32 CPU reference (multi_scale_deformable_attn_pytorch,
ported from mmcv 1.7.2 multi_scale_deform_attn.py L110-169).

BEVFormer tiny shape: bs=2, num_query=100, num_heads=8, channels=32,
                      num_levels=1, num_points=8, bev 50x50.
"""
import os
import subprocess
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
import torch.nn.functional as F

from msda_fp8 import fp8_forward, quantize_fp8_per_tensor


def print_gpu_coexistence():
    """Borrowed-Spark hygiene: show what else is on the GPU."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True).strip()
        if out:
            print(f"--- GPU coexistence ---\n{out}")
        else:
            print("--- GPU coexistence: no other compute apps ---")
    except Exception as e:
        print(f"--- GPU coexistence query failed: {e} ---")


# The reference function below is a port of multi_scale_deformable_attn_pytorch from
# mmcv 1.7.2 (mmcv/ops/multi_scale_deform_attn.py), Copyright (c) OpenMMLab.
# All rights reserved. Licensed under the Apache License, Version 2.0.
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                         sampling_locations, attention_weights):
    """FP32 CPU reference — verbatim port from mmcv 1.7.2 multi_scale_deform_attn.py L110-169.

    value:                (bs, sum_l H_l*W_l, num_heads, embed_dims)
    value_spatial_shapes: tensor (num_levels, 2)
    sampling_locations:   (bs, num_queries, num_heads, num_levels, num_points, 2)
    attention_weights:    (bs, num_queries, num_heads, num_levels, num_points)
    returns:              (bs, num_queries, num_heads * embed_dims)
    """
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

    # BEVFormer tiny dims (single-level, num_points=8)
    bs, num_query, num_heads, channels = 2, 100, 8, 32
    num_levels, num_points = 1, 8
    spatial_shapes_list = [(50, 50)]
    spatial_size = sum(h * w for h, w in spatial_shapes_list)  # 2500

    # Random inputs
    value_fp32 = torch.randn(bs, spatial_size, num_heads, channels,
                             device='cuda', dtype=torch.float32)
    sampling_locations = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                                    device='cuda', dtype=torch.float32)
    # attention_weights normalized over (num_levels * num_points) as mmcv wrapper does
    aw_raw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                         device='cuda', dtype=torch.float32)
    aw_norm = aw_raw.softmax(-1).view(bs, num_query, num_heads, num_levels, num_points)
    attention_weights = aw_norm.contiguous()

    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    level_start_index = torch.zeros(num_levels, dtype=torch.int64, device='cuda')

    # --- FP32 reference ---
    output_ref = multi_scale_deformable_attn_pytorch(
        value_fp32, spatial_shapes.cpu(), sampling_locations, attention_weights)
    print(f"\nFP32 reference: shape={tuple(output_ref.shape)}, dtype={output_ref.dtype}")
    print(f"  abs_max={output_ref.abs().max().item():.4f}, abs_mean={output_ref.abs().mean().item():.4f}")

    # --- FP8 kernel ---
    output_fp8 = fp8_forward(value_fp32, spatial_shapes, level_start_index,
                             sampling_locations, attention_weights)
    print(f"FP8 kernel:     shape={tuple(output_fp8.shape)}, dtype={output_fp8.dtype}")

    assert output_ref.shape == output_fp8.shape, \
        f"shape mismatch: ref={output_ref.shape}, fp8={output_fp8.shape}"

    # --- numerical comparison ---
    abs_diff = (output_fp8.float() - output_ref).abs()
    out_scale = output_ref.abs().mean().clamp_min(1e-6)
    rel_diff = abs_diff / (output_ref.abs() + 1e-6)

    print(f"\n--- FP32 ref vs FP8 kernel ---")
    print(f"abs diff: max={abs_diff.max().item():.4e}, mean={abs_diff.mean().item():.4e}")
    print(f"rel diff (per-element): max={rel_diff.max().item():.4e}, mean={rel_diff.mean().item():.4e}")
    norm_diff = abs_diff.mean().item() / out_scale.item()
    print(f"normalized abs (mean abs diff / mean abs ref): {norm_diff:.4e}")

    # FP8 e4m3 has ~3-bit mantissa = ~1/8 relative precision per single value.
    # BLI sums 4 values × num_points = 32 weighted contributions; statistical
    # averaging tends to reduce mean error, but max error can spike on small
    # output values. Expected: mean normalized < 5%, max rel may be larger on tiny outputs.
    PASS_MEAN = 0.05   # 5%
    PASS_NORM = 0.05   # 5%
    if norm_diff < PASS_NORM:
        print(f"\n[PASS] normalized abs error {norm_diff:.4e} < {PASS_NORM} "
              "(within FP8 e4m3 expected range)")
        verdict = 0
    else:
        print(f"\n[FAIL] normalized abs error {norm_diff:.4e} ≥ {PASS_NORM} "
              "(too large — possible bug in BLI / scale / dispatch)")
        verdict = 1

    _, scale = quantize_fp8_per_tensor(value_fp32)
    print(f"\nvalue_scale = {scale:.6f}, value abs max = {value_fp32.abs().max().item():.4f}")
    print(f"  (= max / 448)")

    return verdict


if __name__ == '__main__':
    import sys
    sys.exit(main())
