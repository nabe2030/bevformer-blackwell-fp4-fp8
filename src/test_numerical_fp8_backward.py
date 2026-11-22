"""Phase 1-C-3 numerical verification (v2, debug step (c)):
FP8 backward vs FP32 ref where ref = my port of mmcv's own backward kernel.

Previous (v1) reference was autograd F.grid_sample (multi_scale_deformable_attn_pytorch),
which produced 88-90x larger grad_sloc/aw than my FP8 kernel.  That gap is now
suspected to be a gradient-definition difference (grid_sample's convention vs
mmcv's hand-written derivative), NOT a bug in the FP8 kernel.

The CORRECT comparison for this research is FP8 (mmcv kernel + FP8 value) vs
FP32 (mmcv kernel + FP32 value), since the project's goal is to FP8-ify mmcv,
not to match autograd grid_sample.  We expect ~2.5% normalized error on all
grads (matching forward 1-A's quantization noise).
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
import torch.nn.functional as F

from msda_fp32_ref import fp32_ref
from msda_fp8_backward import fp8_trainable
from msda_fp8 import quantize_fp8_per_tensor


def compute_stats(ref, got, name):
    """Print abs/rel/normalized diff stats; return normalized-mean-abs."""
    abs_diff = (got.float() - ref.float()).abs()
    out_scale = ref.abs().mean().clamp_min(1e-12)
    rel_diff = abs_diff / (ref.abs() + 1e-12)
    norm = abs_diff.mean().item() / out_scale.item()
    print(f"  {name:20s}: abs max={abs_diff.max().item():.4e} mean={abs_diff.mean().item():.4e}  "
          f"rel max={rel_diff.max().item():.4e} mean={rel_diff.mean().item():.4e}  "
          f"normalized_abs={norm:.4e}")
    return norm


def main():
    torch.manual_seed(0)

    # BEVFormer tiny dims (single-level, num_points=8)
    bs, num_query, num_heads, channels = 2, 100, 8, 32
    num_levels, num_points = 1, 8
    spatial_shapes_list = [(50, 50)]
    spatial_size = sum(h * w for h, w in spatial_shapes_list)

    # Inputs (requires_grad for backward via autograd)
    value0 = torch.randn(bs, spatial_size, num_heads, channels,
                          device='cuda', dtype=torch.float32)
    sloc0 = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                        device='cuda', dtype=torch.float32)
    aw_raw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                          device='cuda', dtype=torch.float32)
    aw0 = aw_raw.softmax(-1).view(bs, num_query, num_heads, num_levels, num_points).contiguous()

    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    level_start_index = torch.zeros(num_levels, dtype=torch.int64, device='cuda')

    # ---- FP32 reference: my port of mmcv's own backward (FP32 path) ----
    v_ref  = value0.clone().detach().requires_grad_()
    sl_ref = sloc0.clone().detach().requires_grad_()
    aw_ref = aw0.clone().detach().requires_grad_()
    out_ref = fp32_ref(v_ref, spatial_shapes, level_start_index, sl_ref, aw_ref)
    print(f"FP32 ref output (mmcv port): shape={tuple(out_ref.shape)}, abs_max={out_ref.abs().max().item():.4f}")

    # Same downstream loss for both paths (so grad_output is identical).
    loss_ref = out_ref.pow(2).sum()
    loss_ref.backward()
    print(f"FP32 ref loss = {loss_ref.item():.4f}")
    print(f"  grad_value:  abs_max={v_ref.grad.abs().max().item():.4e}, mean={v_ref.grad.abs().mean().item():.4e}")
    print(f"  grad_sloc:   abs_max={sl_ref.grad.abs().max().item():.4e}, mean={sl_ref.grad.abs().mean().item():.4e}")
    print(f"  grad_aw:     abs_max={aw_ref.grad.abs().max().item():.4e}, mean={aw_ref.grad.abs().mean().item():.4e}")

    # ---- FP8 trainable: forward + backward ----
    v_fp8  = value0.clone().detach().requires_grad_()
    sl_fp8 = sloc0.clone().detach().requires_grad_()
    aw_fp8 = aw0.clone().detach().requires_grad_()
    out_fp8 = fp8_trainable(v_fp8, spatial_shapes, level_start_index, sl_fp8, aw_fp8)
    print(f"\nFP8 output: shape={tuple(out_fp8.shape)}, abs_max={out_fp8.abs().max().item():.4f}")

    loss_fp8 = out_fp8.pow(2).sum()
    loss_fp8.backward()
    print(f"FP8 loss = {loss_fp8.item():.4f}")

    # ---- forward output comparison (sanity, 1-A known = ~2.55%) ----
    print(f"\n--- forward output: FP32 ref vs FP8 ---")
    fwd_norm = compute_stats(out_ref.detach(), out_fp8.detach(), "output")

    # ---- backward grad comparison ----
    print(f"\n--- backward grads: FP32 ref vs FP8 ---")
    gv_norm = compute_stats(v_ref.grad,  v_fp8.grad,  "grad_value")
    gs_norm = compute_stats(sl_ref.grad, sl_fp8.grad, "grad_sampling_loc")
    ga_norm = compute_stats(aw_ref.grad, aw_fp8.grad, "grad_attn_weight")

    # ---- verdict (vs mmcv's own FP32 backward = correct ref) ----
    print("\n--- verdict (FP8 kernel vs FP32 mmcv ref) ---")
    # All grads should match within FP8 quantization noise (~2.5%).
    TOL = {'output': 0.05, 'grad_value': 0.05, 'grad_sampling_loc': 0.05, 'grad_attn_weight': 0.05}
    results = {'output': fwd_norm, 'grad_value': gv_norm,
               'grad_sampling_loc': gs_norm, 'grad_attn_weight': ga_norm}
    all_pass = True
    for k, norm in results.items():
        tol = TOL[k]
        verdict = "PASS" if norm < tol else "FAIL"
        if norm >= tol:
            all_pass = False
        print(f"  {k:20s}  norm={norm:.4e}  tol={tol}  [{verdict}]")

    _, scale = quantize_fp8_per_tensor(value0)
    print(f"\nvalue_scale = {scale:.6f} (= max/448)")

    return 0 if all_pass else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
