"""Phase 1-C-b-3 numerical: FP4 backward vs mmcv's own FP32 ref.

forward 1-B (FP4 scalar) had normalized abs 13.88% (5.4x the 2.55% of forward 1-A FP8).
The backward grads inherit this via propagation + atomics; the tol is tuned after measuring
(expected to be looser than the 2.5% of the FP8 backward).
"""
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")
import torch

from msda_fp32_ref import fp32_ref
from msda_fp4_backward import fp4_trainable, quantize_fp4_per_tensor


def stats(ref, got, name):
    abs_diff = (got.float() - ref.float()).abs()
    out_scale = ref.abs().mean().clamp_min(1e-12)
    rel = abs_diff / (ref.abs() + 1e-12)
    norm = abs_diff.mean().item() / out_scale.item()
    print(f"  {name:20s}: abs max={abs_diff.max().item():.4e} mean={abs_diff.mean().item():.4e}  "
          f"rel max={rel.max().item():.4e} mean={rel.mean().item():.4e}  "
          f"normalized_abs={norm:.4e}")
    return norm


def main():
    torch.manual_seed(0)
    bs, num_query, num_heads, channels = 2, 100, 8, 32
    num_levels, num_points = 1, 8
    spatial_shapes_list = [(50, 50)]
    spatial_size = sum(h * w for h, w in spatial_shapes_list)

    value0 = torch.randn(bs, spatial_size, num_heads, channels,
                          device='cuda', dtype=torch.float32)
    sloc0 = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                        device='cuda', dtype=torch.float32)
    aw0 = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                      device='cuda', dtype=torch.float32).softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous()
    ss = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    lsi = torch.zeros(num_levels, dtype=torch.int64, device='cuda')

    # FP32 ref (port of mmcv's own kernel)
    v_r = value0.clone().detach().requires_grad_()
    sl_r = sloc0.clone().detach().requires_grad_()
    aw_r = aw0.clone().detach().requires_grad_()
    out_r = fp32_ref(v_r, ss, lsi, sl_r, aw_r)
    out_r.pow(2).sum().backward()
    print(f"FP32 ref: out abs_max={out_r.abs().max().item():.4f}")
    print(f"  grad_v abs_max={v_r.grad.abs().max().item():.4e} mean={v_r.grad.abs().mean().item():.4e}")
    print(f"  grad_sl abs_max={sl_r.grad.abs().max().item():.4e} mean={sl_r.grad.abs().mean().item():.4e}")
    print(f"  grad_aw abs_max={aw_r.grad.abs().max().item():.4e} mean={aw_r.grad.abs().mean().item():.4e}")

    # FP4 trainable
    v_4 = value0.clone().detach().requires_grad_()
    sl_4 = sloc0.clone().detach().requires_grad_()
    aw_4 = aw0.clone().detach().requires_grad_()
    out_4 = fp4_trainable(v_4, ss, lsi, sl_4, aw_4)
    out_4.pow(2).sum().backward()
    print(f"\nFP4: out abs_max={out_4.abs().max().item():.4f}")

    print(f"\n--- forward output ---")
    n_out = stats(out_r.detach(), out_4.detach(), "output")
    print(f"\n--- backward grads ---")
    n_gv = stats(v_r.grad, v_4.grad, "grad_value")
    n_gs = stats(sl_r.grad, sl_4.grad, "grad_sampling_loc")
    n_ga = stats(aw_r.grad, aw_4.grad, "grad_attn_weight")

    # FP4 forward 1-B is known to be ~14%; the backward is expected to be similar
    TOL = {'output': 0.20, 'grad_value': 0.20, 'grad_sampling_loc': 0.20, 'grad_attn_weight': 0.20}
    results = {'output': n_out, 'grad_value': n_gv,
               'grad_sampling_loc': n_gs, 'grad_attn_weight': n_ga}
    print("\n--- verdict (FP4 vs mmcv FP32 ref) ---")
    ok = True
    for k, v in results.items():
        t = TOL[k]
        verdict = "PASS" if v < t else "FAIL"
        if v >= t: ok = False
        print(f"  {k:20s} norm={v:.4e} tol={t}  [{verdict}]")

    _, scale = quantize_fp4_per_tensor(value0)
    print(f"\nvalue_scale = {scale:.6f} (= max/6)")
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
