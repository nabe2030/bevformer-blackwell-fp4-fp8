"""Phase 1-C-b-2 smoke: build + 1 forward+backward, shape/dtype/NaN/Inf check."""
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")
import torch

def main():
    torch.manual_seed(0)
    bs, num_query, num_heads, channels = 1, 4, 1, 32
    num_levels, num_points = 1, 4
    spatial_shapes_list = [(8, 8)]
    spatial_size = sum(h * w for h, w in spatial_shapes_list)

    value_fp32 = torch.randn(bs, spatial_size, num_heads, channels,
                              device='cuda', requires_grad=True)
    sl = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                     device='cuda', requires_grad=True)
    aw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                      device='cuda').softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous().clone().requires_grad_()
    ss = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    lsi = torch.zeros(num_levels, dtype=torch.int64, device='cuda')

    print("--- build + import msda_fp4_backward ---")
    from msda_fp4_backward import fp4_trainable
    print("OK")

    print("--- forward + backward ---")
    out = fp4_trainable(value_fp32, ss, lsi, sl, aw)
    print(f"out shape={tuple(out.shape)} dtype={out.dtype} mean={out.mean().item():.4f}")
    assert out.shape == (bs, num_query, num_heads * channels)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any() and not torch.isinf(out).any()

    out.pow(2).sum().backward()
    for name, g in [('grad_value', value_fp32.grad),
                    ('grad_sl',    sl.grad),
                    ('grad_aw',    aw.grad)]:
        assert g is not None, f"{name} is None"
        assert not torch.isnan(g).any(), f"NaN in {name}"
        assert not torch.isinf(g).any(), f"Inf in {name}"
        print(f"{name}: shape={tuple(g.shape)} dtype={g.dtype} abs_max={g.abs().max().item():.4e}")

    print("\n[SMOKE PASS]")

if __name__ == '__main__':
    main()
