"""Phase 1-C-2 smoke test: build + 1 forward+backward iteration.

Confirms:
  - build succeeds (cpp_extension.load + ninja)
  - 1 forward returns valid shape / dtype
  - 1 backward returns grad_value / grad_loc / grad_weight with expected shape/dtype
  - no NaN/Inf in any output or grad
  - no CUDA error

NOT numerical correctness — see test_numerical_fp8_backward.py for that.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch


def main():
    torch.manual_seed(0)

    # Smallest viable shape (tiny scale)
    bs, num_query, num_heads, channels = 1, 4, 1, 32
    num_levels, num_points = 1, 4
    spatial_shapes_list = [(8, 8)]
    spatial_size = sum(h * w for h, w in spatial_shapes_list)

    value_fp32 = torch.randn(bs, spatial_size, num_heads, channels,
                              device='cuda', dtype=torch.float32,
                              requires_grad=True)
    sampling_locations = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                                     device='cuda', dtype=torch.float32,
                                     requires_grad=True)
    aw_raw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                          device='cuda', dtype=torch.float32)
    attention_weights = aw_raw.softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous().clone().requires_grad_()

    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    level_start_index = torch.zeros(num_levels, dtype=torch.int64, device='cuda')

    print("--- import & build msda_fp8_backward (this triggers nvcc compile) ---")
    from msda_fp8_backward import fp8_trainable
    print("import OK")

    print("--- forward ---")
    output = fp8_trainable(value_fp32, spatial_shapes, level_start_index,
                            sampling_locations, attention_weights)
    print(f"output.shape = {tuple(output.shape)}, dtype = {output.dtype}")
    print(f"output: mean={output.mean().item():.4f}, abs_max={output.abs().max().item():.4f}")
    assert output.shape == (bs, num_query, num_heads * channels), \
        f"unexpected output shape: {output.shape}"
    assert output.dtype == torch.float32
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    print("--- backward ---")
    # arbitrary scalar loss
    loss = output.pow(2).sum()
    print(f"loss = {loss.item():.4f}")
    loss.backward()

    print(f"grad_value: shape={tuple(value_fp32.grad.shape)}, dtype={value_fp32.grad.dtype}, "
          f"abs_max={value_fp32.grad.abs().max().item():.4e}")
    print(f"grad_sampling_loc: shape={tuple(sampling_locations.grad.shape)}, "
          f"abs_max={sampling_locations.grad.abs().max().item():.4e}")
    print(f"grad_attn_weight: shape={tuple(attention_weights.grad.shape)}, "
          f"abs_max={attention_weights.grad.abs().max().item():.4e}")

    for name, g in [('grad_value', value_fp32.grad),
                    ('grad_sampling_loc', sampling_locations.grad),
                    ('grad_attn_weight', attention_weights.grad)]:
        assert g is not None, f"{name} is None"
        assert not torch.isnan(g).any(), f"NaN in {name}"
        assert not torch.isinf(g).any(), f"Inf in {name}"

    assert value_fp32.grad.shape == value_fp32.shape
    assert sampling_locations.grad.shape == sampling_locations.shape
    assert attention_weights.grad.shape == attention_weights.shape

    print("\n[SMOKE PASS] forward + backward + grad shapes/dtypes all OK")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
