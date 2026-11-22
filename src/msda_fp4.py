"""Stage 1-B FP4 MultiScaleDeformableAttention forward.

Key constraints (from PyTorch dtype probe 2026-05-23):
  - torch.float4_e2m1fn_x2 dtype declared but unusable (zeros/.to() NotImplementedError on cuda)
  - → use torch.uint8 storage + own quantize+pack CUDA helper
  - storage shape: (bs, ss, num_heads, channels/2), each byte = 2 packed FP4

Design fidelity (mirrors FP8 path):
  - eval-only (backward raises until 1-C)
  - sampling_loc / attn_weight stay float32 (type_as skip)
  - value pre-quantized in CUDA helper (per-tensor scale = max / 6 — FP4 e2m1 max)
  - output dtype float32 (caller cast if needed)
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.autograd.function import Function, once_differentiable
from torch.utils.cpp_extension import load

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp4")
os.makedirs(_BUILD_DIR, exist_ok=True)

# Auto-detect GPU capability for -gencode (shared by Spark sm_121 / B200 sm_100)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from _arch_cflags import cuda_cflags

_ext = load(
    name="msda_fp4",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp4_kernel.cu")],
    build_directory=_BUILD_DIR,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)


def quantize_fp4_per_tensor(value: torch.Tensor):
    """FP32 → packed FP4 e2m1 (uint8 storage, 2-per-byte) + per-tensor scale.

    scale = max(|value|) / 6   (FP4 e2m1 max representable is 6)
    Returns (value_packed_uint8, scale_python_float).
    value_packed shape: input shape with last dim halved.
    """
    assert value.dim() >= 1 and value.shape[-1] % 2 == 0, \
        f"last dim must be even for FP4 packing; got shape={value.shape}"
    assert value.is_cuda, "quantize requires CUDA tensor"
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 6.0).to(torch.float32)
    value_packed = _ext.quantize_pack_fp4(value.to(torch.float32).contiguous(), float(scale.item()))
    return value_packed, float(scale.item())


class MultiScaleDeformableAttnFunctionFp4(Function):
    """FP4 forward (eval-only). Backward raises until Stage 1-C."""

    @staticmethod
    def forward(ctx, value_packed: torch.Tensor, channels: int,
                spatial_shapes: torch.Tensor, level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor, attention_weights: torch.Tensor,
                value_scale: float) -> torch.Tensor:
        # type_as skip: sampling/attn stay float32 (even more important for FP4: keep coordinate precision)
        assert value_packed.dtype == torch.uint8, \
            f"value_packed must be uint8 (packed FP4 storage), got {value_packed.dtype}"
        assert sampling_locations.dtype == torch.float32
        assert attention_weights.dtype == torch.float32
        return _ext.ms_deform_attn_fp4_forward(
            value_packed, channels, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, float(value_scale))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise RuntimeError(
            "FP4 backward is Stage 1-C (not yet implemented). "
            "Use FP32 path for training."
        )


def fp4_forward(value_fp32: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
    """Convenience: quantize+pack value then call FP4 kernel.

    value_fp32: original FP32 value (bs, ss, num_heads, channels)
    sampling_locations / attention_weights: must be float32
    Returns: (bs, num_query, num_heads * channels) float32
    """
    channels = value_fp32.shape[-1]
    value_packed, scale = quantize_fp4_per_tensor(value_fp32)
    return MultiScaleDeformableAttnFunctionFp4.apply(
        value_packed, channels, spatial_shapes, level_start_index,
        sampling_locations, attention_weights, scale)
