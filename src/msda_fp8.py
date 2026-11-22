"""Stage 1-A FP8 MultiScaleDeformableAttention forward.

  - eval-only (backward raises until 1-C)
  - sampling_loc / attn_weight stay float32 (no type_as(value))
  - value pre-quantized in Python (per-tensor scale = max / 448)
  - output dtype float32
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.autograd.function import Function, once_differentiable
from torch.utils.cpp_extension import load

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp8")
os.makedirs(_BUILD_DIR, exist_ok=True)

# Auto-detect GPU capability for -gencode (shared by Spark sm_121 / B200 sm_100)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from _arch_cflags import cuda_cflags

_ext = load(
    name="msda_fp8",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp8_kernel.cu")],
    build_directory=_BUILD_DIR,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)


def quantize_fp8_per_tensor(value: torch.Tensor):
    """FP32/FP16 tensor → FP8 e4m3 + per-tensor scale.

    scale = max(|value|) / 448   (FP8 e4m3 max representable)
    value_fp8 = (value / scale).clamp(-448, 448).to(float8_e4m3fn)
    Returns (value_fp8_tensor, scale_python_float).
    """
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 448.0).to(torch.float32)
    value_fp8 = (value.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return value_fp8.contiguous(), float(scale.item())


class MultiScaleDeformableAttnFunctionFp8(Function):
    """FP8 forward (eval-only). Backward raises until Stage 1-C."""

    @staticmethod
    def forward(ctx, value_fp8: torch.Tensor, spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor, sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor, value_scale: float) -> torch.Tensor:
        # Crucial: do NOT type_as(value) on sampling_locations / attention_weights.
        # Caller must pass float32.
        assert value_fp8.dtype == torch.float8_e4m3fn, \
            f"value must be float8_e4m3fn, got {value_fp8.dtype}"
        assert sampling_locations.dtype == torch.float32, \
            f"sampling_locations must be float32, got {sampling_locations.dtype}"
        assert attention_weights.dtype == torch.float32, \
            f"attention_weights must be float32, got {attention_weights.dtype}"
        return _ext.ms_deform_attn_fp8_forward(
            value_fp8, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, float(value_scale))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise RuntimeError(
            "FP8 backward is Stage 1-C (not yet implemented). "
            "Use FP32 path for training."
        )


def fp8_forward(value_fp32: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
    """Convenience: quantize value then call FP8 kernel.

    value_fp32: original FP32 value tensor
    sampling_locations / attention_weights: must be float32
    Returns: (bs, num_query, num_heads * channels) float32
    """
    value_fp8, scale = quantize_fp8_per_tensor(value_fp32)
    return MultiScaleDeformableAttnFunctionFp8.apply(
        value_fp8, spatial_shapes, level_start_index,
        sampling_locations, attention_weights, scale)
