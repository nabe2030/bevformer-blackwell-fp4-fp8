"""Stage 1-C-a FP8 MultiScaleDeformableAttention with backward (training).

Design: QAT route, FP32 grad_value staging
  (FP8 atomic non-supported, FP32 staging only realistic)

Forward: reuses existing msda_fp8._ext.ms_deform_attn_fp8_forward (1-A).
Backward: new ext binding ms_deform_attn_fp8_backward (FP32 grad_value staging).

Scale handling (STE / straight-through estimator):
  forward:  output = scale * sum_i (w_i * v_i_fp8),
            where v_i_fp8 ≈ v_i_fp32 / scale (quantize, ignoring round/clip).
  backward (STE): treat v_fp8 = v_fp32 / scale as identity for ∂v_fp8/∂v_fp32 = 1/scale,
            so ∂output/∂v_fp32 = scale * w_i * (1/scale) = w_i.
            Kernel accumulates `w_i * top_grad` directly (no scale multiply).
            Python wrapper returns grad_value without scale adjustment.

Eval-only (1-A) coexistence:
  msda_fp8.MultiScaleDeformableAttnFunctionFp8 keeps raising on backward.
  This module provides MultiScaleDeformableAttnFunctionFp8Trainable for training.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.autograd.function import Function
from torch.utils.cpp_extension import load

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp8_backward")
os.makedirs(_BUILD_DIR, exist_ok=True)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from _arch_cflags import cuda_cflags

# Backward ext (new). NOTE: `name` must NOT collide with this .py filename;
# cpp_extension.load registers `name` in sys.modules, which would shadow the
# .py file when re-imported (observed on RunPod B200; fix: use _ext suffix).
_bwd_ext = load(
    name="msda_fp8_backward_ext",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp8_backward_kernel.cu")],
    build_directory=_BUILD_DIR,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)

# Forward ext (private build; do NOT import msda_fp8 to avoid name collision
# with other msda_fp8 modules that may exist on sys.path inside containers).
_BUILD_FWD = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp8_backward_fwd")
os.makedirs(_BUILD_FWD, exist_ok=True)
_fwd_ext = load(
    name="msda_fp8_backward_fwd",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp8_kernel.cu")],
    build_directory=_BUILD_FWD,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)


def quantize_fp8_per_tensor(value: torch.Tensor):
    """FP32 → FP8 e4m3 with per-tensor scale (inlined from msda_fp8.py).

    scale = max(|value|) / 448 (FP8 e4m3 max representable).
    Returns (value_fp8, scale_python_float).
    """
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 448.0).to(torch.float32)
    value_fp8 = (value.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return value_fp8.contiguous(), float(scale.item())


class MultiScaleDeformableAttnFunctionFp8Trainable(Function):
    """FP8 forward + backward (QAT-compatible training)."""

    @staticmethod
    def forward(ctx, value_fp32: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        # type guards (matching 1-A; sampling_loc / attn_weight MUST stay float32).
        assert sampling_locations.dtype == torch.float32, \
            f"sampling_locations must be float32, got {sampling_locations.dtype}"
        assert attention_weights.dtype == torch.float32, \
            f"attention_weights must be float32, got {attention_weights.dtype}"

        # Quantize FP32 → FP8 (per-tensor scale).
        value_fp8, scale = quantize_fp8_per_tensor(value_fp32)

        # Forward: reuse 1-A binding.
        out = _fwd_ext.ms_deform_attn_fp8_forward(
            value_fp8, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, float(scale))

        # Save for backward.
        ctx.save_for_backward(value_fp8, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)
        ctx.scale = scale
        ctx.value_shape = value_fp32.shape
        ctx.value_dtype = value_fp32.dtype
        ctx.value_device = value_fp32.device
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value_fp8, spatial_shapes, level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors

        # ★ FP32 staging buffer for grad_value (mandatory, FP8 atomic non-supported).
        grad_value = torch.zeros(ctx.value_shape, dtype=torch.float32,
                                  device=ctx.value_device)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        # grad_output must be contiguous float32; squeeze the last dim grouping.
        # forward returns (bs, num_query, num_heads * channels). backward kernel
        # expects same flat layout (grad_col[index] where index = b*q*m*c + m*c + c).
        grad_output_c = grad_output.contiguous().to(torch.float32)

        _bwd_ext.ms_deform_attn_fp8_backward(
            value_fp8, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output_c,
            grad_value, grad_sampling_loc, grad_attn_weight)

        # Scale handling — corrected after Phase 1-C-3 debug (c):
        #   grad_value: STE chain rule cancels scale (forward's scale * w_i and
        #     quantize's 1/scale cancel → w_i directly). NO multiply needed.
        #   grad_sampling_loc / grad_attn_weight: kernel computes `val = sum(w_i * v_i_fp8)`
        #     and `grad_h/w_weight = ... * v_i_fp8` using DEQUANT-WITHOUT-SCALE v values.
        #     To restore the correct scaled values, multiply these grads by `scale` here.
        #   (forward: output = scale * sum(w_i * v_i_fp8) → ∂output/∂attn_w = scale * val_unscaled;
        #    chain rule: ∂loss/∂attn_w = top_grad * scale * val_unscaled)
        ctx_scale = ctx.scale
        grad_sampling_loc.mul_(ctx_scale)
        grad_attn_weight.mul_(ctx_scale)

        return (
            grad_value.to(ctx.value_dtype),
            None,
            None,
            grad_sampling_loc,
            grad_attn_weight,
        )


def fp8_trainable(value_fp32: torch.Tensor,
                  spatial_shapes: torch.Tensor,
                  level_start_index: torch.Tensor,
                  sampling_locations: torch.Tensor,
                  attention_weights: torch.Tensor) -> torch.Tensor:
    """Convenience: FP8 forward+backward end-to-end.

    Inputs: original FP32 value + float32 sampling_loc/attn_weight.
    Returns: (bs, num_query, num_heads * channels) float32.
    Backward: grad_value (FP32 → caller dtype), grad_sampling_loc, grad_attn_weight.
    """
    return MultiScaleDeformableAttnFunctionFp8Trainable.apply(
        value_fp32, spatial_shapes, level_start_index,
        sampling_locations, attention_weights)
