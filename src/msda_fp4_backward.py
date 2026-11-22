"""Stage 1-C-b FP4 MultiScaleDeformableAttention with backward (training).

Design: follows 1-C-a (FP8 backward, msda_fp8_backward.py), changing FP8 → packed FP4.

Lessons applied from the start (avoiding the 2 bugs hit in 1-C-a):
1. ext name suffix: cpp_extension.load(name="..._ext") differs from the .py filename
   (the sys.modules overwrite bug hit in the 2nd attempt of 1-C-a)
2. scale restoration: mul_(scale) on grad_sampling_loc / grad_attn_weight from the start
   (the 88x bug not noticed until the 4th attempt of 1-C-a; the STE chain rule covers grad_value only)
3. independent build: does not import the msda_fp4 module; has its own ext for quantize_pack
   (avoids name collisions with other modules inside the pod)
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.autograd.function import Function
from torch.utils.cpp_extension import load

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_BWD = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp4_backward")
_BUILD_FWD = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp4_backward_fwd")
os.makedirs(_BUILD_BWD, exist_ok=True)
os.makedirs(_BUILD_FWD, exist_ok=True)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from _arch_cflags import cuda_cflags

# ext name = .py filename + "_ext" (avoids overwriting sys.modules)
_bwd_ext = load(
    name="msda_fp4_backward_ext",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp4_backward_kernel.cu")],
    build_directory=_BUILD_BWD,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)

# Forward ext (independent build, NOT importing msda_fp4 to avoid module collision)
_fwd_ext = load(
    name="msda_fp4_backward_fwd_ext",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp4_kernel.cu")],
    build_directory=_BUILD_FWD,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)


def quantize_fp4_per_tensor(value: torch.Tensor):
    """FP32 → packed FP4 e2m1 with per-tensor scale (inlined from msda_fp4.py).

    scale = max(|value|) / 6  (FP4 e2m1 max representable = 6)
    Returns (value_packed_uint8, scale_python_float).
    """
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 6.0).to(torch.float32)
    value_packed = _fwd_ext.quantize_pack_fp4(value.to(torch.float32).contiguous(),
                                                float(scale.item()))
    return value_packed, float(scale.item())


class MultiScaleDeformableAttnFunctionFp4Trainable(Function):
    """FP4 forward + backward (QAT-compatible training)."""

    @staticmethod
    def forward(ctx, value_fp32: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        assert sampling_locations.dtype == torch.float32
        assert attention_weights.dtype == torch.float32

        value_packed, scale = quantize_fp4_per_tensor(value_fp32)
        channels = value_fp32.shape[3]

        out = _fwd_ext.ms_deform_attn_fp4_forward(
            value_packed, int(channels), spatial_shapes, level_start_index,
            sampling_locations, attention_weights, float(scale))

        ctx.save_for_backward(value_packed, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)
        ctx.scale = scale
        ctx.channels = int(channels)
        ctx.value_shape = value_fp32.shape
        ctx.value_dtype = value_fp32.dtype
        ctx.value_device = value_fp32.device
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value_packed, spatial_shapes, level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors

        # FP32 staging buffer (unpacked shape!)
        grad_value = torch.zeros(ctx.value_shape, dtype=torch.float32,
                                  device=ctx.value_device)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight  = torch.zeros_like(attention_weights)

        grad_output_c = grad_output.contiguous().to(torch.float32)

        _bwd_ext.ms_deform_attn_fp4_backward(
            value_packed, ctx.channels, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output_c,
            grad_value, grad_sampling_loc, grad_attn_weight)

        # Scale restoration (applying the 1-C-a STE chain-rule lesson from the start):
        #   grad_value: STE cancels (forward's scale * w_i and quantize's 1/scale cancel) — no mul
        #   grad_sloc / grad_aw: kernel used scale-UNAPPLIED v → restore by mul_(scale)
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


def fp4_trainable(value_fp32, spatial_shapes, level_start_index,
                  sampling_locations, attention_weights):
    """Convenience: FP4 forward+backward end-to-end."""
    return MultiScaleDeformableAttnFunctionFp4Trainable.apply(
        value_fp32, spatial_shapes, level_start_index,
        sampling_locations, attention_weights)
