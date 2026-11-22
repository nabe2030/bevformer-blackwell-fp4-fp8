"""FP32 reference: my forward FP32 + new mmcv-ported FP32 backward.

For Phase 1-C-3 debug (c): replace autograd F.grid_sample ref with mmcv's own
backward to get the *correct* comparison baseline. The FP8 path should match
this within FP8 quantization noise (~2.5%), regardless of autograd grad_sample
scale convention differences.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.autograd.function import Function
from torch.utils.cpp_extension import load

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_FWD = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp32_fwd")
_BUILD_BWD = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp32_bwd")
os.makedirs(_BUILD_FWD, exist_ok=True)
os.makedirs(_BUILD_BWD, exist_ok=True)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from _arch_cflags import cuda_cflags

_fwd_ext = load(
    name="msda_fp32_ref_fwd",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp32_kernel.cu")],
    build_directory=_BUILD_FWD,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)

_bwd_ext = load(
    name="msda_fp32_ref_bwd",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp32_backward_kernel.cu")],
    build_directory=_BUILD_BWD,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)


class MultiScaleDeformableAttnFunctionFp32Ref(Function):
    """FP32 forward + backward (port of mmcv's own kernel, used as ref)."""

    @staticmethod
    def forward(ctx, value: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        assert value.dtype == torch.float32, f"value must be float32, got {value.dtype}"
        out = _fwd_ext.ms_deform_attn_fp32_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights)
        ctx.save_for_backward(value, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value, spatial_shapes, level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors

        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        _bwd_ext.ms_deform_attn_fp32_backward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output.contiguous(),
            grad_value, grad_sampling_loc, grad_attn_weight)

        return (grad_value, None, None, grad_sampling_loc, grad_attn_weight)


def fp32_ref(value, spatial_shapes, level_start_index,
             sampling_locations, attention_weights):
    return MultiScaleDeformableAttnFunctionFp32Ref.apply(
        value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights)
