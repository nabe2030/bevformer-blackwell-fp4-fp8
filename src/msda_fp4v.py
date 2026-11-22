"""Stage 1-B'-a FP4-vector MultiScaleDeformableAttention forward.

Drop-in replacement for msda_fp4 with 1 thread = 2 channels + vector decode.
Quantize storage layout is identical to 1-B (uint8 packed, 2-per-byte).
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.autograd.function import Function, once_differentiable
from torch.utils.cpp_extension import load

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_SRC_DIR, os.pardir, "build", "msda_fp4v")
os.makedirs(_BUILD_DIR, exist_ok=True)

# Auto-detect GPU capability for -gencode (shared by Spark sm_121 / B200 sm_100)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from _arch_cflags import cuda_cflags

_ext = load(
    name="msda_fp4v",
    sources=[os.path.join(_SRC_DIR, "ms_deform_attn_fp4v_kernel.cu")],
    build_directory=_BUILD_DIR,
    extra_cuda_cflags=cuda_cflags(),
    verbose=False,
)


def quantize_fp4v_per_tensor(value: torch.Tensor):
    """Identical storage layout to 1-B's quantize_fp4_per_tensor."""
    assert value.dim() >= 1 and value.shape[-1] % 2 == 0, \
        f"last dim must be even for FP4 packing; got shape={value.shape}"
    assert value.is_cuda
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 6.0).to(torch.float32)
    value_packed = _ext.quantize_pack_fp4v(value.to(torch.float32).contiguous(), float(scale.item()))
    return value_packed, float(scale.item())


class MultiScaleDeformableAttnFunctionFp4v(Function):
    @staticmethod
    def forward(ctx, value_packed: torch.Tensor, channels: int,
                spatial_shapes: torch.Tensor, level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor, attention_weights: torch.Tensor,
                value_scale: float) -> torch.Tensor:
        assert value_packed.dtype == torch.uint8
        assert sampling_locations.dtype == torch.float32
        assert attention_weights.dtype == torch.float32
        return _ext.ms_deform_attn_fp4v_forward(
            value_packed, channels, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, float(value_scale))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise RuntimeError(
            "FP4 backward is Stage 1-C (not yet implemented). Use FP32 for training."
        )


def fp4v_forward(value_fp32: torch.Tensor,
                 spatial_shapes: torch.Tensor,
                 level_start_index: torch.Tensor,
                 sampling_locations: torch.Tensor,
                 attention_weights: torch.Tensor) -> torch.Tensor:
    channels = value_fp32.shape[-1]
    value_packed, scale = quantize_fp4v_per_tensor(value_fp32)
    return MultiScaleDeformableAttnFunctionFp4v.apply(
        value_packed, channels, spatial_shapes, level_start_index,
        sampling_locations, attention_weights, scale)
