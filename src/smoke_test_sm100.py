"""sm capability smoke (auto-detect) — common to Spark sm_121 / B200 sm_100.

Builds + runs a minimal vector add CUDA kernel with cpp_extension and checks that a plain
CUDA extension works on the current GPU capability. The gate before starting on the
FP4/FP8 kernels themselves.
"""
import os
import sys
import time
import traceback
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SRC_DIR, os.pardir, "build", "smoke_vadd")
os.makedirs(BUILD_DIR, exist_ok=True)

# Auto-detect: Spark → "121", B200 → "100"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from _arch_cflags import cuda_cflags, cuda_arch

print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"arch_list (PyTorch wheel)={torch.cuda.get_arch_list()}")
print(f"device capability={torch.cuda.get_device_capability()}  → arch=\"{cuda_arch()}\"")
print(f"device name={torch.cuda.get_device_name()}")
print(f"cflags={cuda_cflags()}")

print("\n--- cpp_extension.load() (build + import) ---")
t0 = time.time()
try:
    smoke = load(
        name="smoke_vadd",
        sources=[os.path.join(SRC_DIR, "smoke_vector_add.cu")],
        build_directory=BUILD_DIR,
        extra_cuda_cflags=cuda_cflags(),
        verbose=True,
    )
    print(f"BUILD+LOAD OK in {time.time()-t0:.1f}s")
except Exception as e:
    print("BUILD/LOAD FAILED")
    traceback.print_exc()
    sys.exit(1)

print("\n--- execute on CUDA ---")
try:
    n = 1024
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.arange(n, 0, -1, dtype=torch.float32, device="cuda")
    c = smoke.vector_add(a, b)
    torch.cuda.synchronize()
    expected = a + b
    if torch.allclose(c, expected):
        print(f"EXEC OK, c[0]={c[0].item()}, c[-1]={c[-1].item()} (expected all {a[0].item()+b[0].item()})")
        print(f"SM{cuda_arch()} cpp_extension OK")
        sys.exit(0)
    else:
        diff = (c - expected).abs().max().item()
        print(f"EXEC MISMATCH, max abs diff={diff}")
        sys.exit(2)
except Exception as e:
    print("EXEC FAILED")
    traceback.print_exc()
    sys.exit(3)
