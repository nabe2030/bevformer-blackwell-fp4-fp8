"""Auto-detect GPU capability and produce cuda_cflags for cpp_extension.load.

A small helper that replaces the `-gencode=arch=compute_121,code=sm_121` that was
hardcoded for Spark sm_121 with automatic detection from the device.

Behavior:
  - uses `get_device_capability()` when torch.cuda.is_available() == True
  - joins (major, minor) into an arch string (e.g. (12,1) → "121", (10,0) → "100")
  - returns `-gencode=arch=compute_<arch>,code=sm_<arch>` + common optimization flags
  - falls back to "121" when there is no GPU (for a logic dry-check; no real build)

Bit-equivalence on Spark:
  On Spark sm_121, cap=(12,1) → "121" → produces the same string as the original hardcode.
  ninja detects the source change and rebuilds, but the output .so is functionally identical.
"""
import torch


def cuda_arch() -> str:
    """Return device capability as concatenated string, e.g. "121" or "100"."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return f"{cap[0]}{cap[1]}"
    return "121"  # fallback (no-GPU dry-check)


def cuda_cflags(extra: list = None) -> list:
    """Return list of CUDA cflags: -gencode for current arch + -O3 + --use_fast_math.

    Optionally append `extra` flags.
    """
    arch = cuda_arch()
    cflags = [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-O3",
        "--use_fast_math",
    ]
    if extra:
        cflags = cflags + list(extra)
    return cflags


if __name__ == "__main__":
    # Dry-check (no GPU → expect "121" fallback)
    print(f"cuda.is_available={torch.cuda.is_available()}")
    print(f"arch={cuda_arch()}")
    print(f"cflags={cuda_cflags()}")
