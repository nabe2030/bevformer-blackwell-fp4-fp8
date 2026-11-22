"""smoke_fp4_test.py — try multiple build flags for FP4 cvt smoke on sm_121.

Goal: determine which `-gencode` flag works for FP4 cvt on Spark sm_121.
Background: cvt FP4 may be arch-specific (sm_121a / sm_120a vs base sm_121 / sm_120).
This smoke does NOT use tensor core (tcgen05/MMA) — only cvt instructions.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SRC_DIR, "smoke_fp4_cvt.cu")
BUILD_BASE = os.path.join(SRC_DIR, os.pardir, "build")

# 4 build flags to try, in order from "most likely if SM121 supports plain cvt"
# to fallback. Each gets its own build_directory.
FLAGS = [
    ("A_sm121",  "-gencode=arch=compute_121,code=sm_121"),
    ("B_sm121a", "-gencode=arch=compute_121a,code=sm_121a"),
    ("C_sm120a", "-gencode=arch=compute_120a,code=sm_120a"),
    ("D_sm120",  "-gencode=arch=compute_120,code=sm_120"),
]


def try_build_and_run(flag_name, gencode):
    print(f"\n========== {flag_name}: {gencode} ==========", flush=True)
    build_dir = os.path.join(BUILD_BASE, f"fp4_cvt_{flag_name}")
    os.makedirs(build_dir, exist_ok=True)
    result = {
        "flag": flag_name, "gencode": gencode,
        "build": None, "exec_scalar": None, "exec_vec4": None,
        "build_err": None, "exec_err": None,
        "scalar_in": None, "scalar_out": None,
        "vec4_in": None, "vec4_out": None,
    }
    try:
        ext = load(
            name=f"smoke_fp4_{flag_name}",
            sources=[SRC],
            build_directory=build_dir,
            extra_cuda_cflags=[gencode, "-O3"],
            verbose=True,
        )
        result["build"] = "OK"
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:600]}"
        result["build"] = "FAIL"
        result["build_err"] = msg
        print(f"[BUILD FAIL] {msg}", flush=True)
        return result

    # Test scalar round-trip with a grid of representative values
    try:
        # FP4 e2m1 grid: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
        x = torch.tensor(
            [ 0.0,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  4.0,
              5.0,  6.0,  7.0, -1.5, -2.5,  8.0, 12.0, -8.0],
            dtype=torch.float32, device='cuda')
        y = ext.fp4_roundtrip_scalar(x)
        torch.cuda.synchronize()
        result["exec_scalar"] = "OK"
        result["scalar_in"] = x.cpu().tolist()
        result["scalar_out"] = y.cpu().tolist()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        result["exec_scalar"] = "FAIL"
        result["exec_err"] = msg
        print(f"[EXEC scalar FAIL] {msg}", flush=True)

    # Test vec4 round-trip
    try:
        x4 = torch.tensor(
            [ 0.5,  1.0,  1.5,  2.0,
              3.0,  4.0,  6.0,  0.0,
             -1.0, -2.0, -3.0, -4.0,
              2.5, -2.5,  5.0,  7.0],
            dtype=torch.float32, device='cuda')
        y4 = ext.fp4_roundtrip_vec4(x4)
        torch.cuda.synchronize()
        result["exec_vec4"] = "OK"
        result["vec4_in"] = x4.cpu().tolist()
        result["vec4_out"] = y4.cpu().tolist()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        result["exec_vec4"] = "FAIL"
        print(f"[EXEC vec4 FAIL] {msg}", flush=True)

    return result


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name()}, "
          f"capability={torch.cuda.get_device_capability()}", flush=True)

    results = []
    for fname, gc in FLAGS:
        try:
            r = try_build_and_run(fname, gc)
        except Exception as e:
            r = {"flag": fname, "gencode": gc, "fatal": str(e)[:300]}
            print(f"[FATAL] {fname}: {e}", flush=True)
        results.append(r)

    # Summary table
    print("\n\n========== SUMMARY ==========")
    print(f"{'flag':<10} {'gencode':<55} {'build':<6} {'scalar':<6} {'vec4':<6}")
    for r in results:
        bld = r.get("build", "?")
        sc = r.get("exec_scalar") if r.get("exec_scalar") else "-"
        vc = r.get("exec_vec4") if r.get("exec_vec4") else "-"
        print(f"{r['flag']:<10} {r['gencode'][:55]:<55} {bld:<6} {sc:<6} {vc:<6}")

    # Show details for the first successful flag (= use this one for 1-B)
    first_ok = None
    for r in results:
        if r.get("exec_scalar") == "OK":
            first_ok = r
            break

    if first_ok:
        print(f"\n--- Round-trip detail (scalar, {first_ok['flag']}) ---")
        print(f"FP4 e2m1 representable grid: ±{{0, 0.5, 1, 1.5, 2, 3, 4, 6}}")
        for inv, outv in zip(first_ok["scalar_in"], first_ok["scalar_out"]):
            print(f"  in={inv:>+7.3f}  out={outv:>+7.3f}  diff={outv-inv:+.3f}")
        if first_ok.get("exec_vec4") == "OK":
            print(f"\n--- Round-trip detail (vec4, {first_ok['flag']}) ---")
            for inv, outv in zip(first_ok["vec4_in"], first_ok["vec4_out"]):
                print(f"  in={inv:>+7.3f}  out={outv:>+7.3f}  diff={outv-inv:+.3f}")

    # Show failure details
    for r in results:
        if r.get("build") == "FAIL":
            print(f"\n--- Build FAIL detail ({r['flag']}) ---")
            print(r.get("build_err", "")[:800])
        elif r.get("exec_scalar") == "FAIL":
            print(f"\n--- Exec FAIL detail ({r['flag']}) ---")
            print(r.get("exec_err", "")[:300])

    if not first_ok:
        print("\n[ALL FLAGS FAILED] No working flag for FP4 cvt on this Spark.")
        return 1
    print(f"\n[SUCCESS] Use {first_ok['flag']} ({first_ok['gencode']}) for Stage 1-B kernel.")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
