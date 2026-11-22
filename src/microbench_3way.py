"""Stage 1-B-6 microbench — FP32 / FP8 / FP4 (3-way).

Extends microbench.py (1-A-6) with FP4. Same workloads.
Kernel-only timing via torch.cuda.Event. Quantize cost excluded from timing.
"""
import os
import subprocess
import statistics
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_FP8  = os.path.join(SRC_DIR, os.pardir, "build", "msda_fp8")
BUILD_FP32 = os.path.join(SRC_DIR, os.pardir, "build", "msda_fp32")
BUILD_FP4  = os.path.join(SRC_DIR, os.pardir, "build", "msda_fp4")
for d in (BUILD_FP8, BUILD_FP32, BUILD_FP4):
    os.makedirs(d, exist_ok=True)

CFLAGS = ["-gencode=arch=compute_121,code=sm_121", "-O3", "--use_fast_math"]

print("--- Loading FP32 twin kernel ---")
ext_fp32 = load(name="msda_fp32",
                sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp32_kernel.cu")],
                build_directory=BUILD_FP32, extra_cuda_cflags=CFLAGS, verbose=False)
print("--- Loading FP8 kernel ---")
ext_fp8 = load(name="msda_fp8",
               sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp8_kernel.cu")],
               build_directory=BUILD_FP8, extra_cuda_cflags=CFLAGS, verbose=False)
print("--- Loading FP4 kernel ---")
ext_fp4 = load(name="msda_fp4",
               sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp4_kernel.cu")],
               build_directory=BUILD_FP4, extra_cuda_cflags=CFLAGS, verbose=False)


def print_gpu_coexistence():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True).strip()
        print(f"--- GPU coexistence ---\n{out or '(none)'}\n")
    except Exception as e:
        print(f"--- GPU coexistence query failed: {e} ---\n")


def quantize_fp8(value):
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 448.0).to(torch.float32)
    v_fp8 = (value / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return v_fp8, float(scale.item())


def quantize_fp4(value):
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 6.0).to(torch.float32)
    v_packed = ext_fp4.quantize_pack_fp4(value.contiguous(), float(scale.item()))
    return v_packed, float(scale.item())


def make_inputs(bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    torch.manual_seed(0)
    spatial_size = sum(h * w for h, w in spatial_shapes_list)
    num_levels = len(spatial_shapes_list)
    value_fp32 = torch.randn(bs, spatial_size, num_heads, channels,
                             device='cuda', dtype=torch.float32)
    sampling_loc = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                              device='cuda', dtype=torch.float32)
    aw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                     device='cuda', dtype=torch.float32).softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous()
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    if num_levels > 1:
        cums = [0]
        for h, w in spatial_shapes_list[:-1]:
            cums.append(cums[-1] + h * w)
        lsi = torch.tensor(cums, dtype=torch.int64, device='cuda')
    else:
        lsi = torch.zeros(num_levels, dtype=torch.int64, device='cuda')
    return value_fp32, sampling_loc, aw, spatial_shapes, lsi, spatial_size


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    se = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        se[i].record(); fn(); ee[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(se, ee))
    return {
        'median': statistics.median(t),
        'mean':   statistics.mean(t),
        'p10':    t[len(t)//10],
        'p90':    t[9*len(t)//10],
        'min':    t[0],
        'max':    t[-1],
        'stdev':  statistics.stdev(t),
    }


def run_workload(name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    print(f"\n========== {name} ==========")
    print(f"bs={bs} q={num_query} h={num_heads} c={channels} L={len(spatial_shapes_list)} P={num_points}")
    print(f"spatial_shapes={spatial_shapes_list}")
    value_fp32, sampling_loc, aw, ss, lsi, ssize = make_inputs(
        bs, num_query, num_heads, channels, spatial_shapes_list, num_points)

    value_fp8, s_fp8 = quantize_fp8(value_fp32)
    value_fp4, s_fp4 = quantize_fp4(value_fp32)

    nbytes_fp32 = value_fp32.numel() * 4
    nbytes_fp8 = value_fp32.numel()
    nbytes_fp4 = value_fp32.numel() // 2
    print(f"value memory: FP32={nbytes_fp32/1024/1024:.2f} MiB, "
          f"FP8={nbytes_fp8/1024/1024:.2f} MiB, FP4={nbytes_fp4/1024/1024:.2f} MiB "
          f"(spatial_size={ssize})")

    def call_fp32(): ext_fp32.ms_deform_attn_fp32_forward(value_fp32, ss, lsi, sampling_loc, aw)
    def call_fp8():  ext_fp8.ms_deform_attn_fp8_forward(value_fp8, ss, lsi, sampling_loc, aw, s_fp8)
    def call_fp4():  ext_fp4.ms_deform_attn_fp4_forward(value_fp4, channels, ss, lsi, sampling_loc, aw, s_fp4)

    s32 = bench(call_fp32)
    s8  = bench(call_fp8)
    s4  = bench(call_fp4)

    sp_fp8 = s32['median'] / s8['median']
    sp_fp4 = s32['median'] / s4['median']
    sp_fp4_over_fp8 = s8['median'] / s4['median']

    print(f"\nFP32:  median={s32['median']:.4f} ms  [p10={s32['p10']:.4f}, p90={s32['p90']:.4f}]  stdev={s32['stdev']:.4f}")
    print(f"FP8:   median={s8['median']:.4f} ms  [p10={s8['p10']:.4f}, p90={s8['p90']:.4f}]  stdev={s8['stdev']:.4f}")
    print(f"FP4:   median={s4['median']:.4f} ms  [p10={s4['p10']:.4f}, p90={s4['p90']:.4f}]  stdev={s4['stdev']:.4f}")
    print(f"speedup vs FP32:   FP8={sp_fp8:.3f}×   FP4={sp_fp4:.3f}×")
    print(f"speedup FP4 vs FP8:  {sp_fp4_over_fp8:.3f}×")

    return {
        'name': name, 'bs': bs, 'q': num_query, 'h': num_heads, 'c': channels,
        'L': len(spatial_shapes_list), 'P': num_points, 'spatial_size': ssize,
        'mib_fp32': nbytes_fp32/1024/1024,
        'mib_fp8':  nbytes_fp8/1024/1024,
        'mib_fp4':  nbytes_fp4/1024/1024,
        's_fp32': s32, 's_fp8': s8, 's_fp4': s4,
        'sp_fp8': sp_fp8, 'sp_fp4': sp_fp4, 'sp_fp4_over_fp8': sp_fp4_over_fp8,
    }


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name()}, "
          f"capability={torch.cuda.get_device_capability()}")
    print_gpu_coexistence()

    workloads = [
        ('tiny  (BEVFormer tiny)',  2, 100,  8, 32, [(50, 50)], 8),
        ('base  (multi-level)',     2, 900,  8, 64, [(100,100), (50,50), (25,25), (13,13)], 4),
        ('large (encoder-scale)',   2, 2500, 8, 64, [(200,200), (100,100), (50,50), (25,25)], 4),
    ]
    results = []
    for w in workloads:
        try:
            results.append(run_workload(*w))
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[SKIP {w[0]}] OOM: {e}")
            results.append({'name': w[0], 'oom': str(e)})

    # Summary table
    print("\n\n=========================================================")
    print("|| Summary (median ms; speedup vs FP32)")
    print("=========================================================")
    print(f"{'workload':<26} {'MiB f32/f8/f4':<22} {'FP32 ms':>9} {'FP8 ms':>9} {'FP4 ms':>9} {'spFP8':>8} {'spFP4':>8} {'spFP4/8':>9}")
    for r in results:
        if r.get('oom'):
            print(f"{r['name']:<26} OOM"); continue
        mib_s = f"{r['mib_fp32']:.1f}/{r['mib_fp8']:.1f}/{r['mib_fp4']:.1f}"
        print(f"{r['name']:<26} {mib_s:<22} "
              f"{r['s_fp32']['median']:>9.4f} "
              f"{r['s_fp8']['median']:>9.4f} "
              f"{r['s_fp4']['median']:>9.4f} "
              f"{r['sp_fp8']:>7.3f}× "
              f"{r['sp_fp4']:>7.3f}× "
              f"{r['sp_fp4_over_fp8']:>8.3f}×")


if __name__ == '__main__':
    main()
