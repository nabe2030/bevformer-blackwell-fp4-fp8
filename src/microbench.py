"""Stage 1-A-6 microbench — FP32 twin baseline vs FP8 kernel.

Methodology:
- Both kernels loaded via cpp_extension.load (separate build dirs)
- Kernel-only timing via torch.cuda.Event(enable_timing=True)
- warmup 20 + measure 100, report median + p10/p90 + min/max
- 3 workload sizes (tiny / base / large) to test scale dependence
- quantize_fp8 and output.mul_(scale) timing recorded separately (kernel comparison
  is the main focus)
- llama-server coexistence noted (Spark borrowed)
"""
import os
import subprocess
import statistics
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_FP8 = os.path.join(SRC_DIR, os.pardir, "build", "msda_fp8")
BUILD_FP32 = os.path.join(SRC_DIR, os.pardir, "build", "msda_fp32")
os.makedirs(BUILD_FP8, exist_ok=True)
os.makedirs(BUILD_FP32, exist_ok=True)

print("--- Loading FP8 kernel ---")
ext_fp8 = load(
    name="msda_fp8",
    sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp8_kernel.cu")],
    build_directory=BUILD_FP8,
    extra_cuda_cflags=["-gencode=arch=compute_121,code=sm_121", "-O3", "--use_fast_math"],
    verbose=False,
)
print("--- Loading FP32 twin kernel ---")
ext_fp32 = load(
    name="msda_fp32",
    sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp32_kernel.cu")],
    build_directory=BUILD_FP32,
    extra_cuda_cflags=["-gencode=arch=compute_121,code=sm_121", "-O3", "--use_fast_math"],
    verbose=False,
)


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


def make_inputs(name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    """Build (value_fp32, sampling_loc, attn_weight, spatial_shapes, level_start_index)."""
    torch.manual_seed(0)
    spatial_size = sum(h * w for h, w in spatial_shapes_list)
    num_levels = len(spatial_shapes_list)

    value_fp32 = torch.randn(bs, spatial_size, num_heads, channels,
                             device='cuda', dtype=torch.float32)
    sampling_loc = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                              device='cuda', dtype=torch.float32)
    aw_raw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                         device='cuda', dtype=torch.float32)
    attn_weight = aw_raw.softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous()

    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    level_start_index = torch.zeros(num_levels, dtype=torch.int64, device='cuda')
    if num_levels > 1:
        cums = [0]
        for h, w in spatial_shapes_list[:-1]:
            cums.append(cums[-1] + h * w)
        level_start_index = torch.tensor(cums, dtype=torch.int64, device='cuda')

    return value_fp32, sampling_loc, attn_weight, spatial_shapes, level_start_index, spatial_size


def bench(fn, warmup=20, iters=100):
    """Kernel-only timing via cuda.Event. Returns dict of stats (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_evt = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_evt[i].record()
        fn()
        end_evt[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]
    times_ms.sort()
    return {
        'median': statistics.median(times_ms),
        'mean':   statistics.mean(times_ms),
        'p10':    times_ms[len(times_ms) // 10],
        'p90':    times_ms[9 * len(times_ms) // 10],
        'min':    times_ms[0],
        'max':    times_ms[-1],
        'stdev':  statistics.stdev(times_ms),
    }


def run_workload(name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    print(f"\n========== {name} ==========")
    print(f"bs={bs} num_query={num_query} num_heads={num_heads} channels={channels} "
          f"num_levels={len(spatial_shapes_list)} num_points={num_points} "
          f"spatial_shapes={spatial_shapes_list}")
    value_fp32, sampling_loc, attn_weight, spatial_shapes, level_start_index, ssize = make_inputs(
        name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points)
    print(f"spatial_size (sum H*W) = {ssize}")
    print(f"value memory FP32 = {value_fp32.numel() * 4 / 1024 / 1024:.2f} MiB, "
          f"FP8 = {value_fp32.numel() / 1024 / 1024:.2f} MiB")

    # Pre-quantize for FP8 path (NOT included in kernel timing — per design)
    value_fp8, scale = quantize_fp8(value_fp32)

    # FP32 baseline (twin)
    def call_fp32():
        ext_fp32.ms_deform_attn_fp32_forward(
            value_fp32, spatial_shapes, level_start_index, sampling_loc, attn_weight)

    # FP8 kernel (kernel only — does NOT include quantize and includes the
    # output.mul_(scale) inside the kernel binding, which is part of the kernel cost)
    def call_fp8():
        ext_fp8.ms_deform_attn_fp8_forward(
            value_fp8, spatial_shapes, level_start_index, sampling_loc, attn_weight, scale)

    stats_fp32 = bench(call_fp32)
    stats_fp8 = bench(call_fp8)

    print(f"\nFP32 baseline:  median={stats_fp32['median']:.4f} ms  "
          f"[p10={stats_fp32['p10']:.4f}, p90={stats_fp32['p90']:.4f}]  "
          f"min={stats_fp32['min']:.4f}, max={stats_fp32['max']:.4f}, stdev={stats_fp32['stdev']:.4f}")
    print(f"FP8 kernel:     median={stats_fp8['median']:.4f} ms  "
          f"[p10={stats_fp8['p10']:.4f}, p90={stats_fp8['p90']:.4f}]  "
          f"min={stats_fp8['min']:.4f}, max={stats_fp8['max']:.4f}, stdev={stats_fp8['stdev']:.4f}")

    speedup_median = stats_fp32['median'] / stats_fp8['median']
    print(f"\nspeedup (FP32 median / FP8 median) = {speedup_median:.3f}×")

    return {
        'name': name,
        'bs': bs, 'num_query': num_query, 'num_heads': num_heads, 'channels': channels,
        'num_levels': len(spatial_shapes_list), 'num_points': num_points,
        'spatial_size': ssize,
        'value_MiB_fp32': value_fp32.numel() * 4 / 1024 / 1024,
        'value_MiB_fp8': value_fp32.numel() / 1024 / 1024,
        'fp32_stats_ms': stats_fp32,
        'fp8_stats_ms': stats_fp8,
        'speedup_median': speedup_median,
    }


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name()}, "
          f"capability={torch.cuda.get_device_capability()}")
    print_gpu_coexistence()

    workloads = [
        # (name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points)
        ('tiny  (BEVFormer tiny)',  2, 100,  8, 32, [(50, 50)], 8),
        ('base  (multi-level)',     2, 900,  8, 64, [(100,100), (50,50), (25,25), (13,13)], 4),
        ('large (encoder-scale)',   2, 2500, 8, 64, [(200,200), (100,100), (50,50), (25,25)], 4),
    ]

    results = []
    for w in workloads:
        try:
            r = run_workload(*w)
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[SKIP] {w[0]}: OOM ({e})")
            results.append({'name': w[0], 'oom': str(e)})

    # Summary table
    print("\n\n=========================================================")
    print("|| Summary (median μs, speedup = FP32 / FP8)")
    print("=========================================================")
    print(f"{'workload':<26} {'value MiB':>10} {'FP32 ms':>10} {'FP8 ms':>10} {'speedup':>10}")
    for r in results:
        if r.get('oom'):
            print(f"{r['name']:<26} {'OOM':>10}")
            continue
        print(f"{r['name']:<26} {r['value_MiB_fp32']:>10.2f} "
              f"{r['fp32_stats_ms']['median']:>10.4f} "
              f"{r['fp8_stats_ms']['median']:>10.4f} "
              f"{r['speedup_median']:>9.3f}×")


if __name__ == '__main__':
    main()
