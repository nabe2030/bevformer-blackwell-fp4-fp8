"""Stage 1-B'-a 4-way microbench: FP32 / FP8 / FP4(scalar) / FP4v(vector)."""
import os
import subprocess
import statistics
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
def build_dir(name):
    d = os.path.join(SRC_DIR, os.pardir, "build", name)
    os.makedirs(d, exist_ok=True)
    return d

# Auto-detect GPU capability for -gencode (shared by Spark sm_121 / B200 sm_100)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from _arch_cflags import cuda_cflags
CFLAGS = cuda_cflags()

print("--- Loading FP32 twin ---")
ext_fp32 = load(name="msda_fp32",
                sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp32_kernel.cu")],
                build_directory=build_dir("msda_fp32"), extra_cuda_cflags=CFLAGS, verbose=False)
print("--- Loading FP8 ---")
ext_fp8  = load(name="msda_fp8",
                sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp8_kernel.cu")],
                build_directory=build_dir("msda_fp8"),  extra_cuda_cflags=CFLAGS, verbose=False)
print("--- Loading FP4 scalar (1-B) ---")
ext_fp4  = load(name="msda_fp4",
                sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp4_kernel.cu")],
                build_directory=build_dir("msda_fp4"),  extra_cuda_cflags=CFLAGS, verbose=False)
print("--- Loading FP4v vector (1-B'-a) ---")
ext_fp4v = load(name="msda_fp4v",
                sources=[os.path.join(SRC_DIR, "ms_deform_attn_fp4v_kernel.cu")],
                build_directory=build_dir("msda_fp4v"), extra_cuda_cflags=CFLAGS, verbose=False)


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
    v = (value / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return v, float(scale.item())


def quantize_fp4(value):
    abs_max = value.detach().abs().max().clamp_min(1e-12)
    scale = (abs_max / 6.0).to(torch.float32)
    v = ext_fp4.quantize_pack_fp4(value.contiguous(), float(scale.item()))
    return v, float(scale.item())


def make_inputs(bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    torch.manual_seed(0)
    spatial_size = sum(h * w for h, w in spatial_shapes_list)
    num_levels = len(spatial_shapes_list)
    value = torch.randn(bs, spatial_size, num_heads, channels,
                        device='cuda', dtype=torch.float32)
    sl = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                    device='cuda', dtype=torch.float32)
    aw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                     device='cuda', dtype=torch.float32).softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous()
    ss = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    if num_levels > 1:
        cums = [0]
        for h, w in spatial_shapes_list[:-1]:
            cums.append(cums[-1] + h * w)
        lsi = torch.tensor(cums, dtype=torch.int64, device='cuda')
    else:
        lsi = torch.zeros(num_levels, dtype=torch.int64, device='cuda')
    return value, sl, aw, ss, lsi, spatial_size


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
        'p10': t[len(t)//10], 'p90': t[9*len(t)//10],
        'stdev': statistics.stdev(t),
    }


def run_workload(name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    print(f"\n========== {name} ==========")
    print(f"bs={bs} q={num_query} h={num_heads} c={channels} L={len(spatial_shapes_list)} P={num_points}")
    value, sl, aw, ss, lsi, ssize = make_inputs(
        bs, num_query, num_heads, channels, spatial_shapes_list, num_points)
    v_fp8, s_fp8 = quantize_fp8(value)
    v_fp4, s_fp4 = quantize_fp4(value)

    nbytes_fp32 = value.numel() * 4
    nbytes_fp8 = value.numel()
    nbytes_fp4 = value.numel() // 2
    print(f"value memory: FP32={nbytes_fp32/1024/1024:.2f} MiB, "
          f"FP8={nbytes_fp8/1024/1024:.2f} MiB, FP4={nbytes_fp4/1024/1024:.2f} MiB")

    def call_fp32(): ext_fp32.ms_deform_attn_fp32_forward(value, ss, lsi, sl, aw)
    def call_fp8():  ext_fp8.ms_deform_attn_fp8_forward(v_fp8, ss, lsi, sl, aw, s_fp8)
    def call_fp4():  ext_fp4.ms_deform_attn_fp4_forward(v_fp4, channels, ss, lsi, sl, aw, s_fp4)
    def call_fp4v(): ext_fp4v.ms_deform_attn_fp4v_forward(v_fp4, channels, ss, lsi, sl, aw, s_fp4)

    s32  = bench(call_fp32)
    s8   = bench(call_fp8)
    s4   = bench(call_fp4)
    s4v  = bench(call_fp4v)

    sp_fp8  = s32['median'] / s8['median']
    sp_fp4  = s32['median'] / s4['median']
    sp_fp4v = s32['median'] / s4v['median']
    sp_fp4v_over_fp8 = s8['median'] / s4v['median']
    sp_fp4v_over_fp4 = s4['median'] / s4v['median']

    print(f"\nFP32:   median={s32['median']:.4f} ms  stdev={s32['stdev']:.4f}")
    print(f"FP8:    median={s8['median']:.4f} ms  stdev={s8['stdev']:.4f}")
    print(f"FP4:    median={s4['median']:.4f} ms  stdev={s4['stdev']:.4f}")
    print(f"FP4v:   median={s4v['median']:.4f} ms  stdev={s4v['stdev']:.4f}")
    print(f"speedup vs FP32:   FP8={sp_fp8:.3f}×  FP4={sp_fp4:.3f}×  FP4v={sp_fp4v:.3f}×")
    print(f"FP4v vs FP8:  {sp_fp4v_over_fp8:.3f}×    FP4v vs FP4(scalar):  {sp_fp4v_over_fp4:.3f}×")

    return {
        'name': name, 'mib_fp32': nbytes_fp32/1024/1024,
        'mib_fp8': nbytes_fp8/1024/1024, 'mib_fp4': nbytes_fp4/1024/1024,
        's32': s32, 's8': s8, 's4': s4, 's4v': s4v,
        'sp_fp8': sp_fp8, 'sp_fp4': sp_fp4, 'sp_fp4v': sp_fp4v,
        'sp_fp4v_over_fp8': sp_fp4v_over_fp8,
        'sp_fp4v_over_fp4': sp_fp4v_over_fp4,
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

    print("\n\n==================================================================")
    print("|| Summary (median ms; speedup vs FP32; FP4v vs FP8 / FP4 scalar)")
    print("==================================================================")
    header = f"{'workload':<26} {'FP32 ms':>9} {'FP8 ms':>9} {'FP4 ms':>9} {'FP4v ms':>9} {'spFP8':>7} {'spFP4':>7} {'spFP4v':>8} {'4v/8':>7} {'4v/4':>7}"
    print(header)
    for r in results:
        if r.get('oom'):
            print(f"{r['name']:<26} OOM"); continue
        print(f"{r['name']:<26} "
              f"{r['s32']['median']:>9.4f} "
              f"{r['s8']['median']:>9.4f} "
              f"{r['s4']['median']:>9.4f} "
              f"{r['s4v']['median']:>9.4f} "
              f"{r['sp_fp8']:>6.3f}× "
              f"{r['sp_fp4']:>6.3f}× "
              f"{r['sp_fp4v']:>7.3f}× "
              f"{r['sp_fp4v_over_fp8']:>6.3f}× "
              f"{r['sp_fp4v_over_fp4']:>6.3f}×")


if __name__ == '__main__':
    main()
