"""Phase 1-C-b-4 microbench 3-way backward: FP32 ref / FP8 / FP4.

Symmetric to the forward microbench (microbench_4way.py) and the backward microbench
(microbench_backward.py); a 3-way version that adds the FP4 backward.

Runs entirely on Spark sm_121 (B200 completed the 4 quadrants in 1-C-a; the FP4 backward
was not measured on B200).
"""
import os
import subprocess
import statistics
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import sys
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("--- Loading FP32 ref ---")
from msda_fp32_ref import fp32_ref
print("--- Loading FP8 trainable ---")
from msda_fp8_backward import fp8_trainable
print("--- Loading FP4 trainable ---")
from msda_fp4_backward import fp4_trainable


def print_gpu_coexistence():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True).strip()
        print(f"--- GPU coexistence ---\n{out or '(none)'}\n")
    except Exception as e:
        print(f"--- GPU coexistence: {e} ---\n")


def make_inputs(bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    torch.manual_seed(0)
    spatial_size = sum(h * w for h, w in spatial_shapes_list)
    num_levels = len(spatial_shapes_list)
    value = torch.randn(bs, spatial_size, num_heads, channels,
                        device='cuda', dtype=torch.float32, requires_grad=True)
    sl = torch.rand(bs, num_query, num_heads, num_levels, num_points, 2,
                    device='cuda', dtype=torch.float32, requires_grad=True)
    aw = torch.randn(bs, num_query, num_heads, num_levels * num_points,
                     device='cuda', dtype=torch.float32).softmax(-1).view(
        bs, num_query, num_heads, num_levels, num_points).contiguous().clone().requires_grad_()
    ss = torch.tensor(spatial_shapes_list, dtype=torch.int64, device='cuda')
    if num_levels > 1:
        cums = [0]
        for h, w in spatial_shapes_list[:-1]:
            cums.append(cums[-1] + h * w)
        lsi = torch.tensor(cums, dtype=torch.int64, device='cuda')
    else:
        lsi = torch.zeros(num_levels, dtype=torch.int64, device='cuda')
    return value, sl, aw, ss, lsi


def make_step(fn, inputs):
    value, sl, aw, ss, lsi = inputs
    def step():
        if value.grad is not None: value.grad = None
        if sl.grad is not None:    sl.grad = None
        if aw.grad is not None:    aw.grad = None
        out = fn(value, ss, lsi, sl, aw)
        out.pow(2).sum().backward()
    return step


def bench(step, warmup=10, iters=50):
    for _ in range(warmup): step()
    torch.cuda.synchronize()
    se = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        se[i].record(); step(); ee[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(se, ee))
    return {'median': statistics.median(t),
            'p10': t[len(t)//10], 'p90': t[9*len(t)//10],
            'stdev': statistics.stdev(t)}


def run_workload(name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    print(f"\n========== {name} ==========")
    print(f"bs={bs} q={num_query} h={num_heads} c={channels} "
          f"L={len(spatial_shapes_list)} P={num_points}")

    in32 = make_inputs(bs, num_query, num_heads, channels, spatial_shapes_list, num_points)
    in8  = make_inputs(bs, num_query, num_heads, channels, spatial_shapes_list, num_points)
    in4  = make_inputs(bs, num_query, num_heads, channels, spatial_shapes_list, num_points)

    s32 = bench(make_step(fp32_ref, in32))
    s8  = bench(make_step(fp8_trainable, in8))
    s4  = bench(make_step(fp4_trainable, in4))

    sp_fp8 = s32['median'] / s8['median']
    sp_fp4 = s32['median'] / s4['median']
    sp_fp4_vs_fp8 = s8['median'] / s4['median']

    nb_fp32 = in32[0].numel() * 4
    print(f"value memory: FP32={nb_fp32/1024/1024:.2f} MiB, "
          f"FP8 storage={in32[0].numel()/1024/1024:.2f} MiB, "
          f"FP4 storage={in32[0].numel()/2/1024/1024:.2f} MiB "
          f"(+ FP32 grad_value staging = FP32 size, common to all)")
    print(f"FP32 ref: median={s32['median']:.4f} ms  stdev={s32['stdev']:.4f}")
    print(f"FP8:      median={s8['median']:.4f} ms  stdev={s8['stdev']:.4f}")
    print(f"FP4:      median={s4['median']:.4f} ms  stdev={s4['stdev']:.4f}")
    print(f"speedup vs FP32: FP8={sp_fp8:.3f}×  FP4={sp_fp4:.3f}×  FP4 vs FP8={sp_fp4_vs_fp8:.3f}×")

    return {'name': name, 's32': s32, 's8': s8, 's4': s4,
            'sp_fp8': sp_fp8, 'sp_fp4': sp_fp4, 'sp_fp4_vs_fp8': sp_fp4_vs_fp8}


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
            print(f"\n[SKIP {w[0]}] OOM"); results.append({'name': w[0], 'oom': str(e)})
        except Exception as e:
            print(f"\n[FAIL {w[0]}] {type(e).__name__}: {e}")
            results.append({'name': w[0], 'fail': str(e)})

    print("\n\n==================================================================")
    print("|| Summary: fwd+bwd combined median ms (Spark sm_121, 3-way)")
    print("==================================================================")
    print(f"{'workload':<26} {'FP32 ms':>10} {'FP8 ms':>10} {'FP4 ms':>10} {'spFP8':>8} {'spFP4':>8} {'4vs8':>7}")
    for r in results:
        if r.get('oom'):
            print(f"{r['name']:<26} OOM"); continue
        if r.get('fail'):
            print(f"{r['name']:<26} FAIL: {r['fail'][:40]}"); continue
        print(f"{r['name']:<26} "
              f"{r['s32']['median']:>10.4f} "
              f"{r['s8']['median']:>10.4f} "
              f"{r['s4']['median']:>10.4f} "
              f"{r['sp_fp8']:>7.3f}× "
              f"{r['sp_fp4']:>7.3f}× "
              f"{r['sp_fp4_vs_fp8']:>6.3f}×")


if __name__ == '__main__':
    main()
