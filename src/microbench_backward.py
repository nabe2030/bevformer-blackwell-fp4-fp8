"""Phase 1-C-4 backward microbench: FP32 ref vs FP8 backward.

Measures the forward + backward wall-clock on the 3 workloads tiny/base/large.
FP32 ref is msda_fp32_ref (port of mmcv's own kernel, made in Phase 1-C-3 debug step (c)),
FP8 is msda_fp8_backward (implemented in Phase 1-C-1; PASS at 2.0-2.6% after the scale-restoration fix).

The shapes are identical to the forward microbench (microbench_4way.py).
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

print("--- Loading FP32 ref (mmcv port fwd+bwd) ---")
from msda_fp32_ref import fp32_ref
print("--- Loading FP8 trainable (fwd+bwd) ---")
from msda_fp8_backward import fp8_trainable


def print_gpu_coexistence():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True).strip()
        print(f"--- GPU coexistence ---\n{out or '(none)'}\n")
    except Exception as e:
        print(f"--- GPU coexistence query failed: {e} ---\n")


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
        # Reset grads (since autograd accumulates)
        if value.grad is not None: value.grad = None
        if sl.grad is not None:    sl.grad = None
        if aw.grad is not None:    aw.grad = None
        out = fn(value, ss, lsi, sl, aw)
        loss = out.pow(2).sum()
        loss.backward()
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
    return {
        'median': statistics.median(t),
        'p10': t[len(t)//10], 'p90': t[9*len(t)//10],
        'stdev': statistics.stdev(t),
    }


def run_workload(name, bs, num_query, num_heads, channels, spatial_shapes_list, num_points):
    print(f"\n========== {name} ==========")
    print(f"bs={bs} q={num_query} h={num_heads} c={channels} "
          f"L={len(spatial_shapes_list)} P={num_points}")

    inputs_fp32 = make_inputs(bs, num_query, num_heads, channels,
                               spatial_shapes_list, num_points)
    inputs_fp8 = make_inputs(bs, num_query, num_heads, channels,
                              spatial_shapes_list, num_points)

    step_fp32 = make_step(fp32_ref, inputs_fp32)
    step_fp8  = make_step(fp8_trainable, inputs_fp8)

    s32 = bench(step_fp32)
    s8  = bench(step_fp8)

    sp_fp8 = s32['median'] / s8['median']

    nbytes_fp32 = inputs_fp32[0].numel() * 4
    nbytes_fp8  = inputs_fp32[0].numel()
    print(f"value memory: FP32={nbytes_fp32/1024/1024:.2f} MiB, "
          f"FP8 storage={nbytes_fp8/1024/1024:.2f} MiB "
          f"(+ FP32 grad_value staging = same size as FP32)")
    print(f"FP32 (mmcv ref):  median={s32['median']:.4f} ms  p10={s32['p10']:.4f}  "
          f"p90={s32['p90']:.4f}  stdev={s32['stdev']:.4f}")
    print(f"FP8  (trainable): median={s8['median']:.4f} ms  p10={s8['p10']:.4f}  "
          f"p90={s8['p90']:.4f}  stdev={s8['stdev']:.4f}")
    print(f"speedup FP8 vs FP32 (fwd+bwd combined): {sp_fp8:.3f}×")

    return {
        'name': name, 's32': s32, 's8': s8, 'sp_fp8': sp_fp8,
        'mib_fp32': nbytes_fp32/1024/1024, 'mib_fp8': nbytes_fp8/1024/1024,
    }


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name()}, "
          f"capability={torch.cuda.get_device_capability()}")
    print_gpu_coexistence()

    # same shape settings as the forward microbench_4way.py
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
        except Exception as e:
            print(f"\n[FAIL {w[0]}] {type(e).__name__}: {e}")
            results.append({'name': w[0], 'fail': str(e)})

    print("\n\n==================================================================")
    print("|| Summary: forward+backward combined median ms (Spark sm_121)")
    print("==================================================================")
    print(f"{'workload':<26} {'FP32 ms':>10} {'FP8 ms':>10} {'speedup':>9}")
    for r in results:
        if r.get('oom'):
            print(f"{r['name']:<26} OOM"); continue
        if r.get('fail'):
            print(f"{r['name']:<26} FAIL: {r['fail'][:40]}"); continue
        print(f"{r['name']:<26} "
              f"{r['s32']['median']:>10.4f} "
              f"{r['s8']['median']:>10.4f} "
              f"{r['sp_fp8']:>8.3f}×")


if __name__ == '__main__':
    main()
