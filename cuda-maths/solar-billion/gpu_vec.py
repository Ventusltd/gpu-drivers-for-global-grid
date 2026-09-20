# gpu_vec.py - the SAME arithmetic on the GPU with CuPy (the backend the GPU-bench
# agent got working on this card: cupy-cuda12x 14.2.0, sm_120). float64 throughout so
# it can be held to 1e-9 of the scalar reference. Everything is reduced ON the card:
# only 7 counts and 2520 packed int64s ever come back - about 20 kB a batch.
import numpy as np
import cupy as cp
from solar_core import *
from cpu_vec import build_tables, kernel

_T = None

def tables():
    global _T
    if _T is None:
        _T = build_tables(cp)
    return _T

def free_batch(target_frac=0.25, cap=1 << 23):
    """Batch sized to the FREE VRAM - the card is shared with other people's servers."""
    free, _tot = cp.cuda.Device(0).mem_info
    per_case = 8 * 22          # ~22 live float64/int64 temporaries a case
    n = int(free * target_frac / per_case)
    n = 1 << max(16, min(int(np.log2(max(n, 1 << 16))), int(np.log2(cap))))
    return n

def aggregate_batch(idx_gpu, counts, best):
    from cupyx import scatter_max
    T = tables()
    v, packed, cell, _ = kernel(cp, T, idx_gpu)
    counts += cp.bincount(v.astype(cp.int64), minlength=len(VERDICTS))
    good = packed > 0
    if bool(good.any()):
        scatter_max(best, cell[good], packed[good])

def run(total, stride=1, offset=0, batch=None, report=None):
    """Enumerate `total` cases on the GPU. Returns (counts, best) as NumPy."""
    batch = batch or free_batch()
    counts = cp.zeros(len(VERDICTS), dtype=cp.int64)
    best = cp.zeros(N_CELLS, dtype=cp.int64)
    done = 0
    while done < total:
        n = min(batch, total - done)
        idx = (cp.arange(done, done + n, dtype=cp.int64) * stride + offset) % SPACE
        aggregate_batch(idx, counts, best)
        done += n
        if report:
            report(done)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(counts), cp.asnumpy(best), batch

def detail(idx_np):
    """Full per-case outputs for the VERIFY step (small arrays only)."""
    T = tables()
    idx = cp.asarray(idx_np)
    v, packed, cell, outs = kernel(cp, T, idx)
    o = {k: cp.asnumpy(val) for k, val in outs.items()}
    return cp.asnumpy(v), cp.asnumpy(packed), cp.asnumpy(cell), o
