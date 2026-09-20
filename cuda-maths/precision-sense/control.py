import sys, json, numpy as np, cupy as cp
sys.path.insert(0, r"(local path removed)")
from solar_core import *
from cpu_vec import build_tables, kernel
from cupyx import scatter_max
T = build_tables(cp); best = cp.zeros(N_CELLS, dtype=cp.int64); done = 0
while done < SPACE:
    n = min(1 << 22, SPACE - done)
    idx = cp.arange(done, done + n, dtype=cp.int64)
    _, pk, cell, _ = kernel(cp, T, idx)
    g = pk > 0
    if bool(g.any()): scatter_max(best, cell[g], pk[g])
    done += n
best = cp.asnumpy(best)
cells = np.arange(N_CELLS); dTn_ = (cells // 120) % 7; dc_ = (cells // 840) % 3
Voc0 = np.array([c[1] for c in CLASSES])[dc_]; beta0 = np.array([c[5] for c in CLASSES])[dc_]
Tmin0 = np.array(axis_values()["Tmin"])[dTn_]
bN = (best >> 44) & 0x3F; hv = best > 0
Nmax = np.floor(1500.0 / (Voc0 * (1.0 + beta0/100.0 * ((Tmin0 + T_ALLOW_K) - 25.0))))
pred = np.minimum(34.0, Nmax); pred = np.where(pred < 20, 0, pred)
ok = (bN == pred) & hv
d = (bN - pred)[hv]
print(json.dumps({"cells": int(hv.sum()), "rule_ok": int(ok.sum()),
  "share": float(ok.sum()/hv.sum()), "below_rule": int((d<0).sum()), "above_rule": int((d>0).sum()),
  "mean_delta": float(d.mean()), "min_delta": float(d.min()), "max_delta": float(d.max())}))
cp.get_default_memory_pool().free_all_blocks()
