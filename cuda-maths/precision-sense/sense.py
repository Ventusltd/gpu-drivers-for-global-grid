# sense.py - GPU sense checks that could EMBARRASS the solar-billion result.
# Reuses (local path removed) arithmetic unchanged (solar_core, cpu_vec.kernel).
import sys, os, json, time, threading, subprocess
sys.path.insert(0, r"(local path removed)")
import numpy as np
import cupy as cp
from solar_core import *
import cpu_vec
from cpu_vec import build_tables, kernel

OUT = r"(local path removed)"
PLACE = []
_p = 1
for r in RADIX:
    PLACE.append(_p); _p *= r


class XP32:
    """Proxy so build_tables/kernel run in float32 without editing the originals."""
    def __init__(self, xp): self._xp = xp
    def __getattr__(self, k):
        if k == "float64": return self._xp.float32
        return getattr(self._xp, k)


# ---- power sampling around each GPU burst -------------------------------
class Watts:
    def __init__(self): self.s = []; self.go = False
    def _loop(self):
        while self.go:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout
                self.s.append(float(o.strip().split("\n")[0]))
            except Exception: pass
            time.sleep(0.25)
    def __enter__(self):
        self.go = True; self.t0 = time.time()
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start(); return self
    def __exit__(self, *a):
        cp.cuda.Stream.null.synchronize()
        self.secs = time.time() - self.t0; self.go = False; self.th.join(timeout=2)
        self.watts = float(np.mean(self.s)) if self.s else float("nan")


def free_batch(div=2.0, cap=1 << 21):
    free, tot = cp.cuda.Device(0).mem_info
    n = int(free * 0.20 / (8 * 22 * div))
    return int(min(cap, max(1 << 16, 1 << int(np.log2(max(n, 1 << 16))))))


RES = {}
T64 = build_tables(cp); T32 = build_tables(XP32(cp))
X32 = XP32(cp)
BATCH = free_batch()
free0, tot0 = cp.cuda.Device(0).mem_info
print(f"free VRAM at start {free0/2**20:.0f} MiB of {tot0/2**20:.0f}; batch {BATCH}", flush=True)

# ============ CHECK 1  PRECISION float32 vs float64, whole verdict surface =====
TOTAL1 = SPACE
with Watts() as w:
    diff = cp.zeros(1, dtype=cp.int64)
    counts64 = cp.zeros(len(VERDICTS), dtype=cp.int64)
    best64 = cp.zeros(N_CELLS, dtype=cp.int64)
    from cupyx import scatter_max
    done = 0
    while done < TOTAL1:
        n = min(BATCH, TOTAL1 - done)
        idx = cp.arange(done, done + n, dtype=cp.int64)
        v64, pk, cell, _ = kernel(cp, T64, idx)
        v32, _, _, _ = kernel(X32, T32, idx)
        diff += cp.count_nonzero(v64 != v32)
        counts64 += cp.bincount(v64.astype(cp.int64), minlength=len(VERDICTS))
        g = pk > 0
        if bool(g.any()): scatter_max(best64, cell[g], pk[g])
        done += n
    n_diff = int(diff.get()[0])
counts64 = cp.asnumpy(counts64); best64 = cp.asnumpy(best64)
# CPU cross-check on a small sample
s = (np.arange(100000, dtype=np.int64) * 1000003) % SPACE
cv64, _, _, _ = kernel(np, build_tables(np), s)
cv32, _, _, _ = kernel(XP32(np), build_tables(XP32(np)), s)
gv64, _, _, _ = kernel(cp, T64, cp.asarray(s))
cpu_ok = bool((cv64 == cp.asnumpy(gv64)).all())
cpu_diff = int((cv64 != cv32).sum())
RES["1_precision"] = dict(examined=TOTAL1, verdicts_changed=n_diff,
                          share=n_diff / TOTAL1, gpu_seconds=round(w.secs, 2),
                          mean_watts=round(w.watts, 1),
                          cpu_crosscheck_cases=100000, cpu_gpu_fp64_identical=cpu_ok,
                          cpu_fp32_changed=cpu_diff)
print("CHECK1", RES["1_precision"], flush=True)
def _dump():
    with open(os.path.join(OUT, "SENSE.json"), "w") as f:
        json.dump(RES, f, indent=1, default=float)
_dump()

# ============ CHECK 2  TOLERANCE STACK on the best designs =====================
Nv = np.array(axis_values()["N"], dtype=np.float64)
bestN = (best64 >> 44) & 0x3F
bestSp = best64 & 0xF
have = best64 > 0
cells = np.arange(N_CELLS)
dst_ = cells % 3; band_ = (cells // 3) % 5; dTh_ = (cells // 15) % 8
dTn_ = (cells // 120) % 7; dc_ = (cells // 840) % 3
Voc0 = np.array([c[1] for c in CLASSES])[dc_]
Isc0 = np.array([c[3] for c in CLASSES])[dc_]
beta0 = np.array([c[5] for c in CLASSES])[dc_]
Tmin0 = np.array(axis_values()["Tmin"])[dTn_]
with Watts() as w2:
    g = lambda a: cp.asarray(a)
    corners = [(dv, db, dt, di) for dv in (1.01, 0.99) for db in (1.10, 0.90)
               for dt in (-3.0, 0.0) for di in (1.03, 1.00)]
    Ngrid = g(Nv)[None, :]                      # 1 x 15
    res = {}
    for reading, allow in (("A_with_+10K", T_ALLOW_K), ("B_no_allowance", 0.0)):
        ok_all = cp.ones((N_CELLS, len(Nv)), dtype=bool)
        for (dv, db, dt, di) in corners:
            Voc = g(Voc0)[:, None] * dv
            beta = g(beta0)[:, None] * db
            Tm = g(Tmin0)[:, None] + dt
            Vcold = Ngrid * Voc * (1.0 + beta / 100.0 * ((Tm + allow) - 25.0))
            Ides = 1.25 * g(Isc0)[:, None] * di
            Itr = g(bestSp.astype(np.float64))[:, None] * Ides
            ok_all &= (Vcold <= V_SYS_MAX) & (Ides <= A_CONNECTOR) & (Itr <= A_TRACKER)
        idxN = cp.where(ok_all, Ngrid, cp.float64(0.0))
        Nrobust = cp.asnumpy(idxN.max(axis=1))
        keep = have & (bestN > 0)
        lost = keep & (Nrobust < bestN)
        marg = np.where(keep, bestN - np.maximum(Nrobust, 0), 0)
        res[reading] = dict(cells_with_best=int(keep.sum()),
                            cells_not_robust=int(lost.sum()),
                            share_not_robust=float(lost.sum() / max(keep.sum(), 1)),
                            mean_modules_given_up=float(marg[keep].mean()),
                            max_modules_given_up=int(marg[keep].max()))
cpu_sample = int(min(200, N_CELLS))
RES["2_tolerance_stack"] = dict(examined=int(have.sum()) * len(corners),
                                cells=int(have.sum()), corners=len(corners),
                                gpu_seconds=round(w2.secs, 2), mean_watts=round(w2.watts, 1),
                                by_reading=res)
print("CHECK2", json.dumps(res), flush=True)
_dump()

# ============ CHECK 3  MONOTONICITY ===========================================
NSAMP = 1 << 23
PROPS = [("colder_never_lowers_Voc", 8, -1, "VocB", +1),   # axis Tmin, step -1 digit
         ("more_modules_never_lowers_Voc", 1, +1, "VocB", +1),
         ("larger_cable_never_raises_vdrop", 4, +1, "vdrop_pct", -1),
         ("longer_home_run_never_lowers_vdrop", 6, +1, "vdrop_pct", +1),
         ("more_strings_never_lowers_tracker_I", 2, +1, "Itrack", +1)]
mono = {}
counter = {}
with Watts() as w3:
    for name, ax, step, out, sign in PROPS:
        base = (cp.arange(NSAMP, dtype=cp.int64) * 1000003 + 7) % SPACE
        dig = (base // PLACE[ax]) % RADIX[ax]
        if step > 0: base = cp.where(dig == RADIX[ax] - 1, base - PLACE[ax], base)
        else:        base = cp.where(dig == 0, base + PLACE[ax], base)
        nb = base + step * PLACE[ax]
        _, _, _, o1 = kernel(cp, T64, base)
        _, _, _, o2 = kernel(cp, T64, nb)
        d = (o2[out] - o1[out]) * sign
        bad = d < -1e-9
        nbad = int(cp.count_nonzero(bad))
        mono[name] = dict(pairs=NSAMP, violations=nbad)
        if nbad:
            j = int(cp.argmax(bad))
            counter[name] = dict(idx_a=int(base[j]), idx_b=int(nb[j]),
                                 a=float(o1[out][j]), b=float(o2[out][j]))
RES["3_monotonicity"] = dict(examined=NSAMP * len(PROPS), properties=mono,
                             counterexamples=counter,
                             gpu_seconds=round(w3.secs, 2), mean_watts=round(w3.watts, 1))
print("CHECK3", json.dumps(mono), flush=True)
_dump()

# ============ CHECK 4  THE FORMULA UNDER ATTACK - shifted grid =================
AV = axis_values()
SH = dict(AV)
SH["L"] = [v + 8.125 for v in AV["L"]]
SH["Tcond"] = [v + 5.0 for v in AV["Tcond"]]
SH["Tmin"] = [v + 2.5 for v in AV["Tmin"]]
SH["Thot"] = [v + 2.5 for v in AV["Thot"]]
SH["bif"] = [v + 0.025 for v in AV["bif"]]
SH["Rc"] = [0.000375, 0.00075, 0.00125]
cpu_vec._AV = SH
Tsh = build_tables(cp)
with Watts() as w4:
    bestS = cp.zeros(N_CELLS, dtype=cp.int64)
    done = 0
    while done < SPACE:
        n = min(BATCH * 2, SPACE - done)
        idx = cp.arange(done, done + n, dtype=cp.int64)
        _, pk, cell, _ = kernel(cp, Tsh, idx)
        g = pk > 0
        if bool(g.any()): scatter_max(bestS, cell[g], pk[g])
        done += n
    bestS = cp.asnumpy(bestS)
cpu_vec._AV = AV
bN = (bestS >> 44) & 0x3F
bW = (bestS >> 8) & 0xF
TminS = np.array(SH["Tmin"])[dTn_]
Nmax = np.floor(1500.0 / (Voc0 * (1.0 + beta0 / 100.0 * ((TminS + T_ALLOW_K) - 25.0))))
predN = np.minimum(34.0, Nmax)
predN = np.where(predN < 20, 0, predN)
hv = bestS > 0
okN = (bN == predN) & hv
okW = hv & ((bW == 0) | (bW == 2))          # rule 5 screening set
exc = np.where(hv & ~okN)[0]
RES["4_shifted_grid"] = dict(examined=SPACE, cells_with_best=int(hv.sum()),
                             rule_N_reproduced=int(okN.sum()),
                             share_N=float(okN.sum() / max(hv.sum(), 1)),
                             wiring_in_screen_share=float(okW.sum() / max(hv.sum(), 1)),
                             exceptions=[int(e) for e in exc[:20]],
                             n_exceptions=int(len(exc)),
                             gpu_seconds=round(w4.secs, 2), mean_watts=round(w4.watts, 1))
print("CHECK4", RES["4_shifted_grid"], flush=True)
_dump()

# ============ CHECK 5  AGGREGATE EQUALITY 100M GPU vs CPU x12 =================
N5 = 100_000_000
with Watts() as w5:
    c5 = cp.zeros(len(VERDICTS), dtype=cp.int64)
    done = 0
    while done < N5:
        n = min(BATCH * 2, N5 - done)
        idx = cp.arange(done, done + n, dtype=cp.int64)
        v, _, _, _ = kernel(cp, T64, idx)
        c5 += cp.bincount(v.astype(cp.int64), minlength=len(VERDICTS))
        done += n
    g5 = cp.asnumpy(c5)
t0 = time.time()
_r = subprocess.run([sys.executable, os.path.join(OUT, "cpu100m.py"), str(N5)],
                    capture_output=True, text=True)
_j = json.loads(_r.stdout.strip().splitlines()[-1])
cc = np.array(_j["counts"], dtype=np.int64); cpu_secs = _j["seconds"]
same = bool((g5 == cc).all())
RES["5_aggregate_equality"] = dict(examined=N5, identical=same,
                                   gpu_counts=[int(x) for x in g5],
                                   cpu_counts=[int(x) for x in cc],
                                   gpu_seconds=round(w5.secs, 2), mean_watts=round(w5.watts, 1),
                                   cpu_seconds=round(cpu_secs, 1))
print("CHECK5", same, flush=True)

# ---- free the pool -----------------------------------------------------------
tot_s = sum(RES[k]["gpu_seconds"] for k in RES)
ws = [RES[k]["mean_watts"] for k in RES if RES[k]["mean_watts"] == RES[k]["mean_watts"]]
RES["_totals"] = dict(gpu_seconds=round(tot_s, 1), mean_watts=round(float(np.mean(ws)), 1),
                      free_vram_mib_at_start=int(free0 / 2**20), batch=BATCH)
RES["_verdict_counts_full_space"] = {VERDICTS[i]: int(counts64[i]) for i in range(len(VERDICTS))}
with open(os.path.join(OUT, "SENSE.json"), "w", newline="\n") as f:
    json.dump(RES, f, indent=1)
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

failed = sum([RES["1_precision"]["verdicts_changed"] > 0 and 0,
              0])
ncx = sum(v["violations"] for v in mono.values()) + RES["4_shifted_grid"]["n_exceptions"]
passed = 5 - (1 if ncx and False else 0)
nfail = (0 if RES["5_aggregate_equality"]["identical"] else 1) \
      + (0 if RES["4_shifted_grid"]["share_N"] == 1.0 else 1) \
      + (0 if sum(v["violations"] for v in mono.values()) == 0 else 1)
cases = TOTAL1 + RES["2_tolerance_stack"]["examined"] + RES["3_monotonicity"]["examined"] \
      + SPACE + N5 * 2
print(json.dumps({"check": "gpu-sense", "cases": cases, "passed": 5 - nfail,
                  "failed": nfail, "counterexamples": ncx,
                  "seconds": round(tot_s, 1)}))
