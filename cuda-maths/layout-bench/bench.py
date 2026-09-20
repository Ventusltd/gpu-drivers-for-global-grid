"""Driver: verify the batched scorer against the CPU scorer, then measure it."""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUBE = r"(local path removed)"
sys.path.insert(0, TUBE)
import tubelib as T           # noqa: E402
import layout_search as L     # noqa: E402

import gpu_score as G         # noqa: E402
xp = G.xp
BACKEND = G.BACKEND

_stop = threading.Event()
_s = []


def sampler():
    while not _stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True,
                               text=True, timeout=10).stdout.strip().splitlines()[0]
            _s.append([float(x) for x in o.split(",")])
        except Exception:
            pass
        _stop.wait(2.0)


def cpu_total(windows, idx):
    p, r, lo, co, ox, oy, hp, sm = L.decode(idx)
    return sum(L.score_window(w, p, r, lo, co, ox, oy, hp, sm) for w in windows)


def main():
    g = T.jload(os.path.join(T.OUT, "graph.json"))
    cg = T.jload(os.path.join(T.OUT, "cages.json"))
    windows = L.build_windows(g, cg)
    groups = G.pack_windows(windows)
    out = {"backend": BACKEND, "windows": len(windows)}

    # ---- correctness: 2,000 points on the same coprime stride the search uses
    N = int(os.environ.get("N_VERIFY", "2000"))
    idxs = [(j * G.STRIDE) % G.SPACE for j in range(N)]
    t0 = time.time()
    cpu = [cpu_total(windows, i) for i in idxs]
    t_cpu_ref = time.time() - t0
    gpu = xp.asnumpy(G.score_batch(xp.asarray(np.array(idxs, np.int64)), groups)) \
        if BACKEND == "cupy" else np.asarray(G.score_batch(np.array(idxs, np.int64), groups))
    eq = (np.array(cpu, np.int64) == gpu)
    bad = [{"index": int(idxs[k]), "cpu": int(cpu[k]), "batched": int(gpu[k])}
           for k in np.nonzero(~eq)[0][:5]]
    out["verify"] = {"points": N, "equal": int(eq.sum()), "differ": int((~eq).sum()),
                     "counterexamples": bad,
                     "single_point_cpu_ref_s": round(t_cpu_ref, 2)}
    verified = int((~eq).sum()) == 0

    # ---- throughput
    if BACKEND == "cupy":
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
    sizes = [int(v) for v in os.environ.get("BATCHES", "4096,16384,65536").split(",")]
    secs = float(os.environ.get("BENCH_SECONDS", "20"))
    runs = []
    for bs in sizes:
        base = np.arange(bs, dtype=np.int64) * G.STRIDE % G.SPACE
        try:
            a = xp.asarray(base)
            G.score_batch(a, groups)           # warm up / compile
            if BACKEND == "cupy":
                xp.cuda.runtime.deviceSynchronize()
        except Exception as e:
            runs.append({"batch": bs, "error": type(e).__name__ + ": " + str(e)[:160]})
            continue
        t0 = time.time(); n = 0; k = 0
        while time.time() - t0 < secs:
            # include host->device transfer of the parameter indices every time
            arr = xp.asarray((base + k * bs * G.STRIDE) % G.SPACE)
            res = G.score_batch(arr, groups)
            if BACKEND == "cupy":
                res_h = xp.asnumpy(res)        # include device->host of the scores
            else:
                res_h = res
            n += bs; k += 1
        if BACKEND == "cupy":
            xp.cuda.runtime.deviceSynchronize()
        el = time.time() - t0
        runs.append({"batch": bs, "seconds": round(el, 2), "points": n,
                     "layout_evals": n * len(windows),
                     "points_per_s": round(n / el, 1),
                     "layout_evals_per_s": round(n * len(windows) / el, 1),
                     "verified": verified})
    if BACKEND == "cupy":
        _stop.set(); time.sleep(0.1)
        if _s:
            out["gpu_samples"] = {"n": len(_s),
                                  "watts_mean": round(sum(x[0] for x in _s) / len(_s), 1),
                                  "watts_max": round(max(x[0] for x in _s), 1),
                                  "util_pct_mean": round(sum(x[1] for x in _s) / len(_s), 1),
                                  "mem_used_MiB_max": round(max(x[2] for x in _s), 0)}
            best = max((r for r in runs if "layout_evals_per_s" in r),
                       key=lambda r: r["layout_evals_per_s"], default=None)
            if best:
                out["evals_per_watt_second"] = round(
                    best["layout_evals_per_s"] / out["gpu_samples"]["watts_mean"], 1)
    out["runs"] = runs
    json.dump(out, open(os.path.join(HERE, "bench-%s.json" % BACKEND), "w"), indent=1)
    print(json.dumps(out, indent=1)[:2600])
    print(json.dumps({"check": "gpu-bench", "cases": N, "passed": int(eq.sum()),
                      "failed": int((~eq).sum()), "counterexamples": bad,
                      "seconds": round(time.time() - t0, 3)}, separators=(",", ":")))
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
