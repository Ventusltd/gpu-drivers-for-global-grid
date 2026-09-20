"""Batched layout scorer: the exact integer score of layout_search.score_window,
for all 40 windows, as tensor ops. Runs on the GPU with CuPy or on one CPU core
with NumPy (xp is chosen at import time by GPU_BENCH_BACKEND).

Every term of the CPU scorer is implemented: lattice snap, collision ring search
(displacement), octilinear error, dropped links, missing nodes, segment
crossings, and label-slot overlap. Nothing is left out.

Shape of the trick: the batch of parameter points is the parallel axis. The 40
windows and the 3 hop subsets are a 120-group Python loop (their node/link sets
differ), and inside each group the node-placement loop and the label loop are
still SEQUENTIAL over nodes, because both read an occupancy set that earlier
nodes wrote. That data dependence is the whole story of this benchmark.
"""
import itertools
import math
import os

import numpy as np

BACKEND = os.environ.get("GPU_BENCH_BACKEND", "numpy")
if BACKEND == "cupy":
    import cupy as xp
else:
    xp = np

N_PITCH, N_ROT, N_LAB, N_COL, N_OFX, N_OFY, N_HOP, N_SIM = 24, 16, 24, 16, 16, 16, 3, 12
SPACE = N_PITCH * N_ROT * N_LAB * N_COL * N_OFX * N_OFY * N_HOP * N_SIM
STRIDE = 1000003
W_LABEL, W_CROSS, W_OCT, W_DISP = 8, 5, 2, 1
W_DROP, W_MISS = 6, 4

NB = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
COLL_ORDERS = [[NB[(j * (1 + 2 * (o % 4)) + o) % 8] for j in range(8)] for o in range(16)]
LAB_SLOTS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
LAB_ORDERS = [list(p) for p in itertools.permutations(range(4))]

ROT_COS = [int(round(1024 * math.cos(math.radians(r * 3.0)))) for r in range(N_ROT)]
ROT_SIN = [int(round(1024 * math.sin(math.radians(r * 3.0)))) for r in range(N_ROT)]
PITCH_M = [500 + p * 500 for p in range(N_PITCH)]

I8 = np.int64
KMUL = 1 << 20          # cell key = gx * KMUL + gy
SENT = -(1 << 60)       # empty occupancy slot

_PITCH = xp.asarray(PITCH_M, I8)
_COS = xp.asarray(ROT_COS, I8)
_SIN = xp.asarray(ROT_SIN, I8)
_ORDX = xp.asarray([[a for (a, b) in o] for o in COLL_ORDERS], I8)   # (16,8)
_ORDY = xp.asarray([[b for (a, b) in o] for o in COLL_ORDERS], I8)
_LABORD = xp.asarray(LAB_ORDERS, I8)                                  # (24,4)
_SLOTX = xp.asarray([s[0] for s in LAB_SLOTS], I8)
_SLOTY = xp.asarray([s[1] for s in LAB_SLOTS], I8)


def pack_windows(windows):
    """Pre-flatten each (window, hop) group into static arrays."""
    groups = []
    for w in windows:
        for h in range(3):
            sub = w["subs"][h]
            pos = {n: k for k, n in enumerate(sub)}
            S = len(sub)
            xs = np.array([w["x"][n] for n in sub], I8)
            ys = np.array([w["y"][n] for n in sub], I8)
            lw = np.array([w["lw"][n] for n in sub], I8)
            act, base_drop = [], 0
            for (a, b, km100) in w["links"]:
                if a in pos and b in pos:
                    act.append((pos[a], pos[b], km100))
                else:
                    base_drop += 1
            la = np.array([t[0] for t in act], I8)
            lb = np.array([t[1] for t in act], I8)
            lkm = np.array([t[2] for t in act], I8)
            L = len(act)
            pi, pj = np.triu_indices(L, 1) if L > 1 else (np.array([], I8), np.array([], I8))
            # static layout of the occupancy table: S node cells, then lw[j] label
            # cells for each node j in order (0 for unnamed nodes).
            lab_at, c = [], S
            for j in range(S):
                lab_at.append(c)
                c += int(lw[j])
            groups.append({
                "w": w, "h": h, "S": S, "L": L, "M": c,
                "xs": xp.asarray(xs), "ys": xp.asarray(ys),
                "lw": [int(v) for v in lw], "lab_at": lab_at,
                "la": xp.asarray(la), "lb": xp.asarray(lb), "lkm": xp.asarray(lkm),
                "pi": xp.asarray(pi.astype(np.int64)), "pj": xp.asarray(pj.astype(np.int64)),
                "base_drop": base_drop, "miss": w["n"] - S,
            })
    return groups


def _occupied(occ, upto, key):
    """key: (B,) or (B,C). True where key is present in occ[:, :upto]."""
    if upto <= 0:
        return xp.zeros(key.shape, dtype=bool)
    o = occ[:, :upto]
    if key.ndim == 1:
        return (o == key[:, None]).any(axis=1)
    return (o[:, :, None] == key[:, None, :]).any(axis=1)


def score_batch(idxs, groups):
    """idxs: (B,) int64 parameter indices. Returns (B,) int64 total score."""
    i = idxs
    p = i % N_PITCH; i = i // N_PITCH
    r = i % N_ROT; i = i // N_ROT
    lo = i % N_LAB; i = i // N_LAB
    co = i % N_COL; i = i // N_COL
    ox = i % N_OFX; i = i // N_OFX
    oy = i % N_OFY; i = i // N_OFY
    hp = i % N_HOP; i = i // N_HOP
    sm = i % N_SIM

    B = idxs.shape[0]
    pitch = _PITCH[p]
    cs = _COS[r]; sn = _SIN[r]
    dx0 = (ox * pitch) // N_OFX
    dy0 = (oy * pitch) // N_OFY
    ordx = _ORDX[co]; ordy = _ORDY[co]          # (B,8)
    slots = _LABORD[lo]                          # (B,4) slot ids
    minkm = sm * 150
    total = xp.zeros(B, I8)

    for g in groups:
        m = (hp == g["h"])
        sel = xp.nonzero(m)[0]
        nb = int(sel.shape[0])
        if nb == 0:
            continue
        gp = pitch[sel]; gcs = cs[sel]; gsn = sn[sel]
        gdx = dx0[sel]; gdy = dy0[sel]
        gox = ordx[sel]; goy = ordy[sel]
        gsl = slots[sel]; gmin = minkm[sel]
        S = g["S"]; M = g["M"]
        occ = xp.full((nb, M), SENT, I8)
        cxs = xp.empty((nb, S), I8)
        cys = xp.empty((nb, S), I8)
        disp = xp.zeros(nb, I8)

        # --- lattice snap + collision ring search (sequential over nodes) ---
        for j in range(S):
            x = g["xs"][j]; y = g["ys"][j]
            gx = ((x * gcs + y * gsn) // 1024 + gdx) // gp
            gy = ((y * gcs - x * gsn) // 1024 + gdy) // gp
            if j > 0:
                key = gx * KMUL + gy
                coll = _occupied(occ, j, key)
                if bool(coll.any()):
                    placed = xp.zeros(nb, dtype=bool)
                    live = coll.copy()
                    for ring in (1, 2, 3):
                        for t in range(8):
                            need = live & ~placed
                            if not bool(need.any()):
                                break
                            cxk = gx + gox[:, t] * ring
                            cyk = gy + goy[:, t] * ring
                            ck = cxk * KMUL + cyk
                            free = ~_occupied(occ, j, ck) & need
                            gx = xp.where(free, cxk, gx)
                            gy = xp.where(free, cyk, gy)
                            disp = disp + xp.where(free, I8(ring), I8(0))
                            placed = placed | free
                        if not bool((live & ~placed).any()):
                            break
                    disp = disp + xp.where(live & ~placed, I8(4), I8(0))
            cxs[:, j] = gx
            cys[:, j] = gy
            occ[:, j] = gx * KMUL + gy

        # --- links: drops and octilinear error ---
        L = g["L"]
        ndrop = xp.full(nb, I8(g["base_drop"]), I8)
        oct_err = xp.zeros(nb, I8)
        if L:
            valid = g["lkm"][None, :] >= gmin[:, None]            # (nb,L)
            ndrop = ndrop + (~valid).sum(axis=1).astype(I8)
            ax = cxs[:, g["la"]]; ay = cys[:, g["la"]]
            bx = cxs[:, g["lb"]]; by = cys[:, g["lb"]]
            adx = xp.abs(ax - bx); ady = xp.abs(ay - by)
            e = xp.where((adx != 0) & (ady != 0) & (adx != ady), xp.abs(adx - ady), I8(0))
            oct_err = xp.where(valid, e, I8(0)).sum(axis=1).astype(I8)

        # --- crossings over every ordered pair of surviving segments ---
        cross = xp.zeros(nb, I8)
        if L > 1:
            pi = g["pi"]; pj = g["pj"]
            Ax = ax[:, pi]; Ay = ay[:, pi]; Bx = bx[:, pi]; By = by[:, pi]
            Px = ax[:, pj]; Py = ay[:, pj]; Qx = bx[:, pj]; Qy = by[:, pj]
            d1 = (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)
            d2 = (Bx - Ax) * (Qy - Ay) - (By - Ay) * (Qx - Ax)
            d3 = (Qx - Px) * (Ay - Py) - (Qy - Py) * (Ax - Px)
            d4 = (Qx - Px) * (By - Py) - (Qy - Py) * (Bx - Px)
            hit = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
            hit = hit & valid[:, pi] & valid[:, pj]
            cross = hit.sum(axis=1).astype(I8)

        # --- label slots (sequential over nodes; writes into the same occupancy) ---
        overl = xp.zeros(nb, I8)
        for j in range(S):
            wdt = g["lw"][j]
            if not wdt:
                continue
            at = g["lab_at"][j]
            gx = cxs[:, j]; gy = cys[:, j]
            done = xp.zeros(nb, dtype=bool)
            bxf = xp.zeros(nb, I8); byf = xp.zeros(nb, I8)
            for si in range(4):
                sid = gsl[:, si]
                sx = _SLOTX[sid]; sy = _SLOTY[sid]
                bxs = gx + sx * xp.where(sx >= 0, I8(1), I8(wdt))
                bys = gy + sy
                offs = xp.arange(wdt, dtype=I8)
                ck = (bxs[:, None] + offs[None, :]) * KMUL + bys[:, None]
                free = ~_occupied(occ, at, ck).any(axis=1) & ~done
                bxf = xp.where(free, bxs, bxf)
                byf = xp.where(free, bys, byf)
                done = done | free
                if bool(done.all()):
                    break
            offs = xp.arange(wdt, dtype=I8)
            keys = (bxf[:, None] + offs[None, :]) * KMUL + byf[:, None]
            occ[:, at:at + wdt] = xp.where(done[:, None], keys, I8(SENT))
            overl = overl + (~done).astype(I8)

        s = (W_LABEL * overl + W_CROSS * cross + W_OCT * oct_err + W_DISP * disp
             + W_DROP * ndrop + W_MISS * g["miss"])
        total[sel] = total[sel] + s
    return total
