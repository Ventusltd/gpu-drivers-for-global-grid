# cpu_vec.py - the same arithmetic as cpu_ref.py, NumPy-vectorised over mixed-radix
# indices, and a multi-process driver capped at 12 workers below normal priority.
import numpy as np
from solar_core import *

_AV = axis_values()

def _tab(xp, name):
    return xp.asarray(_AV[name], dtype=xp.float64)

def build_tables(xp):
    """Every axis as a lookup table, so the kernel is pure gather + arithmetic."""
    C = np.array([[c[1], c[2], c[3], c[4], c[5], c[6], c[8], c[9], c[10]] for c in CLASSES],
                 dtype=np.float64)   # Voc Vmp Isc Imp beta gamma fuse pmax width
    T = dict(C=xp.asarray(C))
    for k in ("N", "Sp", "lead", "site", "L", "Tcond", "Tmin", "Thot", "bif", "Rc"):
        T[k] = _tab(xp, k)
    T["R20"] = xp.asarray([R20[4], R20[6], R20[10]], dtype=xp.float64)
    T["Lband"] = xp.asarray(L_BAND_EDGES, dtype=xp.float64)
    return T

def kernel(xp, T, idx):
    """idx: int64 array of linear case indices -> (verdict, packed, cell, outputs)."""
    i = idx
    dc = (i % RADIX[0]).astype(xp.int64);  i = i // RADIX[0]
    dN = (i % RADIX[1]).astype(xp.int64);  i = i // RADIX[1]
    dSp = (i % RADIX[2]).astype(xp.int64); i = i // RADIX[2]
    dw = (i % RADIX[3]).astype(xp.int64);  i = i // RADIX[3]
    dld = (i % RADIX[4]).astype(xp.int64); i = i // RADIX[4]
    dst = (i % RADIX[5]).astype(xp.int64); i = i // RADIX[5]
    dL = (i % RADIX[6]).astype(xp.int64);  i = i // RADIX[6]
    dTc = (i % RADIX[7]).astype(xp.int64); i = i // RADIX[7]
    dTn = (i % RADIX[8]).astype(xp.int64); i = i // RADIX[8]
    dTh = (i % RADIX[9]).astype(xp.int64); i = i // RADIX[9]
    dbf = (i % RADIX[10]).astype(xp.int64); i = i // RADIX[10]
    dRc = (i % RADIX[11]).astype(xp.int64)

    C = T["C"]
    Voc = C[dc, 0]; Vmp = C[dc, 1]; Isc = C[dc, 2]; Imp = C[dc, 3]
    beta = C[dc, 4]; gamma = C[dc, 5]; fuse = C[dc, 6]; wm = C[dc, 8]

    N = T["N"][dN]; Sp = T["Sp"][dSp]
    L = T["L"][dL]; Tc = T["Tcond"][dTc]; Tmin = T["Tmin"][dTn]
    Thot = T["Thot"][dTh]; bif = T["bif"][dbf]; Rc = T["Rc"][dRc]

    VocA = N * Voc * (1.0 + beta / 100.0 * ((Tmin + T_ALLOW_K) - 25.0))
    VocB = N * Voc * (1.0 + beta / 100.0 * (Tmin - 25.0))
    VmpHot = N * Vmp * (1.0 + gamma / 100.0 * (Thot - 25.0))
    VmpStc = N * Vmp

    Idesign = 1.25 * bif * Isc
    Itrack = Sp * Idesign
    Iop = bif * Imp
    fault = (Sp - 1.0) * 1.25 * Isc
    fuse_needed = (Sp >= 3.0) | (fault > fuse)

    p = wm + GAP_M
    seq = (dw == 0); lf = (dw == 1)
    lead_per_mod = xp.where(seq, LEAD_POS_M + LEAD_NEG_M,
                            xp.where(lf, 2.0 * p + 2.0 * SLACK_M,
                                     (2.0 * p - BOX_S_M) + 2.0 * SLACK_M))
    site_return = xp.where(seq, (N - 2.0) * p + BOX_S_M, 0.0)
    lead_m = N * lead_per_mod
    home_m = 2.0 * L
    site_m = site_return + home_m
    pairs = N + 1.0

    k = 1.0 + ALPHA_CU * (Tc - 20.0)
    r_lead = T["R20"][dld] / 1000.0 * k
    r_site = T["R20"][dst] / 1000.0 * k
    R = r_lead * lead_m + r_site * site_m + pairs * Rc

    vdrop = Iop * R
    vdrop_p = vdrop / VmpHot * 100.0
    loss_p = vdrop / VmpStc * 100.0

    v = xp.zeros(idx.shape, dtype=xp.int8)
    v = xp.where(VocB > V_SYS_MAX, xp.int8(2), v)
    v = xp.where(vdrop_p > VDROP_CAP, xp.int8(6), v)
    v = xp.where(Itrack > A_TRACKER, xp.int8(5), v)
    v = xp.where(Idesign > A_CONNECTOR, xp.int8(4), v)
    v = xp.where(VmpHot < MPPT_MIN_V, xp.int8(3), v)
    v = xp.where(VocA > V_SYS_MAX, xp.int8(1), v)

    # design-surface cell: (class, Tmin, Thot, home-run band, site size)
    band = ((L >= T["Lband"][0]).astype(xp.int64) + (L >= T["Lband"][1]).astype(xp.int64)
            + (L >= T["Lband"][2]).astype(xp.int64) + (L >= T["Lband"][3]).astype(xp.int64))
    cell = (((dc * 7 + dTn) * 8 + dTh) * 5 + band) * 3 + dst

    metres = lead_m + site_m
    packed = ((dN + 20) << 44) | ((0x0FFFFFFF - (metres * 10.0 + 0.5).astype(xp.int64)) << 16) \
             | (dw << 8) | (dld << 4) | dSp.astype(xp.int64)
    packed = xp.where(v == 0, packed, xp.int64(0))

    outs = dict(VocA=VocA, VocB=VocB, VmpHot=VmpHot, Idesign=Idesign, Itrack=Itrack,
                R=R, vdrop_pct=vdrop_p, loss_pct=loss_p, lead_m=lead_m, site_m=site_m,
                fuse=fuse_needed)
    return v, packed, cell, outs

def aggregate_np(idx):
    """counts per verdict + best packed design per cell, on the CPU."""
    T = build_tables(np)
    v, packed, cell, _ = kernel(np, T, idx)
    counts = np.bincount(v.astype(np.int64), minlength=len(VERDICTS))
    best = np.zeros(N_CELLS, dtype=np.int64)
    good = packed > 0
    if good.any():
        np.maximum.at(best, cell[good], packed[good])
    return counts, best

def _worker(args):
    start, n, stride, offset = args
    try:   # below-normal priority: the owner's ceiling is 50% of the machine
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    idx = ((np.arange(start, start + n, dtype=np.int64) * stride + offset) % SPACE)
    return aggregate_np(idx)

def run_multiproc(total, workers=12, stride=1, offset=0, chunk=1 << 20):
    """Enumerate `total` cases over at most `workers` processes."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    done = 0
    while done < total:
        n = min(chunk, total - done)
        jobs.append((done, n, stride, offset))
        done += n
    counts = np.zeros(len(VERDICTS), dtype=np.int64)
    best = np.zeros(N_CELLS, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c, b in ex.map(_worker, jobs, chunksize=1):
            counts += c
            np.maximum(best, b, out=best)
    return counts, best
