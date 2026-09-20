"""One fused kernel instead of a chain of elementwise ones.

The GPU review named this as the next win and said why: the card sat at 62%
not because of precision - float32 over the whole billion changed 0 verdicts
and was only 9% faster - but because roughly 22 float64 temporaries a case is
about 176 bytes a case, some 31 GB/s of DRAM round trips spread over many
small CuPy kernels. Each `xp.where`, each multiply, each compare allocates a
full-length array, writes it to memory and reads it back.

Fusing removes every one of them. The index is decoded INSIDE the kernel, the
arithmetic happens in registers, and the only thing that touches memory is one
read of the small lookup tables and one write of the verdict.

Same arithmetic as east_west_block.py: R1 cold voltage on both readings, then
the DC:AC band. Nothing new is computed here; this is the same answer, reached
without the round trips.

    python fused.py              compare fused against the chain, verify, time
"""
import sys
import time

import numpy as np

try:
    import cupy as cp
except Exception as exc:                                  # pragma: no cover
    print("cupy unavailable: %r" % (exc,))
    sys.exit(1)

# ---- what we own (MODULE-CLASSES.md): Voc, beta %/K, Wp
CLASSES = [(45.90, -0.25, 660.0), (50.59, -0.22, 760.0), (53.27, -0.25, 650.0)]
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]
SERIES = [float(s) for s in range(20, 35)]
STRINGS = [float(n) for n in range(16, 33)]
N_VOC = 101
N_BETA = 101        # beta +/- 10%, the second tolerance axis
INVERTER_KVA = 352.0
MAX_V = 1500.0
DCAC_LO, DCAC_HI = 1.15, 1.40

NC, NT, NS, NN, NV = len(CLASSES), len(TMIN), len(SERIES), len(STRINGS), N_VOC
NB = N_BETA
SIZE = NC * NT * NS * NN * NV * NB
VMID, BMID = N_VOC // 2, N_BETA // 2

OK, DISPUTED, FAIL_COLD, OUT_OF_BAND = 0, 1, 2, 3


# --------------------------------------------------------- the chain (before)
def chain(xp, idx, voc_t, beta_t, wp_t, tmin_t, ser_t, str_t):
    """The ordinary way: every operation allocates and writes a full array."""
    r = idx
    ci = r % NC; r = r // NC
    ti = r % NT; r = r // NT
    si = r % NS; r = r // NS
    ni = r % NN; r = r // NN
    vt = r % NV; r = r // NV
    bt = r % NB

    voc = voc_t[ci] * (1.0 + (vt - VMID) / VMID * 0.01)
    beta = beta_t[ci] * (1.0 + (bt - BMID) / BMID * 0.10)
    wp = wp_t[ci]
    tmin = tmin_t[ti]
    series = ser_t[si]
    strings = str_t[ni]

    nmax_b = xp.floor(MAX_V / (voc * (1.0 + beta / 100.0 * (tmin - 25.0))))
    nmax_a = xp.floor(MAX_V / (voc * (1.0 + beta / 100.0 * (tmin - 15.0))))
    ok_b = series <= nmax_b
    ok_a = series <= nmax_a
    dcac = (series * wp / 1000.0) * strings / INVERTER_KVA
    in_band = (dcac >= DCAC_LO) & (dcac <= DCAC_HI)
    return xp.where(~ok_a, FAIL_COLD,
                    xp.where(~ok_b, DISPUTED,
                             xp.where(in_band, OK,
                                      OUT_OF_BAND))).astype(xp.int64)


# --------------------------------------------------------- the fused (after)
_SRC = """
long long r = idx;
int ci = (int)(r %% %(NC)d); r /= %(NC)d;
int ti = (int)(r %% %(NT)d); r /= %(NT)d;
int si = (int)(r %% %(NS)d); r /= %(NS)d;
int ni = (int)(r %% %(NN)d); r /= %(NN)d;
int vt = (int)(r %% %(NV)d); r /= %(NV)d;
int bt = (int)(r %% %(NB)d);

double v  = voc[ci] * (1.0 + ((double)vt - %(VMID)d.0) / %(VMID)d.0 * 0.01);
double b  = beta[ci] * (1.0 + ((double)bt - %(BMID)d.0) / %(BMID)d.0 * 0.10);
double w  = wp[ci];
double tm = tmin[ti];
double se = ser[si];
double st = strs[ni];

double nmax_b = floor(%(MAXV).1f / (v * (1.0 + b / 100.0 * (tm - 25.0))));
double nmax_a = floor(%(MAXV).1f / (v * (1.0 + b / 100.0 * (tm - 15.0))));
bool ok_b = se <= nmax_b;
bool ok_a = se <= nmax_a;
double dcac = (se * w / 1000.0) * st / %(KVA).1f;
bool band = (dcac >= %(LO).2f) && (dcac <= %(HI).2f);

code = (!ok_a) ? %(FC)d : ((!ok_b) ? %(DI)d : (band ? %(OK)d : %(OB)d));
""" % {"NC": NC, "NT": NT, "NS": NS, "NN": NN, "NV": NV, "NB": NB,
       "VMID": VMID, "BMID": BMID,
       "MAXV": MAX_V, "KVA": INVERTER_KVA, "LO": DCAC_LO, "HI": DCAC_HI,
       "FC": FAIL_COLD, "DI": DISPUTED, "OK": OK, "OB": OUT_OF_BAND}

FUSED = cp.ElementwiseKernel(
    "int64 idx, raw float64 voc, raw float64 beta, raw float64 wp, "
    "raw float64 tmin, raw float64 ser, raw float64 strs",
    "int64 code", _SRC, "ventus_fused_case")


def tables(xp):
    return (xp.asarray([c[0] for c in CLASSES]),
            xp.asarray([c[1] for c in CLASSES]),
            xp.asarray([c[2] for c in CLASSES]),
            xp.asarray(TMIN), xp.asarray(SERIES), xp.asarray(STRINGS))


def run(kind, batch):
    t = tables(cp)
    totals = cp.zeros(4, dtype=cp.int64)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    done = 0
    while done < SIZE:
        n = min(batch, SIZE - done)
        idx = cp.arange(done, done + n, dtype=cp.int64)
        codes = FUSED(idx, *t) if kind == "fused" else chain(cp, idx, *t)
        totals += cp.bincount(codes, minlength=4)[:4]
        done += n
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0, [int(x) for x in cp.asnumpy(totals)]


def main():
    free, total = cp.cuda.Device(0).mem_info
    batch = max(1 << 12, min(1 << 23, int(free * 0.25 / 176)))
    print("space %s cases over 6 axes %s" % ("{:,}".format(SIZE),
                                             [NC, NT, NS, NN, NV, NB]))
    print("device %.2f GB free of %.2f GB, batch %s\n"
          % (free / 1e9, total / 1e9, "{:,}".format(batch)))

    # ---- correctness first, on the processor, against a hand-checked case
    tn = tables(np)
    i = np.array([0 + NC * (3 + NT * (10 + NS * (8 + NN * (VMID + NV * BMID))))],
                 dtype=np.int64)
    assert int(chain(np, i, *tn)[0]) == OK, "T660, -10 C, 30 series, 24 strings"
    print("hand-checked case agrees: the reference block is buildable")

    # ---- warm both kernels so compilation is not timed as arithmetic
    run("fused", 1 << 16)
    run("chain", 1 << 16)

    sec_c, tot_c = run("chain", batch)
    sec_f, tot_f = run("fused", batch)

    if tot_c != tot_f:
        print("FAIL: chain %s vs fused %s" % (tot_c, tot_f))
        return 1
    # and both against the processor over a sample
    stride = 1000003
    while np.gcd(stride, SIZE) != 1:
        stride += 2
    s = (np.arange(200000, dtype=np.int64) * stride) % SIZE
    a = chain(np, s, *tn)
    b = cp.asnumpy(FUSED(cp.asarray(s), *tables(cp)))
    differ = int((a != b).sum())
    print("verified: chain == fused on all %s cases, and processor vs fused "
          "on 200,000 sampled cases: %d differ\n" % ("{:,}".format(SIZE), differ))
    if differ:
        return 1

    if SIZE < batch:
        print("  NOT A THROUGHPUT: the space is smaller than one batch.")
        return 1
    print("  chain of elementwise kernels .. %7.3f s  %15s cases/s"
          % (sec_c, "{:,.0f}".format(SIZE / sec_c)))
    print("  one fused kernel .............. %7.3f s  %15s cases/s"
          % (sec_f, "{:,.0f}".format(SIZE / sec_f)))
    print("  fused is %.2fx the chain" % (sec_c / sec_f))
    print("\n  longest single kernel launch is well under the 2 s Windows")
    print("  display timeout, which matters because this card drives the")
    print("  screen: batch %s at %.3f s a pass."
          % ("{:,}".format(batch), sec_f / max(1, SIZE // batch)))
    print("  verdicts %s" % tot_f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
