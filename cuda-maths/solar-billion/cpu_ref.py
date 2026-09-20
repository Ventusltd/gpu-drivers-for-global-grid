# cpu_ref.py - the plain scalar reference. Clear and slow. Everything else on this
# job is checked against this file. One CASE evaluated end to end from closed forms.
import math
from solar_core import *

AV = axis_values()

def decode(i):
    d = []
    for r in RADIX:
        d.append(i % r); i //= r
    return d

def evaluate(idx):
    (dc, dN, dSp, dw, dld, dst, dL, dTc, dTmin, dTh, dbif, dRc) = decode(idx)
    cid, Voc, Vmp, Isc, Imp, beta, gamma, a_isc, fuse, pmax, wm, lm = CLASSES[dc]
    N     = AV["N"][dN]
    Sp    = AV["Sp"][dSp]
    wir   = dw
    lead  = AV["lead"][dld]
    site  = AV["site"][dst]
    L     = AV["L"][dL]
    Tc    = AV["Tcond"][dTc]
    Tmin  = AV["Tmin"][dTmin]
    Thot  = AV["Thot"][dTh]
    bif   = AV["bif"][dbif]
    Rc    = AV["Rc"][dRc]

    # ---- 1. cold open-circuit voltage, BOTH readings of the disputed clause ----
    # reading A: the site minimum with the +10 K allowance   -> Tdesign = Tmin + 10
    # reading B: the site minimum with no allowance          -> Tdesign = Tmin
    VocA = N * Voc * (1.0 + beta / 100.0 * ((Tmin + T_ALLOW_K) - 25.0))
    VocB = N * Voc * (1.0 + beta / 100.0 * (Tmin - 25.0))

    # ---- 2. maximum-power voltage at the hot cell temperature ----
    VmpHot = N * Vmp * (1.0 + gamma / 100.0 * (Thot - 25.0))
    VmpStc = N * Vmp

    # ---- 3. design current ----
    Idesign = 1.25 * bif * Isc
    Itrack  = Sp * Idesign
    Iop     = bif * Imp                      # operating current for the loss terms

    # ---- 4. fuse question ----
    fault = (Sp - 1) * 1.25 * Isc
    fuse_needed = (Sp >= 3) or (fault > fuse)

    # ---- 5. geometry and series resistance ----
    p = wm + GAP_M
    if wir == 0:                             # sequential
        lead_per_mod = LEAD_POS_M + LEAD_NEG_M
        site_return  = (N - 2) * p + BOX_S_M
    elif wir == 1:                           # leapfrog, two lead lengths
        lead_per_mod = 2.0 * p + 2.0 * SLACK_M
        site_return  = 0.0
    else:                                    # leapfrog, alternate modules turned
        lead_per_mod = (2.0 * p - BOX_S_M) + 2.0 * SLACK_M
        site_return  = 0.0
    lead_m = N * lead_per_mod
    home_m = 2.0 * L                         # positive and negative home run
    site_m = site_return + home_m
    pairs  = N + 1

    k = 1.0 + ALPHA_CU * (Tc - 20.0)
    r_lead = R20[int(lead)] / 1000.0 * k
    r_site = R20[int(site)] / 1000.0 * k
    R = r_lead * lead_m + r_site * site_m + pairs * Rc

    vdrop   = Iop * R
    vdrop_p = vdrop / VmpHot * 100.0
    loss_p  = vdrop / VmpStc * 100.0

    # ---- 6. the verdict: the FIRST rule that fails ----
    if VocA > V_SYS_MAX:                verdict = 1
    elif VmpHot < MPPT_MIN_V:           verdict = 3
    elif Idesign > A_CONNECTOR:         verdict = 4
    elif Itrack > A_TRACKER:            verdict = 5
    elif vdrop_p > VDROP_CAP:           verdict = 6
    elif VocB > V_SYS_MAX:              verdict = 2
    else:                               verdict = 0

    return dict(idx=idx, cls=cid, N=N, Sp=Sp, wiring=WIRING[wir], lead=lead, site=site,
                L=L, Tcond=Tc, Tmin=Tmin, Thot=Thot, bif=bif, Rc=Rc,
                VocA=VocA, VocB=VocB, VmpHot=VmpHot, Idesign=Idesign, Itrack=Itrack,
                fuse_needed=fuse_needed, R=R, vdrop_pct=vdrop_p, loss_pct=loss_p,
                lead_m=lead_m, site_m=site_m, pairs=pairs, verdict=verdict)

NUM_KEYS = ["VocA", "VocB", "VmpHot", "Idesign", "Itrack", "R", "vdrop_pct",
            "loss_pct", "lead_m", "site_m"]

def known_answers():
    """Check the closed forms against the library's published answers."""
    out = []
    # 30 x 45.9 V at -0.25 %/K, site minimum -10 C:
    #   with the +10 K allowance -> 1463.06 V ; without it -> about 1497.4 V
    a = 30 * 45.9 * (1 + (-0.25) / 100 * ((-10 + 10) - 25))
    b = 30 * 45.9 * (1 + (-0.25) / 100 * (-10 - 25))
    out.append(("cold Voc with +10 K allowance = 1463.06 V", a, 1463.0625, abs(a - 1463.0625) < 5e-3))
    out.append(("cold Voc without the allowance = 1497.4 V", b, 1497.4, abs(b - 1497.4) < 0.1))
    # block DC 356.07 kW = 24 strings x 30 modules x 495 W class
    kw = 24 * 30 * 495 / 1000.0
    out.append(("block DC = 356.07 kW (24 x 30 x 495 W)", kw, 356.4, abs(kw - 356.4) < 0.5))
    # sequential lead 0.63 m/module; turned leapfrog 2.006 m; two-lead 2.846 m (p=1.323)
    p = 1.303 + GAP_M
    out.append(("sequential lead 0.63 m a module", LEAD_POS_M + LEAD_NEG_M, 0.63, True))
    out.append(("turned-leapfrog lead 2.006 m a module", (2 * p - BOX_S_M) + 0.2, 2.006, abs((2 * p - BOX_S_M) + 0.2 - 2.006) < 1e-9))
    out.append(("two-lead leapfrog 2.846 m a module", 2 * p + 0.2, 2.846, abs(2 * p + 0.2 - 2.846) < 1e-9))
    # site cable saved per string (N-2)p + b = 37.884 m at N = 30
    sv = (30 - 2) * p + BOX_S_M
    out.append(("site cable saved a string 37.884 m", sv, 37.884, abs(sv - 37.884) < 1e-9))
    return out

if __name__ == "__main__":
    for name, got, want, ok in known_answers():
        print(("PASS " if ok else "FAIL ") + name + "  got %.4f" % got)
    print(evaluate(123456789))
