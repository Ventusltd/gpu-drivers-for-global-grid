"""The east-west reference block, swept on the engine.

The owner's block: 5 modules in portrait per plane, east-west, 30 in series,
24 strings on one string inverter. This sweeps the space around it so the
block can be moved rather than admired - which is the point of typing a system
instead of choosing a preset.

R1 and the block arithmetic are ours (cuda-maths/solar-billion/FORMULA.txt and
MODULE-CLASSES.md). No maker, no site name, anywhere.

    python east_west_block.py
"""
import sys

from ventus_engine import Engine, EngineError, Space

# ---- what we own. Voc, beta %/K, Vmp, gamma %/K, Wp, width mm
CLASSES = [
    (45.90, -0.25, 38.10, -0.34, 660.0, 1303.0),   # T660
    (50.59, -0.22, 42.58, -0.24, 760.0, 1303.0),   # W760
    (53.27, -0.25, 44.67, -0.29, 650.0, 1134.0),   # J-650
]
TMIN = [-14, -12, -11, -10, -8, -6]
SERIES = list(range(20, 35))          # 15
STRINGS = list(range(16, 33))         # 17
N_VOC_TOL = 101                       # Voc +/- 1%
INVERTER_KVA = 352.0
MAX_SYSTEM_V = 1500.0
DCAC_LO, DCAC_HI = 1.15, 1.40         # the band a UK east-west site considers

# verdict codes
OK, DISPUTED, FAIL_COLD, OUT_OF_BAND = 0, 1, 2, 3
LABELS = ["buildable, both readings, in band",
          "DISPUTED: the clause alone decides it",
          "fails cold voltage on both readings",
          "cold voltage fine, DC:AC outside the band"]


def kernel(xp, axes):
    ci, ti, si, ni, vt = axes
    voc = xp.asarray([c[0] for c in CLASSES])[ci]
    beta = xp.asarray([c[1] for c in CLASSES])[ci]
    wp = xp.asarray([c[4] for c in CLASSES])[ci]
    tmin = xp.asarray([float(t) for t in TMIN])[ti]
    series = xp.asarray([float(s) for s in SERIES])[si]
    strings = xp.asarray([float(n) for n in STRINGS])[ni]
    voc = voc * (1.0 + (vt - (N_VOC_TOL // 2)) / (N_VOC_TOL // 2) * 0.01)

    # R1, both readings, never collapsed
    nmax_b = xp.floor(MAX_SYSTEM_V /
                      (voc * (1.0 + beta / 100.0 * (tmin - 25.0))))
    nmax_a = xp.floor(MAX_SYSTEM_V /
                      (voc * (1.0 + beta / 100.0 * (tmin + 10.0 - 25.0))))
    ok_b, ok_a = series <= nmax_b, series <= nmax_a

    # the block
    dcac = (series * wp / 1000.0) * strings / INVERTER_KVA
    in_band = (dcac >= DCAC_LO) & (dcac <= DCAC_HI)

    code = xp.where(
        ~ok_a, FAIL_COLD,
        xp.where(~ok_b, DISPUTED,
                 xp.where(in_band, OK, OUT_OF_BAND)))
    return code.astype(xp.int64)


def closed_forms(np, space, kern):
    """The owner's four figures, asserted before the sweep runs."""
    wp, series, strings = 660.0, 30, 24
    assert abs(series * wp / 1000.0 - 19.80) < 1e-9
    assert abs(series * wp / 1000.0 * strings - 475.20) < 1e-9
    assert strings * 2 == 48
    assert abs(475.20 / INVERTER_KVA - 1.35) < 1e-3
    # T660, -10 C strict: N_max 30. At -11 C: 29. Verified by hand.
    i = [np.array([0]), np.array([3]), np.array([SERIES.index(30)]),
         np.array([STRINGS.index(24)]), np.array([N_VOC_TOL // 2])]
    assert int(kern(np, i)[0]) == OK, "the reference block must be buildable"
    i[1] = np.array([2])                       # -11 C
    assert int(kern(np, i)[0]) == DISPUTED, "-11 C must be the disputed band"
    return True


def main():
    space = Space([len(CLASSES), len(TMIN), len(SERIES), len(STRINGS),
                   N_VOC_TOL],
                  ["module_class", "site_tmin", "series", "strings",
                   "voc_tolerance"])
    eng = Engine(space, kernel, bytes_per_case=112)
    eng.assert_closed_forms(closed_forms)
    v = eng.verify(n=200_000)
    r = eng.run(nbins=4)
    print(eng.report(r, v, label="east-west block"))
    print()
    tot = sum(r.totals)
    for code, n in enumerate(r.totals):
        print("  %-42s %12,d  %5.2f%%".replace(",", ",")
              % (LABELS[code], n, 100.0 * n / tot) if False else
              "  %-42s %12s  %5.2f%%" % (LABELS[code], "{:,}".format(n),
                                         100.0 * n / tot))
    print()
    print("  THE REFERENCE BLOCK: 5 in portrait, east-west, 30 in series,")
    print("  24 strings -> 19.80 kW DC a string, 475.20 kW DC an inverter,")
    print("  1.350 DC:AC against a %.0f kVA class, 48 mated pairs inverter" %
          INVERTER_KVA)
    print("  side. Buildable to -10 degC on the strict reading; at -11 degC")
    print("  and colder the clause alone decides it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
