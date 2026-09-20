"""Self-test for the Ventus engine: our arithmetic, both backends, verified.

The kernel is R1 from the solar formula - the cold-voltage limit - because it
is arithmetic we own and whose answers are already published in
cuda-maths/solar-billion/FORMULA.txt. Nothing here consults a model.

    python selftest.py
"""
import sys

from ventus_engine import Backend, Engine, EngineError, Space

# module classes we own (MODULE-CLASSES.md): Voc, beta %/K, Vmp
CLASSES = [(45.90, -0.25, 38.10),    # T660
           (50.59, -0.22, 42.58),    # W760
           (53.27, -0.25, 44.67)]    # J-650
TMIN = [-14, -12, -11, -10, -8, -6]  # site minimum, degC
SERIES = list(range(20, 35))         # modules in series
MAX_SYSTEM_V = 1500.0
# the tolerance stack, the same two axes the precision-sense work swept
N_VOC_TOL, N_BETA_TOL = 1001, 101    # Voc +/-1%, beta +/-10%
VOC_TOL_NOMINAL, BETA_TOL_NOMINAL = N_VOC_TOL // 2, N_BETA_TOL // 2


def kernel(xp, axes):
    """R1: N_max = floor(1500 / (Voc * (1 + beta/100 * (Tdesign - 25)))).

    Returns a verdict code: 0 buildable on BOTH readings, 1 buildable only
    with the +10 K allowance (the disputed band), 2 buildable on neither.
    """
    ci, ti, si, vt, bt = axes
    voc = xp.asarray([c[0] for c in CLASSES])[ci]
    beta = xp.asarray([c[1] for c in CLASSES])[ci]
    tmin = xp.asarray([float(t) for t in TMIN])[ti]
    series = xp.asarray([float(s) for s in SERIES])[si]
    # tolerance: Voc +/-1% in 1001 steps, beta +/-10% in 101 steps
    voc = voc * (1.0 + (vt - (N_VOC_TOL // 2)) / (N_VOC_TOL // 2) * 0.01)
    beta = beta * (1.0 + (bt - (N_BETA_TOL // 2)) / (N_BETA_TOL // 2) * 0.10)

    voc_b = voc * (1.0 + beta / 100.0 * (tmin - 25.0))          # strict
    voc_a = voc * (1.0 + beta / 100.0 * (tmin + 10.0 - 25.0))   # +10 K
    nmax_b = xp.floor(MAX_SYSTEM_V / voc_b)
    nmax_a = xp.floor(MAX_SYSTEM_V / voc_a)

    ok_b = series <= nmax_b
    ok_a = series <= nmax_a
    return xp.where(ok_b, 0, xp.where(ok_a, 1, 2)).astype(xp.int64)


def closed_forms(np, space, kern):
    """Assert the arithmetic BEFORE any sweep, on cases computed by hand."""
    # T660 at -10 C, strict reading: Voc_cold 49.92 V, N_max 30
    voc = 45.90 * (1 + (-0.25) / 100 * (-10 - 25))
    assert abs(voc - 49.9163) < 1e-3, voc
    assert int(1500 // voc) == 30, 1500 // voc
    # and at -11 C it drops to 29, which is the whole point
    voc11 = 45.90 * (1 + (-0.25) / 100 * (-11 - 25))
    assert int(1500 // voc11) == 29, 1500 // voc11
    # the kernel must agree with those two by hand, at nominal tolerance
    nom_v = np.array([VOC_TOL_NOMINAL, VOC_TOL_NOMINAL])
    nom_b = np.array([BETA_TOL_NOMINAL, BETA_TOL_NOMINAL])
    ci = np.array([0, 0]); ti = np.array([3, 2]); si = np.array([10, 10])
    assert TMIN[3] == -10 and TMIN[2] == -11 and SERIES[10] == 30
    codes = kern(np, [ci, ti, si, nom_v, nom_b])
    assert int(codes[0]) == 0, "30 at -10 C must be buildable on both"
    assert int(codes[1]) == 1, "30 at -11 C must be the disputed band"
    # a worse Voc tolerance must never make a design MORE buildable
    one = np.array([0])
    for vt in (0, VOC_TOL_NOMINAL, N_VOC_TOL - 1):
        c = int(kern(np, [one, np.array([3]), np.array([10]),
                          np.array([vt]), np.array([BETA_TOL_NOMINAL])])[0])
        assert c in (0, 1, 2)
        if vt == N_VOC_TOL - 1:          # the +1% corner
            assert c >= 1, "at +1% Voc, 30 at -10 C must stop being safe"
    return True


def main():
    space = Space([len(CLASSES), len(TMIN), len(SERIES),
                   N_VOC_TOL, N_BETA_TOL],
                  ["module_class", "site_tmin", "modules_in_series",
                   "voc_tolerance", "beta_tolerance"])
    print("space:", space, "\n")

    results = {}
    for prefer in ("cpu", "auto"):
        try:
            eng = Engine(space, kernel, bytes_per_case=96, prefer=prefer)
        except EngineError as exc:
            print("skip %s: %s" % (prefer, exc)); continue
        eng.assert_closed_forms(closed_forms)
        v = eng.verify(n=min(100_000, space.size * 7))
        r = eng.run(nbins=3)
        print(eng.report(r, v, label=eng.backend.name)); print()
        results[eng.backend.name] = (r, v)

    if "cupy" in results and "numpy" in results:
        g, c = results["cupy"][0], results["numpy"][0]
        assert g.totals == c.totals, (g.totals, c.totals)
        print("BOTH BACKENDS AGREE on all %d verdicts: %s" %
              (sum(g.totals), g.totals))
        print("speed-up cupy over numpy: %.2fx" % (g.rate / c.rate))

    print("\n--- the house rules are enforced, not merely stated ---")
    eng = Engine(space, kernel, prefer="cpu")
    try:
        eng.run()
    except EngineError as exc:
        print("  run before closed forms  -> refused:", exc)
    eng.assert_closed_forms(closed_forms)
    try:
        eng.run(limit=0)
    except EngineError as exc:
        print("  run over zero cases      -> refused:", exc)
    try:
        from ventus_engine import Verification
        eng.report(eng.run(nbins=3), Verification(0, 0, 0.0, 0.0, "a", "b"))
    except EngineError as exc:
        print("  report with no evidence  -> refused:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
