"""Is the card the bottleneck, or is feeding it?

The claim to test: the kernel is already blistering, and the real constraint is
supplying it with deterministic work. If that is true, the arithmetic should be
a small fraction of a pass, and the time should sit in manufacturing indices
and reducing results.

Times each stage separately with the device synchronised between them, so the
numbers are stages and not an accident of asynchrony.

    python feeding.py
"""
import sys
import time

try:
    import cupy as cp
except Exception as exc:                                   # pragma: no cover
    print("cupy unavailable: %r" % (exc,))
    sys.exit(1)

from fused import FUSED, SIZE, tables


def timed(fn, reps=5):
    """Median of reps, each synchronised. Never a single sample."""
    out = []
    for _ in range(reps):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        out.append(time.perf_counter() - t0)
    out.sort()
    return out[len(out) // 2]


def main():
    free, total = cp.cuda.Device(0).mem_info
    batch = 1 << 23
    t = tables(cp)
    print("one pass of %s cases, %.2f GB free of %.2f GB\n"
          % ("{:,}".format(batch), free / 1e9, total / 1e9))

    idx = cp.arange(0, batch, dtype=cp.int64)
    codes = FUSED(idx, *t)
    cp.cuda.Stream.null.synchronize()

    stages = [
        ("manufacture the index (cp.arange)",
         lambda: cp.arange(0, batch, dtype=cp.int64)),
        ("the arithmetic (fused kernel)",
         lambda: FUSED(idx, *t)),
        ("reduce on the card (bincount)",
         lambda: cp.bincount(codes, minlength=4)),
        ("whole pass, as run() does it",
         lambda: cp.bincount(FUSED(cp.arange(0, batch, dtype=cp.int64), *t),
                             minlength=4)),
    ]
    res = {}
    for name, fn in stages:
        res[name] = timed(fn)

    whole = res["whole pass, as run() does it"]
    print("  %-38s %9s  %6s" % ("stage", "seconds", "share"))
    print("  " + "-" * 56)
    for name, _fn in stages:
        s = res[name]
        share = "" if name.startswith("whole") else "%5.1f%%" % (100 * s / whole)
        print("  %-38s %9.6f  %6s" % (name, s, share))

    arith = res["the arithmetic (fused kernel)"]
    feed = res["manufacture the index (cp.arange)"]
    red = res["reduce on the card (bincount)"]
    print()
    print("  arithmetic is %.1f%% of a pass; feeding and reducing are %.1f%%"
          % (100 * arith / whole, 100 * (feed + red) / whole))
    print("  at this rate one pass of %s cases costs %.4f s, so a billion"
          % ("{:,}".format(batch), whole))
    print("  cases cost about %.2f s and ten billion about %.1f s."
          % (whole * 1e9 / batch, whole * 1e10 / batch))
    print()
    print("  THE SPACE WE ACTUALLY HAVE: %s cases, which this card finishes"
          % "{:,}".format(SIZE))
    print("  in %.4f s. A working day of the card at this rate is %.2e cases."
          % (whole * SIZE / batch, 8 * 3600 * batch / whole))
    print()
    print("  Not stated: watts per case on the processor side, which was never")
    print("  recorded. These are stage times on one machine, median of 5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
