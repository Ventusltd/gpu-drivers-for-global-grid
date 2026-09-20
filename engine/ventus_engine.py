"""Ventus engine - our own GPU arithmetic engine. No model, no vendor, no LLM.

The intelligence is ours: closed forms we wrote, enumerated over a case space
we define, on whichever backend is present. The card is a multiplier on our
arithmetic, never a source of answers.

Written 2026-09-20 from three pieces that were already proven on this machine:
  - reduce ON the card so a batch returns ~20 kB, not gigabytes
  - size the batch from FREE device memory at run time, never a constant
  - a rate is not a result until the arithmetic has been verified equal to a
    reference implementation

House rules enforced in code, not by discipline:
  * a run over zero cases FAILS. A check that reached nothing is not a pass.
  * a rate cannot be reported for arithmetic that has not been verified.
  * every rate carries the conditions it was taken under.
  * closed forms are asserted BEFORE the sweep, so the sweep can only confirm
    or break them, never discover them.

    from ventus_engine import Backend, Space, Engine
    eng = Engine(Space([7, 11, 13]), kernel)
    eng.assert_closed_forms(my_checks)
    v = eng.verify(n=100_000)          # cpu vs gpu on a coprime stride
    r = eng.run()                      # the whole space
    print(eng.report(r, v))
"""
from __future__ import annotations

import math
import time

__all__ = ["Backend", "Space", "Engine", "EngineError", "Verification", "Run"]


class EngineError(RuntimeError):
    """Raised when a house rule is broken, never a warning."""


# --------------------------------------------------------------- the backend
class Backend:
    """numpy, or cupy when a device is present and actually runs a kernel."""

    def __init__(self, prefer="auto"):
        import numpy
        self.np = numpy
        self.xp = numpy
        self.name = "numpy"
        self.device = None
        if prefer in ("auto", "gpu"):
            try:
                import cupy
                cupy.arange(8, dtype=cupy.float64).sum()   # prove it runs
                self.xp = cupy
                self.name = "cupy"
                self.device = cupy.cuda.Device(0)
            except Exception as exc:
                if prefer == "gpu":
                    raise EngineError("gpu asked for, not available: %r" % exc)
        elif prefer != "cpu":
            raise EngineError("prefer must be auto, gpu or cpu")

    @property
    def is_gpu(self):
        return self.name == "cupy"

    def mem(self):
        """(free, total) device bytes, or (None, None) on the CPU."""
        if not self.is_gpu:
            return None, None
        return self.device.mem_info

    def free_batch(self, bytes_per_case, share=0.25, cap=1 << 23, floor=1 << 12):
        """Batch sized from FREE memory now - never a hard-coded constant."""
        if bytes_per_case <= 0:
            raise EngineError("bytes_per_case must be positive")
        if not self.is_gpu:
            return min(cap, 1 << 20)
        free, _ = self.mem()
        n = int(free * share / bytes_per_case)
        return max(floor, min(cap, n))

    def to_host(self, a):
        return self.xp.asnumpy(a) if self.is_gpu else self.np.asarray(a)

    def conditions(self):
        free, total = self.mem()
        c = {"backend": self.name}
        if self.is_gpu:
            c["device_free_gb"] = round(free / 1e9, 2)
            c["device_total_gb"] = round(total / 1e9, 2)
            c["device_used_gb"] = round((total - free) / 1e9, 2)
            c["card_shared"] = (total - free) > 1.5e9
        c["cpu_watts_measured"] = False      # stated plainly, always
        return c


# ------------------------------------------------------------- the case space
class Space:
    """A mixed-radix case space. Case i decodes to one value per axis."""

    def __init__(self, radices, names=None):
        radices = [int(r) for r in radices]
        if not radices or any(r < 1 for r in radices):
            raise EngineError("every radix must be >= 1")
        self.radices = radices
        self.names = list(names) if names else [
            "axis%d" % i for i in range(len(radices))]
        if len(self.names) != len(radices):
            raise EngineError("names and radices differ in length")
        self.size = math.prod(radices)

    def decode(self, xp, idx):
        """Vectorised: idx -> list of per-axis integer arrays."""
        out, rest = [], idx
        for r in self.radices:
            out.append(rest % r)
            rest = rest // r
        return out

    def coprime_stride(self):
        """A stride that walks the whole space without repeating."""
        s = 1000003
        while math.gcd(s, self.size) != 1:
            s += 2
        return s

    def __repr__(self):
        return "Space(%s, size=%d)" % (self.radices, self.size)


# ------------------------------------------------------------------- results
class Verification:
    def __init__(self, n, differ, worst, seconds, ref, test):
        self.n, self.differ, self.worst = n, differ, worst
        self.seconds, self.ref, self.test = seconds, ref, test

    @property
    def passed(self):
        return self.n > 0 and self.differ == 0

    def __repr__(self):
        return "Verification(%s vs %s, n=%d, differ=%d, worst=%.2e, %s)" % (
            self.ref, self.test, self.n, self.differ, self.worst,
            "PASS" if self.passed else "FAIL")


class Run:
    def __init__(self, cases, seconds, totals, conditions, batch):
        self.cases, self.seconds = cases, seconds
        self.totals, self.conditions, self.batch = totals, conditions, batch

    @property
    def rate(self):
        return self.cases / self.seconds if self.seconds > 0 else float("inf")


# -------------------------------------------------------------------- engine
class Engine:
    """Maps one kernel over one space, reducing on the device."""

    def __init__(self, space, kernel, bytes_per_case=176, prefer="auto"):
        if not callable(kernel):
            raise EngineError("kernel must be callable(xp, axes) -> codes")
        self.space, self.kernel = space, kernel
        self.bytes_per_case = bytes_per_case
        self.backend = Backend(prefer)
        self._closed_forms_asserted = False

    # ---- closed forms first, always
    def assert_closed_forms(self, checks):
        """checks(xp, space, kernel) must raise or return True. Run BEFORE any
        sweep, so the sweep can only confirm the arithmetic."""
        if checks(self.backend.np, self.space, self.kernel) is not True:
            raise EngineError("closed forms did not return True")
        self._closed_forms_asserted = True
        return True

    def _codes(self, xp, idx):
        axes = self.space.decode(xp, idx)
        codes = self.kernel(xp, axes)
        if codes.shape != idx.shape:
            raise EngineError("kernel returned %s for %s inputs"
                              % (codes.shape, idx.shape))
        return codes

    def _reduce(self, xp, codes, nbins):
        """Reduce ON the device: only nbins numbers cross the bus."""
        return xp.bincount(codes.astype(xp.int64), minlength=nbins)[:nbins]

    def run(self, nbins=16, limit=None, batch=None):
        if not self._closed_forms_asserted:
            raise EngineError("assert_closed_forms() must pass before run()")
        xp = self.backend.xp
        total = self.space.size if limit is None else min(limit, self.space.size)
        if total <= 0:
            raise EngineError("a run over zero cases fails: nothing was reached")
        batch = batch or self.backend.free_batch(self.bytes_per_case)
        totals = xp.zeros(nbins, dtype=xp.int64)
        done, t0 = 0, time.perf_counter()
        while done < total:
            n = min(batch, total - done)
            idx = xp.arange(done, done + n, dtype=xp.int64)
            totals += self._reduce(xp, self._codes(xp, idx), nbins)
            done += n
        if self.backend.is_gpu:
            self.backend.xp.cuda.Stream.null.synchronize()
        secs = time.perf_counter() - t0
        host = [int(v) for v in self.backend.to_host(totals)]
        if sum(host) != total:
            raise EngineError("reduction lost cases: %d of %d"
                              % (sum(host), total))
        return Run(total, secs, host, self.backend.conditions(), batch)

    def verify(self, n=100_000, against="cpu"):
        """Same cases, both backends, on a coprime stride through the space."""
        if n <= 0:
            raise EngineError("a verification of zero cases fails")
        ref = Backend(against)
        stride = self.space.coprime_stride()
        t0 = time.perf_counter()
        idx_r = ref.xp.arange(n, dtype=ref.xp.int64) * stride % self.space.size
        a = ref.to_host(self._codes_with(ref, idx_r))
        xp = self.backend.xp
        idx_t = xp.arange(n, dtype=xp.int64) * stride % self.space.size
        b = self.backend.to_host(self._codes_with(self.backend, idx_t))
        differ = int((a != b).sum())
        worst = float(abs(a.astype("float64") - b.astype("float64")).max()) \
            if n else 0.0
        return Verification(n, differ, worst, time.perf_counter() - t0,
                            ref.name, self.backend.name)

    def _codes_with(self, backend, idx):
        axes = self.space.decode(backend.xp, idx)
        return self.kernel(backend.xp, axes)

    # ---- a rate is not a result until it has been verified
    def report(self, run, verification, label="run"):
        if not isinstance(verification, Verification):
            raise EngineError("report() requires a Verification")
        if not verification.passed:
            raise EngineError(
                "refusing to report a rate: verification %s (n=%d, differ=%d)"
                % ("failed" if verification.n else "reached nothing",
                   verification.n, verification.differ))
        if run.cases <= 0:
            raise EngineError("refusing to report a run over zero cases")
        c = run.conditions
        lines = [
            "%s  %s" % (label.upper(), self.space),
            "  verified   %s vs %s, %d cases on stride %d: %d differ, worst "
            "%.2e" % (verification.ref, verification.test, verification.n,
                      self.space.coprime_stride(), verification.differ,
                      verification.worst),
            "  cases      %d (denominator: the whole space is %d)"
            % (run.cases, self.space.size),
            "  seconds    %.3f" % run.seconds,
            "  rate       {:,.0f} cases/s{}".format(
                run.rate,
                "   NOT A THROUGHPUT: the space is smaller than one batch, so "
                "this is setup time" if run.cases < run.batch else ""),
            "  batch      %d, sized from free memory at run time" % run.batch,
            "  backend    %s" % c["backend"],
        ]
        if c["backend"] == "cupy":
            lines.append("  device     %.2f GB free of %.2f GB%s"
                         % (c["device_free_gb"], c["device_total_gb"],
                            "  CARD SHARED AT RUN TIME"
                            if c.get("card_shared") else ""))
        lines.append("  cpu watts  NOT MEASURED - no wall meter, no RAPL. "
                     "No efficiency ratio is stated.")
        lines.append("  totals     %s" % run.totals)
        return "\n".join(lines)
