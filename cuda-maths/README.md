# cuda-maths — where the card earns its place, measured

Lane opened 20 September 2026. Three pieces of arithmetic were taken to the
GPU and checked against a CPU reference that was itself checked against
published answers. Every rate below was measured on the date given, on one
machine, on a card that was **shared with other work at the time**. None of
them is a property of the hardware in general.

Card: 16 GB, sm_120, 300 W class. CPU: 24 threads, 32 GB.
**CPU watts were never measured** — no wall meter, no RAPL read. Any
"per watt" figure below is GPU-side only, and is stated as such.

---

## 1. solar-billion — the one that gets it all right

A complete string-and-block design space: **1,020,600,000 designs**, mixed
radix over 12 axes.

| implementation | rate | a billion takes |
|---|---:|---|
| scalar reference | 419,526 cases/s | 0.66 hours |
| NumPy × 12 workers | 12,767,553 cases/s | 78.3 s |
| **CuPy, batch 8,388,608** | **177,289,777 cases/s** | **5.6 s** |

**GPU over 12 CPU workers: 13.89×.** The billion was run for real, not
extrapolated.

**Correctness, which is the only reason the rate means anything:**
7 of 7 published library answers reproduced by the scalar reference. Then
100,000 cases on a coprime stride of 1,000,003: reference vs NumPy **0 differ**,
reference vs GPU **0 differ**, packed best-design words NumPy vs GPU **0 differ**,
worst relative difference on 10 numeric outputs **0.00e+00** against a 1e-9
tolerance. All three implementations equal.

GPU during the run: 129 W mean, 147 W peak, 62% utilisation, 13,029 MiB.
1,378,695 cases per watt-second.

Why this one works: the reduction happens **on the card** (`gpu_vec.py`) so only
about 20 kB returns per batch, and the batch is sized from free VRAM at run
time rather than hard-coded.

Files: `solar-billion/` — `cpu_ref.py` (scalar reference), `cpu_vec.py`
(vectorised, multiprocess), `gpu_vec.py` (CuPy), `main.py`, plus `FORMULA.txt`
and `RESULTS.txt` with the full verdict counts.

## 2. layout-bench — real, but only half the card

The layout scorer, every term implemented, none omitted.

| | rate |
|---|---:|
| CPU, 8 workers, existing search | 630,553 evals/s |
| NumPy, one core | 51,932 evals/s |
| CuPy, batch 4,096 | 68,219 evals/s |
| CuPy, batch 16,384 | 278,748 evals/s |
| CuPy, batch 65,536 | 1,093,221 evals/s |
| **CuPy, batch 262,144** | **3,343,329 evals/s** |

**GPU over the 8-core CPU: 5.30×**, verified 2000/2000 integer-equal on the
same coprime stride.

**A claim that did not survive contact:** "tens of millions a second" was
**not reached** — 3.0× short. But 68k → 279k → 1.09M → 3.34M across batch
4k → 256k is a **49× gain from batch size alone**, which is the signature of
kernel-launch overhead, not of arithmetic. The ceiling was never found; the
largest batch took card use to 11.4 GB and the contending work cut the curve
off there.

GPU: 122 W mean, 132 W peak, 46% utilisation, 27,494 evals per watt-second.

## 3. precision-sense — is float64 the limit? No.

The same kernel run in float32 over the **whole** 1,020,600,000-case surface:
**verdicts changed: 0 (0.000000%)**. 5.14 s in float32 against 5.6 s in
float64 — only about **9% faster**.

So FP64 throughput is **not** what holds the card at 62%. What is left is
memory traffic: roughly 22 float64 temporaries per case is ~176 B/case, about
31 GB/s of DRAM round trips spread over many small elementwise kernels.

**The headroom is in fusing the case kernel into one pass, not in precision.**
That is the next piece of work, and it is named here so it is not rediscovered.

Also measured in this lane: a tolerance stack over 16 enumerated corners
(21,760 corner evaluations) and a monotonicity check over 41,943,040 neighbour
pairs across five properties — **0 violations**, no counterexample to report.

---

## What this lane does not claim

- No figure here describes any hardware other than the one machine named above.
- No CPU power figure exists, so no CPU-versus-GPU efficiency ratio is stated.
- The card was shared during every run. Where that capped a measurement, it is
  said in the results file rather than quietly corrected for.
- A rate is only reported for arithmetic that was verified equal to the
  reference first. Unverified throughput is not a result.

## What was removed, and why it is worth recording

A local-model lab ran on the same card until 20 September 2026: 15 proposals,
0 accepted, about 168 W average — **zero work per watt-second**. It also held
around 14.6 GB of the 16 GB card while the benchmarks above were being taken,
which is why the layout-bench curve stops where it does. It has been deleted.
The arithmetic in this lane is what the card is for.
