# engine — our own GPU arithmetic engine

No model. No vendor. No LLM. The intelligence is the closed forms we wrote;
the card is a multiplier on our arithmetic and never a source of answers.

Opened 20 September 2026, the day the local model lab was deleted. It
generalises the three things that were already proven on this machine:

1. **reduce on the card** — a batch returns a handful of numbers, not gigabytes
2. **size the batch from free device memory at run time**, never a constant
3. **a rate is not a result until the arithmetic is verified against a
   reference implementation**

## Use

```python
from ventus_engine import Engine, Space

eng = Engine(Space([3, 6, 15, 1001, 101]), my_kernel)
eng.assert_closed_forms(my_checks)   # BEFORE any sweep
v   = eng.verify(n=100_000)          # cpu vs gpu, coprime stride
r   = eng.run(nbins=3)               # the whole space
print(eng.report(r, v))
```

A kernel is written once, against `xp`, and runs on either backend:

```python
def kernel(xp, axes):
    ...
    return xp.where(ok, 0, 1).astype(xp.int64)
```

## The house rules are enforced in code, not by discipline

| rule | what happens |
|---|---|
| a check that reached nothing is not a pass | `run(limit=0)` raises |
| the sweep may only confirm the arithmetic | `run()` raises until `assert_closed_forms()` passes |
| a rate needs evidence | `report()` raises without a passing `Verification` |
| a rate needs a real measurement | a space smaller than one batch is labelled **NOT A THROUGHPUT** |
| state the denominator | every report prints cases against the whole space |
| CPU watts were never measured | every report says so, and no efficiency ratio is stated |
| the card is often shared | the report flags `CARD SHARED AT RUN TIME` from live memory |

The reduction is also checked: if the bins do not sum to the number of cases
run, the engine raises rather than reporting a total that lost work.

## First result — `selftest.py`

R1, the cold-voltage limit, over module class × site minimum temperature ×
modules in series × Voc tolerance (±1%) × β tolerance (±10%).

```
space   27,297,270 cases
numpy        7,507,811 cases/s   3.636 s
cupy       306,440,267 cases/s   0.089 s      40.82x
verified   numpy vs cupy, 100,000 cases on stride 1,000,003: 0 differ
totals     [15,100,997 both readings | 1,180,231 disputed | 11,016,042 neither]
```

Taken on a 16 GB card with 14.61 GB free, flagged shared. CPU watts not
measured, so no efficiency ratio is stated.

**The finding, not the speed:** 1,180,231 of 27,297,270 designs — 4.3% — are
buildable on one reading of the cold-voltage clause and not on the other. The
clause alone decides them. That number is why the disputed reading is never
collapsed to one answer anywhere in this work.

The 40.82× is not comparable with the 13.89× in `cuda-maths/solar-billion`:
this kernel is simpler and does less per case. Speed-ups belong to a kernel,
not to a card.

## Requires

`numpy` always; `cupy` optional — without it the engine runs on the CPU and
says so. Nothing else. No network call is made by any part of this engine.
